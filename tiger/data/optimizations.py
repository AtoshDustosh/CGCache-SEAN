import torch
from torch import Tensor

import numpy as np
from collections import OrderedDict, deque
from typing import List, Tuple

from .graph import Graph

from torch.profiler import record_function
from enum import IntEnum


class AsyncSignal(IntEnum):
    PARTITION_SENT = 1
    CACHED_SENT = 2
    MISSED_SENT = 3
    START_UPDATE = 4
    UPDATE_FINISHED = 5


class CGCache:

    slots_nids: List[Tensor]
    slots_eids: List[Tensor]
    slots_tss: List[Tensor]
    offsets: Tensor

    def __init__(
        self,
        cap: int,
        n_neighbor: int,
        n_layer: int,
        graph: Graph,
        policy: str = "lru",
        device: torch.device = torch.device("cpu"),
    ):
        self.cap = cap
        self.policy = policy
        self.device = device

        self.L = n_layer
        self.k = n_neighbor
        self.graph = graph

        # Cache slots for all layers
        self.slots_nids = [torch.zeros([])]
        self.slots_eids = [torch.zeros([])]
        self.slots_tss = [torch.zeros([])]
        self.slots_res = [torch.zeros([])]
        self.slots_degs = [torch.zeros([])]
        self.offsets = torch.zeros([self.cap], device=self.device, dtype=torch.long)

        # Initialize cached layers: layer 1 -> layer L
        for l in range(1, self.L + 1):
            self.slots_nids.append(
                torch.zeros([self.cap, self.k**l], device=self.device, dtype=torch.long)
            )
            self.slots_eids.append(
                torch.zeros([self.cap, self.k**l], device=self.device, dtype=torch.long)
            )
            self.slots_tss.append(
                torch.zeros(
                    [self.cap, self.k**l], device=self.device, dtype=torch.float
                )
            )
            self.slots_res.append(
                torch.zeros(
                    [self.cap, self.k**l], device=self.device, dtype=torch.float
                )
            )
            self.slots_degs.append(
                torch.zeros(
                    [self.cap, self.k**l], device=self.device, dtype=torch.float
                )
            )

        # Cache slot index
        # We use a CPU dict because querying for a single batch is usually fast
        self.mapping = OrderedDict()  # map: nid -> cache slot
        self.free_slots = deque(range(self.cap))

        # Temporary objects to reduce redundant operations
        self.old_keys = list(self.mapping.keys())

        print(f"cg cache initialized with cap {self.cap}")
        pass

    def debug_msg(self) -> str:
        msg = f"cap: {self.cap}\n"
        msg += f"n_cached: {len(self.mapping)}\n"
        msg += f"n_free_slots: {len(self.free_slots)}\n"
        return msg

    @property
    def capacity(self) -> int:
        return self.cap

    @property
    def n_cached(self) -> int:
        ret = len(self.mapping)
        assert ret <= self.cap, "Unexpected cache behavior: exceeding capacity!"
        return ret

    def neg_sample(self, size):
        return np.random.choice(
            np.array(list(self.mapping.keys()), dtype=int), size=size, replace=True
        )

    def reset(self):
        """Clear cache."""
        self.mapping.clear()
        self.free_slots = deque(range(self.cap))
        pass

    def redo_NS(self, bs):
        # OPT A better in-cache NS strategy
        ns_cache = int((len(self.mapping) / self.capacity) * bs)
        ns_rand = bs - ns_cache
        ndsts_cache = np.random.choice(
            np.array(list(self.mapping.keys()), dtype=int), size=ns_cache, replace=True
        )
        ndsts_rand = np.random.randint(0, self.graph.num_node, ns_rand)
        ndsts = np.concatenate([ndsts_cache, ndsts_rand])
        return ndsts

    @torch.no_grad()
    def partition_batch(self, batch_nids: np.ndarray):
        """
        Partition the nodes in a batch into cached unique nodes and missed unique nodes.

        Args:
            batch_nids: [bs * 3] concatenation of [srcs, dsts, ndsts].
        Returns:
            n_unique: the number of unique nodes.
            uninids: [n_unique] unique nodes.
            mask_uni2cached: [n_unique] mask from unique nodes to unique cached nodes.
            mask_uni2missed: [n_unique] inverse of mask_unique2cached.
            index_uni2batch: [bs * 3] inverse index from unique nodes to batch nodes.
            unicounts: [n_unique] the counts of each unique nodes.
        """

        uninids, index_uni2batch, unicounts = np.unique(
            batch_nids, return_inverse=True, return_counts=True
        )
        n_unique = len(uninids)

        self.old_keys = list(self.mapping.keys())

        mask_uni2cached = np.isin(uninids, self.old_keys, assume_unique=True)
        mask_uni2missed = ~mask_uni2cached

        return (
            n_unique,
            uninids,
            mask_uni2cached,
            mask_uni2missed,
            index_uni2batch,
            unicounts,
        )

    @torch.no_grad()
    def fetch_cached(self, unicached_nids: np.ndarray):
        """
        Extract cgs of unique cached nodes.

        Args:
            unicached_nids: [n_cached] unique nodes.
        Returns:
            cached_layers: L + 1 layers of [nids, eids, tss]
        """
        cached_layers: List = [[] for _ in range(self.L + 1)]
        cached_slots = (
            torch.from_numpy(
                np.fromiter(
                    (self.mapping[nid] for nid in unicached_nids),
                    dtype=int,
                    count=len(unicached_nids),
                )
            )
            .long()
            .to(self.device)
        )

        # Initialize the 0-th layer
        t_cached_nids = torch.from_numpy(unicached_nids).long().to(self.device)
        cached_layers[0] = [
            t_cached_nids,  # nid
            torch.tensor([]),  # eid
            torch.tensor([]),  # ts
            torch.tensor([]),  # re
            torch.tensor([]),  # deg
        ]

        # Gather other L layers of CGs
        for l in range(1, self.L + 1):
            # This should be much faster than enumerating cached nids
            index = (
                torch.arange(start=0, end=self.k**l, device=self.device).unsqueeze(0)
                + (self.offsets[cached_slots] * self.k ** (l - 1)).unsqueeze(1)
            ) % (self.k**l)

            # "L + 1 - l" is used to match TIGER's twisted cg design
            cached_layers[self.L + 1 - l] = [
                self.slots_nids[l][cached_slots].gather(1, index).view(-1, self.k),
                self.slots_eids[l][cached_slots].gather(1, index).view(-1, self.k),
                self.slots_tss[l][cached_slots].gather(1, index).view(-1, self.k),
                self.slots_res[l][cached_slots].gather(1, index).view(-1, self.k),
                self.slots_degs[l][cached_slots].gather(1, index).view(-1, self.k),
            ]
            pass
        return cached_layers

    @torch.no_grad()
    def exec_eviction(
        self,
        srcs: np.ndarray,
        dsts: np.ndarray,
        ndsts: np.ndarray,
        unimissed: np.ndarray,
        ignore_ns: bool = True,
    ):
        """
        Given batch nids, update the cache index with pre-defined cache policy.

        Args:
            bs: batch size.
            batch_nids: [bs * 3] concatenation of [srcs, dsts, ndsts].
            uni2missed: unique missed nids.
            ignore_ns: whether to ignore negative samples when doing eviction.
        Returns:
            mask_unimissed_put: mask from unique missed to put.
            unimissed_put_nids: which unique missed nids' layers should be put in cache.
        """
        evicted_mapping = []

        if self.policy == "lru":
            for src, dst, ndst in zip(srcs, dsts, ndsts, strict=True):
                nids = (src, dst) if ignore_ns else (src, dst, ndst)
                for nid in nids:
                    if nid in self.mapping:
                        self.mapping.move_to_end(nid)
                    else:
                        if self.free_slots:
                            slot_id = self.free_slots.pop()
                        else:
                            evicted_nid, slot_id = self.mapping.popitem(last=False)
                            if evicted_nid in self.old_keys:
                                evicted_mapping.append((evicted_nid, slot_id))
                        self.mapping[nid] = slot_id
                    pass
                pass
        else:
            raise NotImplementedError

        reordered_nids = []
        old_slot_ids = []
        new_slot_ids = []
        for nid, slot_id in evicted_mapping:
            if nid in self.mapping:
                reordered_nids.append(nid)
                old_slot_ids.append(slot_id)
                new_slot_ids.append(self.mapping[nid])

        index_old = torch.tensor(old_slot_ids, dtype=torch.long, device=self.device)
        index_new = torch.tensor(new_slot_ids, dtype=torch.long, device=self.device)

        evicted_offsets = self.offsets.index_select(0, index_old)
        self.offsets.index_copy_(0, index_new, evicted_offsets)
        for l in range(1, self.L + 1):
            evicted_nids = self.slots_nids[l].index_select(0, index_old)
            evicted_eids = self.slots_eids[l].index_select(0, index_old)
            evicted_tss = self.slots_tss[l].index_select(0, index_old)
            self.slots_nids[l].index_copy_(0, index_new, evicted_nids)
            self.slots_eids[l].index_copy_(0, index_new, evicted_eids)
            self.slots_tss[l].index_copy_(0, index_new, evicted_tss)

            # SEAN-specific
            evicted_res = self.slots_res[l].index_select(0, index_old)
            evicted_degs = self.slots_degs[l].index_select(0, index_old)
            self.slots_res[l].index_copy_(0, index_new, evicted_res)
            self.slots_degs[l].index_copy_(0, index_new, evicted_degs)

        mask_unimissed_put = np.zeros_like(unimissed, dtype=np.bool_)
        for i, nid in enumerate(unimissed):
            if nid in self.mapping:
                mask_unimissed_put[i] = True

        return mask_unimissed_put, unimissed[mask_unimissed_put]

    @torch.no_grad()
    def put_missed(
        self,
        unimissed_layers: List,
        mask_unimissed_put: np.ndarray,
        unimissed_put_nids: np.ndarray,
    ):
        """
        Given batch nids, update the cache index with pre-defined cache policy.

        Args:
            unimissed_layers: cg layers of unique missed nodes.
            mask_unimissed_put: mask from unique missed to put.
            unimissed_put_nids: which unique missed nids' layers should be put in cache.
        """
        unimissed_put_slots = (
            torch.from_numpy(
                np.fromiter(
                    (self.mapping[nid] for nid in unimissed_put_nids),
                    dtype=int,
                    count=len(unimissed_put_nids),
                )
            )
            .long()
            .to(self.device)
        )
        t_mask_unimissed_put = torch.from_numpy(mask_unimissed_put).to(self.device)

        self.offsets.index_fill_(0, unimissed_put_slots, 0)

        for l in range(1, self.L + 1):
            self.slots_nids[l].index_copy_(
                0,
                unimissed_put_slots,
                # "L + 1 - l" is used to match TIGER's twisted cg design
                # ".view([-1, k**l])" is, again, because of TIGER's design for CGs.
                # layer 0: [600], layer 1: [600, 10], layer 2: [6000, 10], ...
                # TIGER arranges timestamps from left to right (old to new)
                unimissed_layers[self.L + 1 - l][0].view([-1, self.k**l])[
                    t_mask_unimissed_put
                ],
            )
            self.slots_eids[l].index_copy_(
                0,
                unimissed_put_slots,
                unimissed_layers[self.L + 1 - l][1].view([-1, self.k**l])[
                    t_mask_unimissed_put
                ],
            )
            self.slots_tss[l].index_copy_(
                0,
                unimissed_put_slots,
                unimissed_layers[self.L + 1 - l][2].view([-1, self.k**l])[
                    t_mask_unimissed_put
                ],
            )
            self.slots_res[l].index_copy_(
                0,
                unimissed_put_slots,
                unimissed_layers[self.L + 1 - l][3].view([-1, self.k**l])[
                    t_mask_unimissed_put
                ],
            )
            self.slots_degs[l].index_copy_(
                0,
                unimissed_put_slots,
                unimissed_layers[self.L + 1 - l][4].view([-1, self.k**l])[
                    t_mask_unimissed_put
                ],
            )
            pass
        pass

    @torch.no_grad()
    def update_cached(
        self,
        srcs: np.ndarray,
        dsts: np.ndarray,
        eids: np.ndarray,
        tss: np.ndarray,
    ):
        """
        Update cached cgs with new batch data. Missed nodes are skipped and not updated.

        Args:
            srcs: [bs] source nodes.
            dsts: [bs] destination nodes.
            eids: [bs] event ids.
            tss: [bs] timestamps.
        """
        bound_tss = min(tss)

        # Step 1: collect "updatable" cache slots
        updatables = {}
        for _, (src, dst, eid, ts) in enumerate(zip(srcs, dsts, eids, tss)):
            # src -> dst
            if src in self.mapping:
                if src in updatables:
                    updatables[src][0].append(dst)
                    updatables[src][1].append(eid)
                    updatables[src][2].append(ts)
                else:
                    updatables[src] = [[], [], []]
                    updatables[src][0].append(dst)
                    updatables[src][1].append(eid)
                    updatables[src][2].append(ts)
            # dst -> src
            if dst in self.mapping:
                if dst in updatables:
                    updatables[dst][0].append(src)
                    updatables[dst][1].append(eid)
                    updatables[dst][2].append(ts)
                else:
                    updatables[dst] = [[], [], []]
                    updatables[dst][0].append(src)
                    updatables[dst][1].append(eid)
                    updatables[dst][2].append(ts)
            pass

        # Step 2:prepare the 0th and 1st layer (truncation + 0-padding)
        n_updated = len(updatables)
        all_updates_nids = np.zeros([n_updated * self.k], dtype=int)
        all_updates_eids = np.zeros([n_updated * self.k], dtype=int)
        all_updates_tss = np.zeros([n_updated * self.k], dtype=float)
        all_updates_res = np.zeros([n_updated * self.k], dtype=float)
        all_updates_degs = np.zeros([n_updated * self.k], dtype=float)
        updated_nids = []
        for idx, (updated_nid, updates) in enumerate(updatables.items()):
            len_overwrite = min(len(updates[0]), self.k)
            updates_nids, updates_eids, updates_tss = updates
            # Truncate unnecessary neighbors
            all_updates_nids[idx * self.k : idx * self.k + len_overwrite] = (
                updates_nids[-len_overwrite:]
            )
            all_updates_eids[idx * self.k : idx * self.k + len_overwrite] = (
                updates_eids[-len_overwrite:]
            )
            all_updates_tss[idx * self.k : idx * self.k + len_overwrite] = updates_tss[
                -len_overwrite:
            ]
            # SEAN's neighbor reoccurrences and node degree
            local_neighbors, _, _, _ = self.graph.find_before(updated_nid, bound_tss)
            _, inverse, cnts = np.unique(
                local_neighbors, return_inverse=True, return_counts=True
            )
            local_res = cnts[inverse].astype(np.float32)[-len_overwrite:]
            local_deg = len(local_neighbors)

            all_updates_res[
                idx * self.k : idx * self.k + min(len_overwrite, len(local_res))
            ] = local_res
            all_updates_degs[idx * self.k : idx * self.k + len_overwrite] = local_deg

            updated_nids.append(updated_nid)
        updated_nids = np.array(updated_nids)  # [n_updated]

        # Initialize the first two layers of incremental CGs
        updated_layers: List[Tuple] = [
            (updated_nids, np.array([]), np.array([]), np.array([]), np.array([]))
        ]
        updated_layers.append(
            (
                all_updates_nids,
                all_updates_eids,
                all_updates_tss,
                all_updates_res,
                all_updates_degs,
            )
        )

        # Step 3: sample and initialize higher layers (layer depth >= 2)
        for l in range(2, self.L + 1):
            layer_nids, layer_eids, layer_tss, _, layer_res, layer_degs = (
                self.graph.sample_temporal_neighbor(
                    updated_layers[l - 1][0], updated_layers[l - 1][2], self.k
                )
            )
            updated_layers.append(
                (layer_nids, layer_eids, layer_tss, layer_res, layer_degs)
            )
            pass

        # Convert to tensors
        for depth in range(len(updated_layers)):
            neigh_nids, neigh_eids, neigh_tss, neigh_res, neigh_degs = updated_layers[
                depth
            ]
            updated_layers[depth] = (
                torch.from_numpy(neigh_nids).to(self.device).long(),
                torch.from_numpy(neigh_eids).to(self.device).long(),
                torch.from_numpy(neigh_tss).to(self.device).float(),
                torch.from_numpy(neigh_res).to(self.device).float(),
                torch.from_numpy(neigh_degs).to(self.device).float(),
            )
            pass

        # Step 4: find cache slots
        updated_slots = (
            torch.from_numpy(
                np.fromiter(
                    (self.mapping[nid] for nid in updated_nids),
                    dtype=int,
                    count=len(updated_nids),
                )
            )
            .long()
            .to(self.device)
        )

        # Step 5: parallel update
        for l in range(1, self.L + 1):
            # Roll new data by their slots' offsets
            index = (
                torch.arange(start=0, end=self.k**l, device=self.device).unsqueeze(0)
                - (self.offsets[updated_slots] * self.k ** (l - 1)).unsqueeze(1)
            ) % (self.k**l)
            layer_nids, layer_eids, layer_tss, layer_res, layer_degs = updated_layers[l]
            layer_nids_rolled = layer_nids.view(-1, self.k**l).gather(1, index)
            layer_eids_rolled = layer_eids.view(-1, self.k**l).gather(1, index)
            layer_tss_rolled = layer_tss.view(-1, self.k**l).gather(1, index)
            layer_res_rolled = layer_res.view(-1, self.k**l).gather(1, index)
            layer_degs_rolled = layer_degs.view(-1, self.k**l).gather(1, index)

            # Merge old data with new data
            mask = layer_nids_rolled != 0
            merged_nids = torch.where(
                mask, layer_nids_rolled, self.slots_nids[l][updated_slots]
            )
            merged_eids = torch.where(
                mask, layer_eids_rolled, self.slots_eids[l][updated_slots]
            )
            merged_tss = torch.where(
                mask, layer_tss_rolled, self.slots_tss[l][updated_slots]
            )
            merged_res = torch.where(
                mask, layer_res_rolled, self.slots_res[l][updated_slots]
            )
            merged_degs = torch.where(
                mask, layer_degs_rolled, self.slots_degs[l][updated_slots]
            )

            # Replace old slots with merged data
            self.slots_nids[l].index_copy_(0, updated_slots, merged_nids)
            self.slots_eids[l].index_copy_(0, updated_slots, merged_eids)
            self.slots_tss[l].index_copy_(0, updated_slots, merged_tss)
            self.slots_res[l].index_copy_(0, updated_slots, merged_res)
            self.slots_degs[l].index_copy_(0, updated_slots, merged_degs)
            pass
        pass
