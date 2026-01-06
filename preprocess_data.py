import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm


def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(data_name) as f:
        s = next(f)
        for idx, line in enumerate(tqdm(f, desc=f"Parsing {data_name}")):
            if idx > 100_000:
                break
            e = line.strip().split(",")
            u = int(e[0])
            i = int(e[1])

            ts = float(e[2])
            label = float(e[3])

            if data_name == "GDELT":
                feat = np.array([x == "1" for x in e[4:]])
            else:
                feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)
    return pd.DataFrame(
        {"u": u_list, "i": i_list, "ts": ts_list, "label": label_list, "idx": idx_list}
    ), np.array(feat_l)


def reindex(df, bipartite=True):
    new_df = df.copy()
    if bipartite:
        assert df.u.max() - df.u.min() + 1 == len(df.u.unique())
        assert df.i.max() - df.i.min() + 1 == len(df.i.unique())

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

    return new_df


def run(data_name, bipartite=True):
    Path("data/").mkdir(parents=True, exist_ok=True)
    PATH = "./data_raw/{}.csv".format(data_name)
    OUT_DF = "./data/ml_{}.csv".format(data_name)
    OUT_FEAT = "./data/ml_{}.npy".format(data_name)
    OUT_NODE_FEAT = "./data/ml_{}_node.npy".format(data_name)

    df, feat = preprocess(PATH)
    new_df = reindex(df, bipartite)

    empty = np.zeros(feat.shape[1])[np.newaxis, :].astype(feat.dtype)
    feat = np.vstack([empty, feat])

    max_idx = max(new_df.u.max(), new_df.i.max())
    rand_feat = np.zeros((max_idx + 1, 172))

    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)


def run_opted(data_name, bipartite=True):
    Path("data/").mkdir(parents=True, exist_ok=True)
    PATH = "./data_raw/{}.csv".format(data_name)
    OUT_DF = "./data/ml_{}.csv".format(data_name)
    OUT_FEAT = "./data/ml_{}.npy".format(data_name)
    OUT_NODE_FEAT = "./data/ml_{}_node.npy".format(data_name)

    num_edges = 0
    dim_efeat = 0

    # 1st scan
    with open(PATH) as f:
        s = next(f)  # skip the first line (col names)
        for idx, line in enumerate(tqdm(f, desc=f"Scanning {data_name}")):
            if dim_efeat == 0:
                e = line.strip().split(",")
                dim_efeat = len(e) - 4
            num_edges += 1
    pad_efeat = True if dim_efeat == 0 else False
    dim_efeat = 2 if pad_efeat else dim_efeat

    dtype_efeat = np.bool_ if data_name == "GDELT" else float
    efeat_mmap = np.lib.format.open_memmap(
        OUT_FEAT,
        mode="w+",
        dtype=dtype_efeat,
        shape=(num_edges + 1, dim_efeat),
    )
    # 2nd scan
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []
    cnt_error = 0
    invalid_lines = []
    with open(PATH) as f:
        s = next(f)  # skip the first line (col names)

        # Pad one empty row to make eid starts with 1 instead 0
        empty = np.zeros([dim_efeat])[np.newaxis, :].astype(dtype_efeat)
        efeat_mmap[0] = empty
        for idx, line in enumerate(tqdm(f, desc=f"Parsing {data_name}")):
            e = line.strip().split(",")
            try:
                u = int(e[0])
                i = int(e[1])

                ts = float(e[2])
                label = float(e[3])

                if data_name == "GDELT":
                    efeat_mmap[idx + 1] = np.fromiter(
                        (x == "1" for x in e[4:]),
                        dtype=dtype_efeat,
                        count=dim_efeat,
                    )
                else:
                    if pad_efeat:
                        efeat_mmap[idx + 1] = np.zeros([1], dtype=dtype_efeat)
                    else:
                        efeat_mmap[idx + 1] = np.fromiter(
                            (float(x) for x in e[4:]),
                            dtype=dtype_efeat,
                            count=dim_efeat,
                        )

                u_list.append(u)
                i_list.append(i)
                ts_list.append(ts)
                label_list.append(label)
                idx_list.append(idx)
            except ValueError:
                cnt_error += 1
                invalid_lines.append(line)
                continue
    df = pd.DataFrame(
        {"u": u_list, "i": i_list, "ts": ts_list, "label": label_list, "idx": idx_list}
    )

    print("Invalid and skipped lines: ", cnt_error)
    df_invalid = pd.DataFrame({"lines": invalid_lines})
    df_invalid.to_csv("invalid_lines.csv")

    print("Reindexing ...")
    new_df = reindex(df, bipartite)

    max_idx = max(new_df.u.max(), new_df.i.max())
    rand_feat = np.zeros((max_idx + 1, 172))

    print("Saving files ...")
    new_df.to_csv(OUT_DF)
    np.save(OUT_NODE_FEAT, rand_feat)


parser = argparse.ArgumentParser("Interface for TGN data preprocessing")
parser.add_argument(
    "--data", type=str, help="Dataset name (eg. wikipedia or reddit)", default="TemFin"
)
parser.add_argument(
    "--bipartite", action="store_true", help="Whether the graph is bipartite"
)

args = parser.parse_args()

# run(args.data, bipartite=args.bipartite)

run_opted(args.data, args.bipartite)
