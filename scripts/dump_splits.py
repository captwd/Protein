# -*- coding: utf-8 -*-
"""
重建训练/验证/测试划分的 ID 账本（splits.csv）——数据可追溯性。
================================================================
精确复现 new.py 的 trrosetta 分支划分逻辑（new.py L1808-L1823）：

    ids  = sorted(glob("<npz>/*.npz"))            # 文件名排序，池 = 15049
    rng  = np.random.default_rng(seed)            # 独立 generator，与全局种子无关
    rng.shuffle(ids)                              # seed 固定 -> 乱序固定
    valid      = ids[:100]                        # valid = 100
    test_cameo = ids[100:200]                     # held-out test = 100（log 里 test=100）
    pool       = ids[200:]
    train_{s}  = pool[:s]                         # 500/1500/3000 嵌套前缀（500⊂1500⊂3000）

并复现 TRRosettaDataset 的可用性过滤（new.py L1153-L1191）标记 usable：
  mask1d.sum() >= 20，过滤后长度与 fasta 子序列一致，20 <= L <= 1024，形状合法。
usable 只依赖蛋白本身，与 seed 无关，故只算一次。

注意：trRosetta 原始 npz 与 fasta 不随仓库重分发。本脚本是"再生成/审计"工具，
需要你本地有一份 trRosetta 数据副本，运行前用 --npz-dir / --fasta 指定。

输出：<repo-root>/splits.csv 长格式账本（seed, split, train_size, order, protein_id, usable）。
用法：
    python scripts/dump_splits.py \
        --npz-dir /path/to/trRosetta/npz \
        --fasta  /path/to/trRosetta/15051.fasta
    python scripts/dump_splits.py --npz-dir ... --fasta ... --no-check   # 只出划分，不加载 npz
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT_CSV = os.path.join(REPO_ROOT, "splits.csv")
VALID_N, TEST_N = 100, 100
SIZES = [500, 1500, 3000]
SEEDS = [0, 1, 2]


def load_fasta(path):
    """同 new.py load_fasta（L1039）。"""
    fasta = {}
    cur, buf = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur:
                    fasta[cur] = "".join(buf)
                cur, buf = line[1:].strip(), []
            else:
                buf.append(line)
    if cur:
        fasta[cur] = "".join(buf)
    return fasta


def is_usable(pid, fasta, npz_dir):
    """同 TRRosettaDataset.__init__ 的过滤（new.py L1153-L1191）。"""
    try:
        x = np.load(os.path.join(npz_dir, f"{pid}.npz"), allow_pickle=True)
        mask = x["mask1d"].astype(bool)
        dmap = x["dist"].astype(np.float32)
    except Exception:
        return False
    seq_full = fasta.get(pid, "")
    if mask.sum() < 20:
        return False
    seq_s = "".join(c for c, m in zip(seq_full, mask) if m)
    dmap = dmap[mask][:, mask]           # 与 new.py 一致：先掩码再取形状
    L = dmap.shape[0]
    if L != len(seq_s) or not (20 <= L <= 1024):
        return False
    if dmap.shape != (L, L):
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz-dir", required=True,
                    help="trRosetta npz 目录（含 15049 个 .npz，splits.csv 的池来源）")
    ap.add_argument("--fasta", required=True,
                    help="trRosetta 全长 fasta（15051.fasta，usable 过滤用）")
    ap.add_argument("--no-check", action="store_true",
                    help="跳过 usable 计算（只出划分，不加载 npz）")
    ap.add_argument("--out", default=OUT_CSV, help="输出 CSV 路径（默认 <repo-root>/splits.csv）")
    args = ap.parse_args()

    ids_all = sorted(os.path.splitext(os.path.basename(f))[0]
                     for f in glob.glob(os.path.join(args.npz_dir, "*.npz")))
    fasta = load_fasta(args.fasta)
    print(f"[splits] pool={len(ids_all)} npz, fasta={len(fasta)}  "
          f"(log 对照 total=15049)")

    usable_map = {}
    if args.no_check:
        usable_map = {pid: -1 for pid in ids_all}   # -1 = 未检查
    else:
        import tqdm
        for pid in tqdm.tqdm(ids_all, desc="usability"):
            usable_map[pid] = int(is_usable(pid, fasta, args.npz_dir))

    rows = []
    for seed in SEEDS:
        ids = ids_all[:]                       # 拷贝！shuffle 是 in-place
        rng = np.random.default_rng(seed)      # 首次使用即 shuffle，与 new.py 一致
        rng.shuffle(ids)
        vlist, tlist, pool = ids[:VALID_N], ids[VALID_N:VALID_N + TEST_N], ids[VALID_N + TEST_N:]
        for order, pid in enumerate(vlist):
            rows.append([seed, "valid", "", order, pid, usable_map[pid]])
        for order, pid in enumerate(tlist):
            rows.append([seed, "test_cameo", "", order, pid, usable_map[pid]])
        for size in SIZES:
            for order, pid in enumerate(pool[:size]):
                rows.append([seed, "train", size, order, pid, usable_map[pid]])

    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "split", "train_size", "order", "protein_id", "usable"])
        w.writerows(rows)
    print(f"[splits] wrote {len(rows)} rows -> {args.out}")

    # 交叉核对：log 显示 500-train 有效 496、valid 98、test 100
    if not args.no_check:
        for seed in SEEDS:
            u_train = sum(r[5] for r in rows if r[0] == seed and r[1] == "train" and r[2] == 500)
            u_val = sum(r[5] for r in rows if r[0] == seed and r[1] == "valid")
            u_test = sum(r[5] for r in rows if r[0] == seed and r[1] == "test_cameo")
            print(f"  seed{seed}: train500 usable={u_train}  valid usable={u_val}  test usable={u_test}")


if __name__ == "__main__":
    main()
