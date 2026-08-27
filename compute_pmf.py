"""
从 PDB 文件集统计残基对距离统计势（PMF）。
=============================================

输入
  --pdb-dir       PDB 文件目录（.pdb/.ent，文件名 = 小写 pdb id，如 12as.pdb）
  --chain-file    可选：download_pisces.py 生成的 "PDBID CHAIN" 清单。
                  未提供时处理目录内所有 PDB 的所有链。
  --exclude-file  可选：要排除的 PDB id 或 "pdb chain" 清单
                  （同源 / 泄漏过滤，例如 PDNET 测试集、CASP 目标）
  --min-seq-sep   最小序列间隔（默认 2：排除 i,i+1 共价相邻对，
                  其距离 ~3.8Å 由键长决定，会制造无信息量的假峰）
  --pseudo-count  逆玻尔兹曼伪计数 λ（默认 1.0，避免 0 概率）
  --smooth        bin 轴高斯平滑 sigma（默认 0 = 不平滑）
  --max-files     只处理前 N 个 PDB 文件（冒烟测试用）

输出（--out，默认 pmf.npz）
  pmf            (N_BINS,20,20) float32 统计势 E = -log[ P(r|aa,aa) / P(r) ]  （37-bin Cβ）
  counts         (N_BINS,20,20) int64 原始对称计数（不含伪计数）
  pair_counts    (20,20)   每对氨基酸总计数
  bin_counts     (N_BINS,) 每个 bin 总计数
  bin_edges / bin_centers / bin_widths   与 new.py 完全一致
  n_structures / n_chains / n_pairs
  aa_order

符号约定：E 越低表示该 (aa 对, 距离 bin) 越有利。
new.py 的 attention bias 中使用的是 -pair_energy（见 new.py），接入时注意符号。

用法:
  python compute_pmf.py --pdb-dir pdbs --chain-file pdbs/cull_chains.txt --out pmf.npz
  python compute_pmf.py --pdb-dir pdbs --chain-file pdbs/cull_chains.txt --max-files 200
"""
import argparse
import glob
import os
import sys

# Windows 控制台默认 GBK，强制 UTF-8 输出避免 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1_extended as three_to_one

from pmf_bins import AA_ORDER, AA_TO_IDX, N_BINS, BIN_EDGES, BIN_CENTERS, BIN_WIDTHS, dist_to_bin_idx

PDB_PARSER = PDBParser(QUIET=True)


def load_exclude(exclude_file):
    """返回 (set_of_pdbids, set_of_(pdb, chain))。"""
    pdbs, pairs = set(), set()
    if not exclude_file:
        return pdbs, pairs
    with open(exclude_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if not toks:
                continue
            pdb = toks[0].lower()
            if len(pdb) >= 4 and pdb[:4].isalnum():
                pdb = pdb[:4]
            pdbs.add(pdb)
            if len(toks) > 1 and toks[1].isalnum():
                pairs.add((pdb, toks[1].lower()))
    return pdbs, pairs


def load_chain_map(chain_file):
    """返回 {pdb: set(chain_lower)}；chain 可能为 ''（= 整文件）。"""
    cmap = {}
    if not chain_file:
        return None
    with open(chain_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if not toks:
                continue
            pdb = toks[0].lower()[:4]
            chain = toks[1].lower() if len(toks) > 1 else ""
            cmap.setdefault(pdb, set()).add(chain)
    return cmap


def atom_name(rn, atom):
    """距离图原子约定（trRosetta）: 默认 Cβ；Gly 无 Cβ 用 Cα 兜底。"""
    if atom == "ca":
        return "CA"
    return "CA" if rn == "GLY" else "CB"


def extract_atoms(chain, atom="cb"):
    """提取一条链的可信残基 -> (res_num, aa_idx, coord) 三元组列表。
    跳过: 杂原子/水 (res.id[0] != ' ')、负/零残基号、非标准氨基酸、缺目标原子。"""
    out = []
    for res in chain:
        if res.id[0] != " " or res.id[1] <= 0:
            continue
        rn = res.resname.strip()
        one = three_to_one.get(rn)          # 三位代码 -> 单字母（含 MSE->M 等）
        if one is None or one not in AA_TO_IDX:
            continue                          # 非标准氨基酸跳过
        aname = atom_name(rn, atom)
        if aname not in res:
            continue
        out.append((res.id[1], AA_TO_IDX[one], res[aname].get_coord()))
    return out


def process_chain(chain, min_seq_sep, atom="cb"):
    """对单链累积 counts，返回 (counts, n_pairs)。"""
    data = extract_atoms(chain, atom)
    L = len(data)
    counts = np.zeros((N_BINS, 20, 20), dtype=np.int64)
    if L < min_seq_sep + 2:
        return counts, 0
    coords = np.array([d[2] for d in data], dtype=np.float64)      # (L,3)
    aa_idx = np.array([d[1] for d in data], dtype=np.int64)        # (L,)
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)  # (L,L)
    dist = np.sqrt(d2)
    i, j = np.triu_indices(L, k=min_seq_sep)                       # j - i >= min_seq_sep
    dist_flat = dist[i, j]
    aai, aaj = aa_idx[i], aa_idx[j]
    b = dist_to_bin_idx(dist_flat)
    flat_idx = b * 400 + aai * 20 + aaj                            # 400 = 20*20
    flat_counts = np.bincount(flat_idx, minlength=N_BINS * 400).astype(np.int64)
    counts = flat_counts.reshape(N_BINS, 20, 20)
    counts = counts + counts.transpose(0, 2, 1)                    # 对称累积
    return counts, len(dist_flat)


def build_pmf(counts, pseudo=1.0, smooth=0.0):
    """计数 -> 逆玻尔兹曼统计势。"""
    if smooth > 0:
        try:
            from scipy.ndimage import gaussian_filter1d
            counts = gaussian_filter1d(counts.astype(np.float64), sigma=smooth, axis=0)
        except ImportError:
            print("[warn] scipy 不可用，跳过平滑")
    counts_s = counts.astype(np.float64) + pseudo
    pair_counts = counts_s.sum(axis=0)        # (20,20)
    bin_counts = counts_s.sum(axis=(1, 2))    # (N_BINS,)
    N = bin_counts.sum()
    P_ref = bin_counts / N
    P_cond = counts_s / pair_counts[None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        E = -np.log(P_cond / P_ref[:, None, None])
    E = np.nan_to_num(E, nan=0.0, posinf=4.0, neginf=-4.0)
    E = np.clip(E, -4.0, 4.0)
    E = (E + E.transpose(0, 2, 1)) / 2.0
    return E.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--chain-file", default=None)
    parser.add_argument("--exclude-file", default=None)
    parser.add_argument("--min-seq-sep", type=int, default=2)
    parser.add_argument("--pseudo-count", type=float, default=1.0)
    parser.add_argument("--smooth", type=float, default=0.0)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--out", default="pmf.npz")
    parser.add_argument("--atom", type=str, default="cb", choices=["ca", "cb"],
                        help="距离原子: cb(Cβ,Gly用Cα兜底,默认) / ca(Cα)")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.pdb_dir, "*.pdb")) +
                   glob.glob(os.path.join(args.pdb_dir, "*.ent")))
    if not files:
        parser.error(f"{args.pdb_dir} 下没有 .pdb/.ent 文件")
    if args.max_files:
        files = files[: args.max_files]

    exclude_pdbs, exclude_pairs = load_exclude(args.exclude_file)
    chain_map = load_chain_map(args.chain_file)

    total_counts = np.zeros((N_BINS, 20, 20), dtype=np.int64)
    total_pairs, n_structures, n_chains = 0, 0, 0
    skipped = 0

    print(f"处理 {len(files)} 个 PDB 文件 (min_seq_sep={args.min_seq_sep}) ...")
    for k, path in enumerate(files, 1):
        base = os.path.splitext(os.path.basename(path))[0].lower()
        if exclude_pdbs and base in exclude_pdbs:
            skipped += 1
            continue
        try:
            structure = PDB_PARSER.get_structure("x", path)
        except Exception as e:
            skipped += 1
            continue
        model = structure[0]
        for chain in model:
            cid = chain.id.strip().lower()
            if chain_map is not None:
                want = chain_map.get(base, set())
                if want and cid not in want and "" not in want:
                    continue
            if (base, cid) in exclude_pairs:
                continue
            c_counts, n_pairs = process_chain(chain, args.min_seq_sep, args.atom)
            if n_pairs:
                total_counts += c_counts
                total_pairs += n_pairs
                n_chains += 1
        n_structures += 1
        if k % 200 == 0 or k == len(files):
            print(f"       {k}/{len(files)}  chains={n_chains} pairs={total_pairs:,}")

    if total_pairs == 0:
        parser.error("没有任何有效残基对！请检查链清单/格式")

    pmf = build_pmf(total_counts, pseudo=args.pseudo_count, smooth=args.smooth)

    np.savez_compressed(
        args.out,
        pmf=pmf,                              # (N_BINS,20,20) E = -log[P_cond/P_ref]
        counts=total_counts,                  # 原始计数
        pair_counts=total_counts.sum(axis=0),
        bin_counts=total_counts.sum(axis=(1, 2)),
        bin_edges=BIN_EDGES.astype(np.float32),
        bin_centers=BIN_CENTERS.astype(np.float32),
        bin_widths=BIN_WIDTHS.astype(np.float32),
        n_structures=np.array([n_structures]),
        n_chains=np.array([n_chains]),
        n_pairs=np.array([total_pairs]),
        aa_order=AA_ORDER,
        atom=args.atom,
    )

    peak_bin = int(total_counts.sum(axis=(1, 2)).argmax())
    print("\n=========== 统计报告 ===========")
    print(f"文件数      : {n_structures} (跳过 {skipped})")
    print(f"有效链数    : {n_chains}")
    print(f"残基对总数  : {total_pairs:,}")
    print(f"每个 bin 平均计数: {total_pairs / N_BINS:,.0f}")
    print(f"最密集 bin  : bin {peak_bin} 中心 {BIN_CENTERS[peak_bin]:.2f}Å")
    print(f"非零 bin 数 : {(total_counts.sum(axis=(1,2)) > 0).sum()} / {N_BINS}")
    print(f"PMF 范围    : [{pmf.min():.2f}, {pmf.max():.2f}]")
    print(f"PMF 非零单元: {(pmf != 0).sum()} / {N_BINS * 400}")
    print(f"\n已保存: {args.out}")
    print(f"  接入 new.py 时注意: attention bias 使用 -pmf（见 new.py pair_energy 用法）")


if __name__ == "__main__":
    main()
