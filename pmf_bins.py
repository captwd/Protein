"""
共享的距离分桶定义 —— 与 new/new.py 的 bin 配置逐字对齐。
=============================================================

new.py 中（见 new/new.py 的 BIN_CONFIG / make_bin_edges）:
    BIN_CONFIG = [(2.0, 20.0, 0.5)]
    trRosetta / AlphaFold 标准 37-bin:
      2-20Å     宽 0.5Å  (36 bin)
      >20Å      1 个 far bin（第 37 个，记为 FAR_BIN=36）

数据集侧（Rosettosa npz 的 `dist` 字段）语义:
    dist == 0   → far 哨兵（真实距离 >= 20Å 的对全部编码成 0.0）→ FAR_BIN
    dist > 0    → 真实 Cβ-Cβ 距离（Gly 用 Cα 兜底），走正常分桶

本模块用 numpy 复刻同一套 edges/centers 与归属规则，保证统计 PMF 时 bin 不会分错。
归属规则复刻 new.py 的:
    torch.bucketize(dist, edges, right=True) - 1，再 clamp 到 [0, 36]
其中 right=True 即 np.searchsorted(edges, dist, side='right')。
（far 哨兵 0.0 → FAR_BIN 的映射是"数据集语义"，只在新.py 的目标转换里做，
   本模块的 dist_to_bin_idx 保持纯几何映射。）

对齐验证：运行 test_bin_alignment.py，会解析 new.py 源码里的 BIN_CONFIG / AA_ORDER
并与本模块逐一比对，还会用 torch 在随机距离上比对归属结果。
"""
import numpy as np

# ---------------------------------------------------------------
# 氨基酸顺序 —— 必须与 new.py 完全一致
# 顺序是决定 PMF 矩阵 (37, 20, 20) 第 1、2 维语义的唯一依据。
# ---------------------------------------------------------------
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

# ---------------------------------------------------------------
# Bin 配置 —— 必须与 new.py 的 BIN_CONFIG 完全一致
# ---------------------------------------------------------------
BIN_CONFIG = [
    (2.0, 20.0, 0.5),
]
FAR_BIN = 36            # 第 37 个 bin（0 起编号）: >20Å
FAR_BIN_CENTER = 25.0   # far bin 的期望距离中心（论文里要写明这个约定）


def make_bin_edges(config):
    """与 new.py 的 make_bin_edges 逻辑逐字一致，末尾追加 inf 表示 far bin。"""
    edges = []
    for start, end, width in config:
        edges.extend([start + i * width for i in range(int((end - start) / width))])
    edges.append(end)
    edges.append(np.inf)          # far bin 上界
    return np.asarray(edges, dtype=np.float64)


BIN_EDGES = make_bin_edges(BIN_CONFIG)          # (38,) 最后一个是 inf
N_BINS = len(BIN_EDGES) - 1                      # 37
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0   # (37,)
BIN_CENTERS[-1] = FAR_BIN_CENTER                # far bin 显式中心（(20+inf)/2 是 inf）
BIN_WIDTHS = BIN_EDGES[1:] - BIN_EDGES[:-1]             # (37,)
BIN_WIDTHS[-1] = np.inf


def dist_to_bin_idx(dist, bin_edges=None):
    """
    距离 -> bin 索引，复刻 new.py 的 dist_to_bin_idx 语义：
        torch.bucketize(dist, bin_edges, right=True) - 1,  clamp 到 [0, n_bins-1]

    dist : 标量或 (N,) 数组
    返回 : 同形状 int64 数组
    注意 : 纯几何映射。数据集 far 哨兵（dist==0 → FAR_BIN）由 new.py 的目标转换处理。
    """
    edges = BIN_EDGES if bin_edges is None else np.asarray(bin_edges, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)
    idx = np.searchsorted(edges, dist, side='right') - 1
    return np.clip(idx, 0, len(edges) - 2)


if __name__ == "__main__":
    print(f"N_BINS = {N_BINS}  (FAR_BIN={FAR_BIN}, 中心={FAR_BIN_CENTER})")
    print(f"BIN_EDGES[:5]   = {BIN_EDGES[:5].round(3)}")
    print(f"BIN_EDGES[-3:]  = {BIN_EDGES[-3:]}")
    print(f"BIN_CENTERS[-3:]= {BIN_CENTERS[-3:].round(3)}")
    # 抽查典型距离的归属
    for d in [0.0, 2.0, 2.1, 3.8, 7.9, 8.0, 12.0, 19.75, 20.0, 25.0, 48.0, 1e9]:
        print(f"dist {d:8.1f} -> bin {dist_to_bin_idx(d):3d} (center {BIN_CENTERS[dist_to_bin_idx(d)]:6.2f})")
