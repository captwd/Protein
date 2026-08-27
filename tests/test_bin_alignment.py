"""
验证 pmf_bins.py 与 new.py 的 bin 定义完全一致。
====================================================

不能直接 import new.py（它顶层会加载 ESM2 模型、触发 ModelScope 下载），
所以本测试只解析 new.py 源码中的 BIN_CONFIG / AA_ORDER 常量文本，
用同一套 make_bin_edges 逻辑重算后逐一比对，并在随机距离上
用 torch 复刻 new.py 的 dist_to_bin_idx 语义与本模块比对。

这验证的是最容易出现"悄悄不一致"的地方：
new.py 与 pmf_bins.py 的 BIN_CONFIG、氨基酸顺序、37-bin edges，
以及 torch.bucketize(..., right=True) 的归属语义。

用法（在仓库根目录运行）:
    python tests/test_bin_alignment.py            # --new-py 默认指向仓库根 new.py
    python tests/test_bin_alignment.py --new-py /path/to/new.py
"""
import os
import re
import ast
import argparse

import sys
import numpy as np

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)          # 使 `import pmf_bins` 能找到仓库根目录的 pmf_bins.py

import pmf_bins

# Windows 控制台默认 GBK，强制 UTF-8 输出避免 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_NEW_PY = os.path.join(REPO_ROOT, "new.py")


def extract_constants(new_py_path):
    with open(new_py_path, "r", encoding="utf-8") as f:
        src = f.read()
    m_bin = re.search(r"BIN_CONFIG\s*=\s*(\[[^\]]*\])", src)
    m_aa = re.search(r'AA_ORDER\s*=\s*"([A-Z]+)"', src)
    assert m_bin, "new.py 中未找到 BIN_CONFIG"
    assert m_aa, "new.py 中未找到 AA_ORDER"
    return ast.literal_eval(m_bin.group(1)), m_aa.group(1)


def make_bin_edges_ref(config):
    """与 new.py / pmf_bins 相同的实现，独立写一遍用于交叉验证（末尾追加 inf = far bin）。"""
    edges = []
    for start, end, width in config:
        edges.extend([start + i * width for i in range(int((end - start) / width))])
    edges.append(end)
    edges.append(np.inf)
    return np.asarray(edges, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-py", default=DEFAULT_NEW_PY)
    args = parser.parse_args()

    new_py_path = os.path.abspath(args.new_py)
    assert os.path.exists(new_py_path), f"找不到 new.py: {new_py_path}"
    print(f"[1/4] 解析 {new_py_path} ...")

    new_config, new_aa_order = extract_constants(new_py_path)
    print(f"      new.py BIN_CONFIG = {new_config}")
    print(f"      new.py AA_ORDER   = {new_aa_order}")

    # ---- 1. BIN_CONFIG 一致 ----
    assert pmf_bins.BIN_CONFIG == new_config, "BIN_CONFIG 不一致!"
    print("[2/4] BIN_CONFIG 一致 ✓")

    # ---- 2. AA_ORDER 一致 ----
    assert pmf_bins.AA_ORDER == new_aa_order, "AA_ORDER 不一致!"
    assert len(pmf_bins.AA_ORDER) == 20, "AA_ORDER 必须恰好 20 个氨基酸"
    print("[3/4] AA_ORDER 一致 ✓ (20 aa)")

    # ---- 3. edges/centers 一致 ----
    ref_edges = make_bin_edges_ref(new_config)
    assert len(ref_edges) - 1 == pmf_bins.N_BINS == 37, f"N_BINS 应为 37, 实为 {pmf_bins.N_BINS}"
    assert np.allclose(pmf_bins.BIN_EDGES, ref_edges, atol=1e-6), "BIN_EDGES 不一致!"
    assert np.isposinf(ref_edges[-1]), "far bin 上界应为 inf"
    print(f"      N_BINS = {pmf_bins.N_BINS} ✓ (FAR_BIN={pmf_bins.FAR_BIN}, 中心={pmf_bins.FAR_BIN_CENTER})")

    # ---- 4. 归属规则与 torch.bucketize(right=True) 语义一致 ----
    import torch
    edges_t = torch.tensor(ref_edges.tolist())          # float32，与 new.py 一致
    rng = np.random.RandomState(0)
    dists = np.concatenate([
        rng.uniform(0.0, 100.0, size=5000),
        np.array([0.0, 2.0, 2.1, 3.8, 7.9, 8.0, 12.0, 19.75, 20.0, 25.0, 47.0, 1e9]),
    ])
    d_t = torch.tensor(dists, dtype=torch.float32)
    torch_idx = (torch.bucketize(d_t, edges_t, right=True) - 1)
    torch_idx = torch_idx.clamp(0, len(edges_t) - 2).numpy()
    np_idx = pmf_bins.dist_to_bin_idx(dists)
    assert np.array_equal(np_idx, torch_idx), \
        f"归属不一致! 例: dists={dists[np_idx != torch_idx][:10]}, np={np_idx[np_idx != torch_idx][:10]}, torch={torch_idx[np_idx != torch_idx][:10]}"
    print(f"[4/4] 归属规则与 torch.bucketize(right=True) 完全一致 ✓ ({len(dists)} 个距离)")

    print("\n全部通过：bin 定义与 new.py 精确对齐。")


if __name__ == "__main__":
    main()
