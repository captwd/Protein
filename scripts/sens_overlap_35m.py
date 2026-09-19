"""35M OOD 内容效应的同源重叠敏感性分析（纯重分析，无训练/无推理）。

d_t = mean_seed(real_t) - mean_{perm,seed}(shuffled_t)
对比：全部 target  vs  剔除与 PMF 语料有 >=X% 同源命中的 target。
判定阈值取 homolog_hits.csv 的 n_id>=50 / n_id>=90 / n_id=100 列。
"""
import csv
import numpy as np
from collections import defaultdict

BASE = r"D:\PycharmProjects\extensions\new1"

# ---- 1. 逐 target 分数 ----
real = defaultdict(dict)      # (bench, target) -> {seed: score}
shuf = defaultdict(list)      # (bench, target) -> [scores over perm x seed]
with open(BASE + r"\hard_target_scores.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        if row["backbone"] != "ESM2-35M":
            continue
        key = (row["benchmark"], row["target_id"])
        if row["condition"] == "phys35m" and row["permutation"] == "real":
            real[key][int(row["seed"])] = float(row["p_at_l5"])
        elif row["condition"].startswith("pert35m_p"):
            shuf[key].append(float(row["p_at_l5"]))

d = defaultdict(dict)         # bench -> {target: d_t}
for key in real:
    bench, tgt = key
    if len(real[key]) and shuf.get(key):
        r = np.mean(list(real[key].values()))
        s = np.mean(shuf[key])
        d[bench][tgt] = r - s

# ---- 2. 同源命中清单 ----
hits = {}                     # (bench, target) -> dict(n50, n90, n100)
with open(BASE + r"\homolog_hits.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        hits[(row["test_set"], row["target"])] = (
            int(row["n_id>=50"]), int(row["n_id>=90"]), int(row["n_id=100"]))


def stats(vals):
    v = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(0)
    boots = [rng.choice(v, len(v), replace=True).mean() for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    t = m / se if se > 0 else np.nan
    try:
        from scipy.stats import ttest_1samp
        p = ttest_1samp(v, 0.0).pvalue
    except ImportError:
        p = np.nan
    return m, lo, hi, t, p, len(v)


def report(bench, drop_pred, label):
    tgts = [t for t in d[bench] if not (drop_pred and drop_pred(t))]
    m, lo, hi, t, p, n = stats([d[bench][t] for t in tgts])
    tag = "all        " if drop_pred is None else label
    print(f"  {bench:8s} {tag} n={n:3d}  d={100*m:+6.2f} pp "
          f"[{100*lo:+.2f}, {100*hi:+.2f}]  t={t:.2f}  p={p:.4f}")


print("35M target 级内容效应：剔除与 PMF 语料同源 target 的敏感性")
for bench in ["casp14", "cameo", "casp15"]:
    tg = list(d[bench])
    n90 = sum(1 for t in tg if hits.get((bench, t), (0, 0, 0))[1] >= 1)
    n50 = sum(1 for t in tg if hits.get((bench, t), (0, 0, 0))[0] >= 1)
    print(f"\n[{bench}]  高同源命中: >=90%: {n90} 个 target, >=50%: {n50} 个")
    report(bench, None, "")
    report(bench, lambda t, b=bench: hits.get((b, t), (0, 0, 0))[1] >= 1,
           "drop >=90% ")
    report(bench, lambda t, b=bench: hits.get((b, t), (0, 0, 0))[0] >= 1,
           "drop >=50% ")
    report(bench, lambda t, b=bench: hits.get((b, t), (0, 0, 0))[2] >= 1,
           "drop =100%  ")
