"""ESM2-650M official zero-shot contact head — same protocol as the 35M reference row."""
import os, sys, csv
import numpy as np
import torch

sys.path.insert(0, r"D:\PycharmProjects\extensions\new")
import new as N

DEV = "cpu"  # 650M contact head needs L^2 attention APC: too big for 8GB GPU
NEW1 = r"D:\PycharmProjects\extensions\new1"

import esm
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.eval().to(DEV)
batch_converter = alphabet.get_batch_converter()

def eval_set(name, npz_dir):
    ids = sorted(f[:-4] for f in os.listdir(npz_dir) if f.endswith(".npz"))
    rows = []
    for pid in ids:
        x = np.load(os.path.join(npz_dir, pid + ".npz"), allow_pickle=True)
        seq, dist = str(x["seq"]), x["dist"].astype(np.float32)
        L = len(seq)
        if L > 1000:
            continue
        data = [(pid, seq)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(DEV)
        with torch.no_grad():
            out = model(tokens, return_contacts=True)
        contacts = out["contacts"][0].float().cpu().numpy()[:L, :L]
        tc = ((dist > 0) & (dist < 8.0)).astype(np.float32)
        p5 = N.calc_precision(-contacts, tc)["P@L/5"]
        p1 = N.calc_precision(-contacts, tc, k_list=[1])["P@L/1"]
        rows.append(dict(set=name, target=pid, L=L, p_at_l5=p5, p_at_l1=p1))
        del out
        torch.cuda.empty_cache()
    arr5 = [r["p_at_l5"] for r in rows]
    arr1 = [r["p_at_l1"] for r in rows]
    print(f"[{name}] n={len(rows)}  P@L/5={np.mean(arr5):.4f}  P@L/1={np.mean(arr1):.4f}", flush=True)
    return rows

all_rows = []
all_rows += eval_set("casp14", os.path.join(NEW1, "test_sets", "casp14"))
all_rows += eval_set("casp15", os.path.join(NEW1, "test_sets", "casp15"))
all_rows += eval_set("cameo", os.path.join(NEW1, "test_sets", "cameo"))

out = os.path.join(NEW1, "esm2_650m_zeroshot_reference.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    w.writeheader(); w.writerows(all_rows)
print("saved ->", out)
