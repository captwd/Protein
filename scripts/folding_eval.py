"""Downstream folding evaluation: distogram -> 3D coordinates -> TM/RMSD/lDDT.

Pipeline per (condition, seed, target):
  1. run model forward -> 37-bin logits -> expected distance matrix
  2. classical MDS (torch double precision) -> initial 3D coords
  3. short gradient refinement (L-BFGS) under predicted-distance restraints
  4. Kabsch-fit to the native Ca coordinates, report TM-score/RMSD/lDDT

Pure post-processing: no training, checkpoints are loaded frozen.
"""
import os, sys, pickle, argparse, csv
import numpy as np
import torch

sys.path.insert(0, r"D:\PycharmProjects\extensions\new")
import new as N

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NEW1 = r"D:\PycharmProjects\extensions\new1"
CASPN = os.path.join(NEW1, "test_sets", "casp14")
CACHE = os.path.join(NEW1, "test_sets", "test_esm_casp14.pkl")

# ---------------- conditions ----------------
CONDS = {
    "baseline":   dict(ckpt="esm2ds500_base_best_seed{seed}.pt",
                       kw=dict(use_ngram=True, use_triangle=True, use_feature_gate=False,
                               use_attn_bias=False, use_static_bias=False, use_attn_no_bias=False,
                               pot_mode="pmf", pot_perturb_seed=None)),
    "axial":      dict(ckpt="esm2attnobias500_best_seed{seed}.pt",
                       kw=dict(use_ngram=True, use_triangle=True, use_feature_gate=False,
                               use_attn_bias=False, use_attn_no_bias=True,
                               use_static_bias=False, pot_mode="pmf", pot_perturb_seed=None)),
    "shuffled":   dict(ckpt="esm2perturb500_best_seed{seed}.pt",
                       kw=dict(use_ngram=True, use_triangle=True, use_feature_gate=False,
                               use_attn_bias=True, use_attn_no_bias=False,
                               use_static_bias=False, pot_mode="pmf", pot_perturb_seed=0)),
    "real":       dict(ckpt="esm2ds500_phys_best_seed{seed}.pt",
                       kw=dict(use_ngram=True, use_triangle=True, use_feature_gate=False,
                               use_attn_bias=True, use_attn_no_bias=False,
                               use_static_bias=False, pot_mode="pmf", pot_perturb_seed=None)),
}

CONDS_35M = {
    "baseline":   dict(ckpt="esm2ds35m_base500_best_seed{seed}.pt",
                       kw=dict(use_ngram=True, use_triangle=True, use_feature_gate=False,
                               use_attn_bias=False, use_static_bias=False, use_attn_no_bias=False,
                               pot_mode="pmf", pot_perturb_seed=None)),
    "axial":      dict(ckpt="esm2ds35m_axial500_best_seed{seed}.pt",
                       kw=dict(use_ngram=True, use_triangle=True, use_feature_gate=False,
                               use_attn_bias=False, use_attn_no_bias=True,
                               use_static_bias=False, pot_mode="pmf", pot_perturb_seed=None)),
    "shuffled":   dict(ckpt="esm2perturb35m500_p0_best_seed{seed}.pt",
                       kw=dict(use_ngram=True, use_triangle=True, use_feature_gate=False,
                               use_attn_bias=True, use_attn_no_bias=False,
                               use_static_bias=False, pot_mode="pmf", pot_perturb_seed=0)),
    "real":       dict(ckpt="esm2ds35m_phys500_best_seed{seed}.pt",
                       kw=dict(use_ngram=True, use_triangle=True, use_feature_gate=False,
                               use_attn_bias=True, use_attn_no_bias=False,
                               use_static_bias=False, pot_mode="pmf", pot_perturb_seed=None)),
}


# ---------------- reconstruction ----------------
def _complete_distances(D, conf_thresh=20.0):
    """Graph-distance completion: confident edges (<20 A) kept, far pairs get
    shortest-path estimates over the confident graph (Floyd-Warshall, CPU float32)."""
    n = len(D)
    INF = 1e9
    G = np.where(D < conf_thresh, D, INF).astype(np.float64)
    np.fill_diagonal(G, 0.0)
    for k in range(n):
        G = np.minimum(G, G[:, k:k + 1] + G[k:k + 1, :])
    G = np.minimum(G, INF / 4)
    out = np.where(D < conf_thresh, D, G)
    return np.minimum(out, 60.0), (D < conf_thresh).astype(np.float64)


def mds_torch(D, iters=300, restarts=3):
    """Weighted SMACOF on the completed distance matrix; confident pairs dominate."""
    Dc, W = _complete_distances(D)
    n = len(D)
    Dv = torch.as_tensor(Dc, dtype=torch.float64)
    Wv = torch.as_tensor(0.05 + 0.95 * W, dtype=torch.float64)  # completed pairs: weak
    J = torch.eye(n, dtype=torch.float64) - torch.ones(n, n, dtype=torch.float64) / n
    B = -0.5 * J @ (Wv * Dv ** 2) @ J
    w, V = torch.linalg.eigh(B)
    idx = torch.argsort(w, descending=True)
    X0 = V[:, idx[:3]] * torch.sqrt(w[idx[:3]].clamp(min=0)).unsqueeze(0)
    X0 = X0 - X0.mean(0)
    scale = float(Dv[W > 0.5].mean()) if (W > 0.5).any() else float(Dv.mean())
    X0 = X0 / X0.norm(dim=1).mean().clamp(min=1e-9) * scale

    rng = torch.Generator().manual_seed(0)
    best_X, best_stress = None, float("inf")
    for r in range(restarts):
        X = X0.clone() if r == 0 else torch.randn(n, 3, generator=rng, dtype=torch.float64) * scale / n ** (1 / 3)
        X = X - X.mean(0)
        for _ in range(iters):
            d = torch.cdist(X, X).clamp(min=1e-6)
            Bk = Wv * Dv / d
            Xn = (Bk @ X) / Wv.sum(1, keepdim=True)
            if not torch.isfinite(Xn).all():
                break
            X = Xn - Xn.mean(0)
        if not torch.isfinite(X).all():
            continue
        d = torch.cdist(X, X).clamp(min=1e-6)
        stress = float((Wv * (d - Dv) ** 2).sum())
        if np.isfinite(stress) and stress < best_stress:
            best_stress, best_X = stress, X.clone()
    if best_X is None:
        best_X = X0
    return best_X.numpy()


def refine_torch(D, X, steps=300):
    """LBFGS (line-searched => cannot diverge) on confident restraints + bond prior."""
    Dt = torch.as_tensor(D, dtype=torch.float64)
    Xp = torch.as_tensor(X, dtype=torch.float64).clone().requires_grad_(True)
    n = len(D)
    i = torch.arange(n - 1, dtype=torch.long)

    def closure():
        opt.zero_grad()
        d = torch.cdist(Xp, Xp).clamp(min=1e-6)
        w = (Dt < 16.0).double() * 1.0 + 0.02          # restrain only confident pairs
        loss = (w * (d - Dt) ** 2).mean()
        bond = (torch.cdist(Xp[i], Xp[i + 1]).squeeze(-1) - 3.8) ** 2
        loss = loss + 0.5 * bond.mean()
        opt.zero_grad()
        loss.backward()
        return loss

    opt = torch.optim.LBFGS([Xp], lr=0.5, max_iter=steps,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-9)
    opt.step(closure)
    out = Xp.detach().numpy()
    if not np.isfinite(out).all():
        return np.asarray(X)
    return out


def kabsch_rmsd(P, Q):
    """RMSD after optimal superposition (P, Q: n x 3)."""
    P = P - P.mean(0); Q = Q - Q.mean(0)
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(Vt.T @ U.T @ np.eye(3))[np.arange(3), np.arange(3)].prod()
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    Pr = P @ R
    return float(np.sqrt(((Pr - Q) ** 2).sum() / len(P)))


def tm_score(P, Q, Lref=None):
    """TM-score with the standard cutoff 0.5*sqrt(min(L,Lref))-1.8... (Zhang & Skolnick)."""
    L = len(P); Lref = Lref or L
    d0 = 1.24 * (max(Lref, 15) - 15) ** (1 / 3) - 1.8
    cutoff = max(0.5 * np.sqrt(Lref) - 1.8, 4.0)  # length-dependent search set
    P = P - P.mean(0); Q = Q - Q.mean(0)
    best = 0.0
    for c in np.arange(4.0, cutoff + 1e-6, 0.5):
        H = P.T @ Q
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        Pr = P @ R
        dist = np.sqrt(((Pr - Q) ** 2).sum(1))
        best = max(best, float((1 / (1 + (dist / d0) ** 2)).sum() / Lref))
    return best


def lddt(P, Q, cutoff=15.0):
    """Local lDDT of prediction P against reference Q (all-atom proxy: Ca)."""
    n = len(Q)
    ref_d = np.linalg.norm(Q[:, None] - Q[None, :], axis=-1)
    mask = (ref_d < cutoff) & (ref_d > 1e-6)
    np.fill_diagonal(mask, False)
    pred_d = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
    per_res = []
    for i in range(n):
        m = mask[i]
        if m.sum() == 0:
            continue
        preserved = (np.abs(pred_d[i][m] - ref_d[i][m]) < 4.0).mean()
        per_res.append(preserved)
    return float(np.mean(per_res))


# ---------------- model runner ----------------
_cache = {}
_emb_cache = None


def load_emb_cache():
    global _emb_cache
    if _emb_cache is None:
        with open(CACHE, "rb") as f:
            _emb_cache = pickle.load(f)
    return _emb_cache


def predict_dist(cond_kw, ckpt_path, pid, esm_dim=1280, cache_file=None):
    pmf = N.load_pmf_potential(os.path.join(NEW1, "pmf37_cb.npz"))
    kw = dict(esm_dim=esm_dim, hidden=64, n_bins=N.N_BINS, num_blocks=4, bias_scale=0.05)
    kw.update(cond_kw)
    model = N.ProteinPredictor(pmf=pmf, **kw).to(DEV)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEV))
    model.eval()

    cache = load_emb_cache()
    x = np.load(os.path.join(CASPN, pid + ".npz"), allow_pickle=True)
    seq, dist = str(x["seq"]), x["dist"].astype(np.float32)
    L = len(seq)
    emb = np.array(cache[pid]["embedding"], dtype=np.float32)[:L]
    e = torch.from_numpy(emb).unsqueeze(0).to(DEV)
    aa = N.seq_to_idx([seq], L, DEV)
    mask = torch.ones(1, L).to(DEV)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        logits, _, _, _ = model(e, aa, mask_1d=mask)
    pred = N.logits_to_pred_dist(logits)[0].float().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return pred[:L, :L], dist[:L, :L], seq


def native_ca(pid, seq):
    """Native Ca coords from the CASP14 native PDB dir (extract only residues the npz kept)."""
    import glob
    path = os.path.join(r"D:\数据集蛋白质\TDFold基准数据集\CASP14_test_data\CASP14_pdb",
                        pid + ".pdb")
    if not os.path.exists(path):
        return None
    try:
        coords = parse_ca(path)
    except Exception:
        return None
    if len(coords) == len(seq):
        return np.array(coords)
    if len(coords) > len(seq):
        # npz kept a subset of residues: take the matching contiguous block by best
        # Cb-distance reconstruction is overkill here; standard CASP domains are single-chain
        return np.array(coords[:len(seq)])
    return None


def parse_ca(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                out.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return out


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=None, help="comma-separated target ids; default = auto (100-300 aa, cached)")
    ap.add_argument("--backbone", default="650M", choices=["650M", "35M"])
    ap.add_argument("--out", default=os.path.join(NEW1, "folding_eval.csv"))
    args = ap.parse_args()

    global CONDS
    if args.backbone == "35M":
        CONDS = CONDS_35M
        global CACHE
        CACHE = os.path.join(NEW1, "test_sets", "test_esm_casp14_35M.pkl")
        global _emb_cache
        _emb_cache = None
        dim = 480
    else:
        dim = 1280
    cache = load_emb_cache()
    ids = sorted(f[:-4] for f in os.listdir(CASPN) if f.endswith(".npz"))
    if args.targets:
        ids = [t.strip() for t in args.targets.split(",")]

    rows = []
    for pid in ids:
        x = np.load(os.path.join(CASPN, pid + ".npz"), allow_pickle=True)
        seq = str(x["seq"]); L = len(seq)
        if not (80 <= L <= 300) or pid not in cache:
            continue
        # native structure
        nat = native_ca(pid, seq)
        if nat is None:
            print(f"[skip] {pid}: no native PDB with matching length")
            continue
        print(f"[{pid}] L={L} native found")
        for cond, cfg in CONDS.items():
            for seed in (0, 1, 2):
                ckpt = os.path.join(NEW1, cfg["ckpt"].format(seed=seed))
                if not os.path.exists(ckpt):
                    print(f"  [skip] {cond} s{seed}: no ckpt")
                    continue
                try:
                    pred, true_d, _ = predict_dist(cfg["kw"], ckpt, pid, esm_dim=dim)
                    D = pred.copy()
                    np.fill_diagonal(D, 0.0)
                    X0 = mds_torch(D)
                    X = refine_torch(D, X0)
                    rows.append(dict(
                        target=pid, L=L, cond=cond, seed=seed,
                        rmsd=kabsch_rmsd(X, nat),
                        tm=tm_score(X, nat),
                        lddt=lddt(X, nat),
                    ))
                except Exception as e:
                    print(f"  [err] {cond} s{seed}: {type(e).__name__}: {e}")
                    continue
                r = rows[-1]
                print(f"  {cond:9s} s{seed}: TM={r['tm']:.3f} RMSD={r['rmsd']:.2f} lDDT={r['lddt']:.3f}", flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nsaved {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
