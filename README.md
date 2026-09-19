# Statistical Potentials for Protein Distance Prediction

Injecting PDB-derived statistical potentials (PMF, inverse Boltzmann) into an ESM2-based
distance prediction model via attention-bias / feature-gating mechanisms, with controlled
shuffled-potential baselines to separate the *mechanism* of injection from the *content* of
the potential.

This repository accompanies the paper

> *"Disentangling Architectural and Content Effects of Statistical Potentials in Protein
> Distance Prediction"* (under review).

Everything needed to reproduce the reported experiments, analyses, and figures is shipped here:
the training code, the PMF build pipeline, the exact data splits, and the per-run / per-target
ledger files that back every number in the paper.

---

## 1. Repository contents

| File | Description |
|:--|:--|
| `new.py` | Model + training + evaluation. ESM2 backbone → n-gram + Evoformer-style triangle blocks + axial attention → 37-bin Cβ distance head. Single self-contained CLI. |
| `compute_pmf.py` | Builds the inverse-Boltzmann statistical potential from PDB structures (`--atom cb`, Cβ with Gly→Cα fallback, 37 bins `[2,20] Å @ 0.5 Å` + far bin). |
| `pmf_bins.py` | Shared 37-bin / amino-acid ordering definitions. Hard dependency of `compute_pmf.py`; also used to assert bin alignment between PMF and the model. |
| `pmf37_cb.npz` | **The statistical potential used in all reported experiments.** 37-bin Cβ PMF, built from 4,948 PDB structures / 5,119 chains / 167,127,835 residue pairs. Contains `pmf`, raw `counts`, `bin_edges/centers/widths`, and provenance metadata. |
| `cullpdb_pc15.0_res0.0-2.5_len40-10000_R0.3_Xray_d2026_07_13_chains5241` | PISCES cull list used to select the PMF source structures (15% sequence identity, resolution ≤ 2.5 Å, R-factor ≤ 0.3, X-ray, min length 40). |
| `LICENSE` | MIT license. |

All per-run / per-target result ledgers that back the paper's numbers live in `results/`:

| File | Description |
|:--|:--|
| `results/splits.csv` | Exact train/valid/test assignment for every seed (0/1/2) at every training size (500/1500/3000). Valid = 100, test = 100, train is a nested prefix (500 ⊂ 1500 ⊂ 3000). Plus a per-protein `usable` flag. |
| `results/training_curves.csv` | Per-run × per-epoch valid/test P@L/5 (source of convergence figures and epoch-wise content decomposition). |
| `results/paired_seed_scores.csv` | Per-condition × seed held-out P@L/5 (source of all main-effect tables and per-seed paired differences). |
| `results/hard_target_scores.csv` | ESM2-35M per-target P@L/5 for the real potential vs. 15 shuffled-potential zero-distribution runs (source of target-level statistics). |
| `results/cka_per_protein.csv` | Per-protein linear CKA values (n=32 CASP14 proteins) — raw data behind the representational analysis figure. |
| `results/cka_results.csv` | Aggregated CKA statistics (pairwise + PMF-alignment + paired t-tests). |
| `results/master_table.csv` | Consolidated mean±std P@L/5 across all conditions / test sets. |
| `results/pmf_source_chains.csv` | Chain-level list of the 5,119 PMF source structures (PDB id, chain, length, residue-pair count). |
| `results/homolog_hits.csv` | Per-benchmark homolog hits against the PMF source corpus: best-hit identity/coverage and hit counts at 25/30/50/90/100% identity thresholds. |
| `results/sens_overlap_35m.txt` | 35M target-level content contrasts after excluding targets with high-identity homologs in the PMF corpus (source of the homolog-overlap sensitivity table). |
| `results/decomp_paired_stats.txt` | Paired seed-level statistics for the architecture / mechanism / content decomposition steps (500 proteins, 3 seeds). |

### Scripts and tests

| Path | Description |
|:--|:--|
| `scripts/dump_splits.py` | Audit/re-generation script that reconstructs `splits.csv` from a local copy of the trRosetta npz + fasta, replaying `new.py`'s exact split logic (`--npz-dir` / `--fasta`). |
| `scripts/sens_overlap_35m.py` | Homolog-overlap sensitivity analysis: recomputes the 35M target-level contrasts after excluding targets with high-identity homologs in the PMF corpus. |
| `tests/test_bin_alignment.py` | Asserts the model bins (`new.py`) and PMF bins (`pmf_bins.py`) are exactly aligned: `BIN_CONFIG`, amino-acid order, 37-bin edges, and `torch.bucketize(..., right=True)` semantics. |

Additional analysis scripts that regenerate the CSV ledgers from raw run logs
(`build_ledgers.py`, `aggregate_all.py`, `analyze_35m.py`) and the test-set preparation
script (`prepare_test_sets.py`) are available from the authors on request; the CSV ledgers
in `results/` are the published record.

---

## 2. Environment

```bash
conda create -n distpred python=3.10
conda activate distpred
pip install -r requirements.txt
```

Core dependencies: `torch`, `numpy`, `transformers` (EsmModel/AutoTokenizer), `modelscope`
(ESM2 weights download), `tqdm`, `biopython` (PMF build only). CUDA GPU recommended
(650M runs fit comfortably on a single 24 GB GPU; crop 256).

ESM2 weights are fetched from ModelScope automatically on first run. To use a local weights
directory instead, pass `--esm-weights <dir>`.

> Windows note: the scripts force UTF-8 console output; set `PYTHONIOENCODING=utf-8` if your
> terminal still shows encoding errors.

---

## 3. Data availability

**Training data (trRosetta format).** The model is trained on the trRosetta training set
(Yang et al., *PNAS* 2020), as 15,049 per-protein `.npz` files (`seq`, `dist`, `mask1d`),
with the 20-Å far-sentinel convention (`dist == 0` encodes `≥ 20 Å`). The exact files used,
their split into train/valid/test, and the `usable` filter are fully specified by
`results/splits.csv` (see §4). The npz directory itself is the standard trRosetta-processed dataset;
point `--trr-npz-dir` / `--trr-fasta` at your copy.

**PMF source structures (PDB).** The statistical potential is derived from the PISCES
cull list shipped as `cullpdb_pc15.0_..._chains5241`:
*15% sequence identity, resolution ≤ 2.5 Å, R-factor ≤ 0.3, X-ray only, min length 40*.
The 5,059 listed PDB entries are downloaded from RCSB PDB; 4,948 downloaded successfully
and contributed the 5,119 chains used for the PMF. Chain selection, residue filtering, and
distance binning are deterministic and fully described in §7.

**ESM2.** Pre-trained weights come from the public ESM2 release (facebook/esm2_t33_650M_UR50D
and esm2_t12_35M_UR50D), fetched via ModelScope.

**External test sets.** CASP14 / CASP15 / CAMEO targets are standardized to the same npz
format (from their public releases) by the test-set preparation script — available from the
authors on request (see §6).

---

## 4. Reproduce the reported analyses

The CSV files in this repository **are** the exact numbers reported in the paper. Each ledger
records its provenance and is regenerable:

| Ledger | Regenerated by | Input data |
|:--|:--|:--|
| `results/splits.csv` | `scripts/dump_splits.py` | local trRosetta npz directory + `15051.fasta` + seeds 0/1/2 (pass via `--npz-dir` / `--fasta`). Replays `new.py`'s split logic (sorted glob → `default_rng(seed)` shuffle → valid 100 / test 100 / train prefix) and the dataset usability filter. |
| `results/training_curves.csv` | *(available from the authors on request)* | per-run `*_history_seed*.csv` logs |
| `results/paired_seed_scores.csv` | *(available from the authors on request)* | per-run history logs, `test_cameo` row |
| `results/hard_target_scores.csv` | *(available from the authors on request)* | per-target pickle files from `--eval-per-protein` |
| `results/cka_per_protein.csv` / `cka_results.csv` | *(published record; CKA computation script available from the authors on request)* | trained checkpoints (final-head pair representations) |
| `results/master_table.csv` | *(available from the authors on request)* | the ledgers above |

Key reproducibility facts, all enforced by `results/splits.csv` and cross-checked against run logs:

- **Split.** `sorted(glob("*.npz"))` → `np.random.default_rng(seed).shuffle` →
  `valid = ids[:100]`, `test_cameo = ids[100:200]`, `pool = ids[200:]`,
  `train_{s} = pool[:s]`. Seeds **0/1/2**. Training sets are **nested**: `500 ⊂ 1500 ⊂ 3000`,
  with valid/test identical across sizes.
- **`test_cameo` naming.** This is the model's internal held-out split of **100 trRosetta
  training-set proteins**, kept aside before the nested train prefixes. It is *not* the
  PDNET CAMEO benchmark (that appears separately under `cameo` in the external-test tables).
- **Usability filter.** A training protein counts only if its masked length is ≥ 20,
  equals the filtered FASTA length, and satisfies `20 ≤ L ≤ 1024` (masked). For
  train-500 this yields 496/499/499 usable proteins at seeds 0/1/2.

---

## 5. Train a model

All experiments use **40 epochs**, crop 256, batch 4, lr 5e-4, and **seeds 0/1/2**.
Valid/test sizes are fixed at 100/100. The `--pmf` flag points at `pmf37_cb.npz`.

Baseline (no physics):

```bash
python new.py \
  --data-source trrosetta \
  --trr-npz-dir /path/to/trrosetta/npz \
  --trr-fasta  /path/to/trrosetta/15051.fasta \
  --pmf pmf37_cb.npz \
  --epochs 40 --train-size 3000 --trr-valid-size 100 --trr-test-size 100 \
  --exp-name ds3000_base --seed 0
```

Soft lookup injection (attention bias) — the main "physical prior" condition:

```bash
python new.py --data-source trrosetta \
  --trr-npz-dir /path/to/trrosetta/npz --trr-fasta /path/to/trrosetta/15051.fasta \
  --pmf pmf37_cb.npz \
  --epochs 40 --train-size 3000 --trr-valid-size 100 --trr-test-size 100 \
  --attn-bias --exp-name ds3000_phys --seed 0
```

Other injection mechanisms and controls (used for the mechanism/content decomposition):

| Condition | Flags |
|:--|:--|
| Shuffled potential (content control) | `--attn-bias --pot-perturb-seed 0` |
| Feature gating | `--feature-gate` |
| Static aa-pair bias | `--attn-bias-static` |
| Static bias, shuffled | `--attn-bias-static --pot-perturb-seed 0` |
| Physical energy loss L_phys | `--phys-loss` |
| Architecture-only control (no bias) | `--attn-no-bias` |
| 35M backbone variant | `--esm-model esm2_t12_35M_UR50D` |

Smaller training sizes: repeat any of the above with `--train-size 1500` or `--train-size 500`.
Outputs: `esm2{exp_name}_history_seed{seed}.csv` (per-epoch logs) and
`esm2{exp_name}_best_seed{seed}.pt` (best checkpoint by valid P@L/5).

---

## 6. Evaluate external targets

First, standardize the public test targets and pre-compute their ESM2 embeddings. The
test-set preparation script (available from the authors on request) builds
`test_sets/{casp14,casp15,cameo}` npz files and their ESM caches from the public CASP14/15
and CAMEO releases.

Then evaluate a trained checkpoint:

```bash
python new.py \
  --eval-external-test test_sets/casp14 \
  --weights esm2ds3000_base_best_seed0.pt \
  --eval-test-cache test_sets/test_esm_casp14.pkl \
  --eval-per-protein pprot_casp14.pkl \
  --pmf pmf37_cb.npz --seed 0
```

- `--eval-per-protein` additionally writes a per-target P@L/5 pickle (needed for
  target-level statistics such as `results/hard_target_scores.csv`).
- Reconstruct the model with the **same injection flags** used at training (e.g. add
  `--attn-bias` for a soft-lookup checkpoint), since the checkpoint state must match the
  architecture.
- Use `--esm-model esm2_t12_35M_UR50D` when evaluating a 35M checkpoint.

---

## 7. Build the PMF

`pmf37_cb.npz` is regenerable end-to-end:

```bash
# 1) Download the PDB entries listed in the PISCES cull list
#    (cullpdb_pc15.0_..._chains5241) from RCSB PDB into pdbs/

# 2) Build the potential
python compute_pmf.py \
  --pdb-dir pdbs \
  --chain-file pdbs/cull_chains.txt \
  --min-seq-sep 2 --pseudo-count 1.0 --smooth 0 \
  --atom cb --out pmf37_cb.npz
```

Conventions (matching the shipped `pmf37_cb.npz`):

- **Distance atom.** Cβ–Cβ; Gly (no Cβ) falls back to Cα (trRosetta convention).
- **Bins.** 37 bins over `[2, 20] Å` at width 0.5 Å, plus a far bin for `> 20 Å`
  (`FAR_BIN = 36`, center 25 Å). Binning is identical to the model's
  `true_dist_to_bin_idx`; `pmf_bins.py` + `tests/test_bin_alignment.py` assert this alignment.
- **Pair filtering.** Only residue pairs with sequence separation `≥ 2` are counted
  (excludes covalent i,i+1 neighbors at ~3.8 Å). Non-standard residues and residues
  missing the target atom are skipped.
- **Smoothing.** Off (`--smooth 0`); pseudo-count λ = 1.0 to avoid zero probabilities.
- **Inverse Boltzmann.** `E = −log[ P(r|aa,aa) / P(r) ]`, symmetrized, clipped to ±4.
  Lower `E` = more favorable. The model uses `−E` as an attention bias (see `new.py`).

> **Transparency note.** The shipped `pmf37_cb.npz` was built **without an exclude file**:
> no CASP14/15 or CAMEO targets were removed from the PMF source set. `compute_pmf.py`
> supports `--exclude-file` for leakage control; we deliberately report the potential as
> actually used, and the target-level statistics in the paper are robust to this choice.

---

## 8. Citation

If you use this code or the PMF in your work, please cite the paper (and this repository):

```bibtex
@article{<your-paper-key>,
  title   = {Injecting Physical Priors into Protein Distance Prediction:
             Controlled Evidence from PDB-Derived Statistical Potentials},
  author  = {<authors>},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2026},
  doi     = {<doi>}
}
```

---

## License

MIT — see [LICENSE](LICENSE). This applies to the code and analysis files in this
repository. The PDB structures, PISCES cull lists, ESM2 weights, and benchmark data remain
subject to their respective source licenses.
