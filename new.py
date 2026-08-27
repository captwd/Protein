"""
ESM2 + 物理约束蛋白质距离图预测 — 完整框架
=============================================
架构:
  序列 → ESM2 → n-gram → outer sum → 三角乘法 → 距离分桶 → 输出
                                                    ↓
                  可学习势（特征门控/注意力bias/L_stat）
                                                    ↓
                  辅助 head + distogram 局部约束（bond/angle）

物理注入方式（可消融，基座为 PDB 统计势 PMF）:
  - 先验基座:  PairPotential 冻结 pmf.npz（E_pmf）+ 可学习残差 ΔE，E = scale·E_pmf + ΔE
  - Loss 级:    L_phys（冻结 PMF 能量约束）+ L_stat（可学习）+ L_clash + 几何约束
  - 特征级:     feature-gate 门控融合 ESM2 ∥ 物理势能
  - 注意力级:   axial attention + (-E) bias（低能有利 → 高注意力）

Usage:
  # 基线（4层三角块，无物理注入）
  python new.py --epochs 30

  # 默认: trRosetta 训练集（Cβ 37-bin，dist 字段 far 哨兵=0）+ 37-bin Cβ PMF（--pmf 自动指向 ../new1/pmf37_cb.npz）

  # 用 PDB-PMF 作注意力 bias（残差微调）
  python new.py --epochs 30 --attn-bias

  # 用 PDB-PMF 作特征门控
  python new.py --epochs 30 --feature-gate

  # 冻结 PMF 能量约束损失 L_phys
  python new.py --epochs 30 --phys-loss

  # 残差全冻结（纯先验消融）
  python new.py --epochs 30 --attn-bias --pot-mode pmf_only

  # 打乱 PMF（随机先验扰动实验，验冗余性）
  python new.py --epochs 30 --attn-bias --pot-perturb-seed 0

  # 旧的可学习势（作对比）
  python new.py --epochs 30 --attn-bias --pot-mode learned

  # PDNet 数据源（Cβ 全距离，37-bin 模型自动把 >20Å 归入 far bin）
  python new.py --data-source pdnet --epochs 30

  # 小样本冒烟（trRosetta）
  python new.py --epochs 3 --train-size 300 --trr-valid-size 100 --trr-test-size 100

  # 全开
  python new.py --epochs 30 --num-blocks 8 --attn-bias --feature-gate --phys-loss --clash --tri
"""

import os, sys, pickle, argparse, warnings, time, glob
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmModel
from modelscope import snapshot_download
from tqdm import tqdm
os.environ["TQDM_ASCII"] = "1"

# ==========================================
# 0. Config
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

DATA_DIR = r"D:\数据集蛋白质\PDNET\data"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "esm2_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 训练历史 / checkpoint 输出目录（new1）
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "new1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 注意：现有缓存 key = pdb_id（如 "1qzmA"），value = {'embedding': (L,1280) fp16}。
# 老代码用 pdb_id:sha1(seq) 作 key 且文件名不同，这里直接对齐现有缓存。
CACHE_FILE = os.path.join(CACHE_DIR, "esm2_all_proteins.pkl")

# ==========================================
# AA constants
# ==========================================
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

# ==========================================
# trRosetta 37-bin distance binning: [2,20]Å @0.5Å (36 bin) + far bin (>20Å)
# ==========================================
BIN_CONFIG = [
    (2.0, 20.0, 0.5),
]
FAR_BIN = 36            # 第 37 个 bin（0 起编号）: >20Å
FAR_BIN_CENTER = 25.0   # far bin 的期望距离中心（论文需写明此约定）

def make_bin_edges(config):
    edges = []
    for start, end, width in config:
        edges.extend([start + i * width for i in range(int((end - start) / width))])
    edges.append(end)
    edges.append(float("inf"))     # far bin 上界
    return torch.tensor(edges)

BIN_EDGES = make_bin_edges(BIN_CONFIG)          # (38,) 最后是 inf
N_BINS = len(BIN_EDGES) - 1                      # 37
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0   # (37,)
BIN_CENTERS[-1] = FAR_BIN_CENTER
BIN_WIDTHS = BIN_EDGES[1:] - BIN_EDGES[:-1]             # (37,)
BIN_WIDTHS[-1] = torch.tensor(float("inf"))

def dist_to_bin_idx(dist, bin_edges):
    idx = torch.bucketize(dist, bin_edges, right=True) - 1
    idx = idx.clamp(0, len(bin_edges) - 2)
    return idx

def true_dist_to_bin_idx(dist, bin_edges):
    """目标距离 -> bin。Rosettosa `dist` 字段的 0.0 是 far 哨兵（≥20Å 全编码成 0）→ FAR_BIN。
    真实 0 距离（对角线）在损失里已被 sep>=1 的 mask 排除，映射到 far bin 无害。"""
    idx = dist_to_bin_idx(dist, bin_edges)
    return torch.where(dist == 0, torch.full_like(idx, FAR_BIN), idx)


def logits_to_pred_dist(logits, bin_centers=None):
    """从分桶 logits 计算期望距离（无数据泄露）。各 loss/eval 共用。"""
    centers = BIN_CENTERS if bin_centers is None else bin_centers
    centers = centers.to(logits.device)
    pred_prob = F.softmax(logits.float(), dim=1)
    return (pred_prob * centers.view(1, -1, 1, 1)).sum(dim=1)


def seq_to_idx(seq_str_list, max_L, device):
    B = len(seq_str_list)
    idx = torch.zeros(B, max_L, dtype=torch.long, device=device)
    for b, s in enumerate(seq_str_list):
        L = min(len(s), max_L)
        idx[b, :L] = torch.tensor([AA_TO_IDX.get(aa, 0) for aa in s[:L]],
                                   dtype=torch.long, device=device)
    return idx


def build_sep_map(L, device):
    idx = torch.arange(L, device=device)
    return (idx.unsqueeze(1) - idx.unsqueeze(0)).abs().float()


# ==========================================
# 1. ESM2 Backbone (Frozen) - Lazy Load
# ==========================================
_ESM_MODEL = None
_ESM_TOKENIZER = None

def _get_esm_model():
    global _ESM_MODEL, _ESM_TOKENIZER
    if _ESM_MODEL is None:
        model_id = _ESM_MODEL_ID
        print("\n" + "="*60)
        print(f"[1/6] Loading {model_id} ...")
        print("="*60)
        if _ESM_WEIGHTS_DIR and os.path.isdir(_ESM_WEIGHTS_DIR):
            MODEL_PATH = _ESM_WEIGHTS_DIR
            print(f"   Loading local weights from: {MODEL_PATH}")
        else:
            print("   Downloading/checking model via ModelScope ...")
            MODEL_PATH = snapshot_download(f"facebook/{model_id}",
                                           ignore_file_pattern=['*.bin', '*.h5'])
            print(f"   Model cached at: {MODEL_PATH}")
        _ESM_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
        torch_dtype = torch.float32 if DEVICE == "cpu" else torch.float16
        _ESM_MODEL = EsmModel.from_pretrained(
            MODEL_PATH, attn_implementation="eager", torch_dtype=torch_dtype)
        _ESM_MODEL = _ESM_MODEL.to(DEVICE)
        _ESM_MODEL.eval()
        for p in _ESM_MODEL.parameters():
            p.requires_grad = False
        print(f"   {model_id} loaded, frozen, emb_dim={_ESM_MODEL.config.hidden_size}")
    return _ESM_MODEL, _ESM_TOKENIZER

# ---- ESM backbone registry（CLI 可切换；默认 650M，保持现有全部结果向后兼容） ----
ESM_MODELS = {
    "esm2_t33_650M_UR50D": {"dim": 1280, "tag": "650M"},
    "esm2_t12_35M_UR50D":  {"dim": 480,  "tag": "35M"},
}
_ESM_MODEL_ID = "esm2_t33_650M_UR50D"    # 由 main()/外部脚本在加载前设置
_ESM_WEIGHTS_DIR = None                   # 本地权重目录（--esm-weights）；None 则 ModelScope 下载
ESM_DIM = ESM_MODELS[_ESM_MODEL_ID]["dim"]

# ---- ESM2 Cache ----
_EMBEDDING_CACHE = {}

def extract_esm2_emb(seq):
    model, tokenizer = _get_esm_model()
    inp = tokenizer(seq, return_tensors="pt", add_special_tokens=True).to(DEVICE)
    with torch.no_grad():
        emb = model(**inp, output_hidden_states=False).last_hidden_state[0, 1:-1]
        if DEVICE == "cpu":
            emb = emb.float()
        return emb.cpu().numpy()

def _seq_key(pdb_id, seq):
    # PDNET 中序列按 pdb_id 唯一，直接以 pdb_id 为 key（对齐现有缓存）
    return pdb_id

def cache_embeddings(datasets):
    global _EMBEDDING_CACHE
    if os.path.exists(CACHE_FILE):
        print(f"   Loading cached embeddings ...")
        with open(CACHE_FILE, "rb") as f:
            _EMBEDDING_CACHE = pickle.load(f)
        print(f"   Loaded {len(_EMBEDDING_CACHE)} proteins")
    
    seen = set()
    items = []
    missing_items = []
    for ds in datasets:
        for s in ds.samples:
            key = _seq_key(s["pdb_id"], s["seq"])
            if key not in seen:
                seen.add(key)
                if key not in _EMBEDDING_CACHE:
                    missing_items.append(s)
                items.append(s)
    
    if missing_items:
        print(f"   Missing {len(missing_items)} embeddings, computing ...")
        model, _ = _get_esm_model()
        results = {}
        failed = []
        for item in tqdm(missing_items, desc="ESM2"):
            try:
                emb = extract_esm2_emb(item["seq"])
                if emb.shape[0] == item["L"]:
                    key = _seq_key(item["pdb_id"], item["seq"])
                    results[key] = {"embedding": emb.astype(np.float16), "seq": item["seq"]}
                else:
                    failed.append(item["pdb_id"])
            except Exception as e:
                failed.append(item["pdb_id"])
        _EMBEDDING_CACHE.update(results)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(_EMBEDDING_CACHE, f)
        print(f"   Saved {len(results)} new embeddings")
        if failed:
            print(f"   WARNING: {len(failed)} proteins failed to extract, will be skipped")
    
    def has_valid_cache(s):
        key = _seq_key(s["pdb_id"], s["seq"])
        item = _EMBEDDING_CACHE.get(key)
        return item is not None and item["embedding"].shape[0] == s["L"]

    for ds in datasets:
        ds.samples = [s for s in ds.samples if has_valid_cache(s)]
        print(f"   {type(ds).__name__}: {len(ds.samples)} remaining")
    
    for ds in datasets:
        if len(ds.samples) == 0:
            print(f"   [ERROR] {type(ds).__name__} has 0 samples after cache filtering!")
            raise ValueError(f"{type(ds).__name__} has 0 samples after cache filtering")

def get_emb(pdb_id, seq):
    key = _seq_key(pdb_id, seq)
    return _EMBEDDING_CACHE.get(key)


# ==========================================
# 2. n-gram 1D Encoder
# ==========================================
class NgramEncoder(nn.Module):
    def __init__(self, in_dim=1280, hidden_dim=32, kernels=(3, 5, 7), dropout=0.1):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(in_dim, hidden_dim, kernel_size=k, padding=k // 2)
            for k in kernels
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in kernels
        ])

        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * len(kernels)),
            nn.Linear(hidden_dim * len(kernels), hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x.transpose(1, 2)

        outs = []
        for conv, norm in zip(self.convs, self.norms):
            out = conv(x)
            out = out.transpose(1, 2)
            out = norm(out)
            out = F.gelu(out)
            outs.append(out)

        out = torch.cat(outs, dim=-1)
        out = self.fusion(out)
        return out


# ==========================================
# 3. Triangle Multiplication
# ==========================================
class TriangleMultiplication(nn.Module):
    """
    AlphaFold-style triangle multiplicative update with chunking.
    Σ_k a_ik ⊙ b_jk → update z_ij
    
    Note: This module returns only the update (not the residual connection).
    The residual is handled by the caller (TriangleBlock) for standard residual around update pattern.
    
    Uses chunking to reduce memory peak while maintaining O(L^3) complexity.
    """
    def __init__(self, c=64, chunk_size=64):
        super().__init__()
        self.c_hidden = c // 2
        self.chunk_size = chunk_size

        self.a = nn.Sequential(nn.Linear(c, c//2), nn.ReLU())
        self.b = nn.Sequential(nn.Linear(c, c//2), nn.ReLU())
        self.g = nn.Linear(c, c//2)
        self.out = nn.Linear(c//2, c)

    def forward(self, z, pair_mask=None):
        """z: (B, L, L, c) → (B, L, L, c)"""
        B, L, _, _ = z.shape

        a = self.a(z)
        b = self.b(z)
        g = torch.sigmoid(self.g(z))

        if pair_mask is not None:
            m = pair_mask.unsqueeze(-1).to(a.dtype)
            a = a * m
            b = b * m
            g = g * m

        outs = []
        scale = L ** -0.5

        for start in range(0, L, self.chunk_size):
            end = min(start + self.chunk_size, L)

            out_o = torch.einsum(
                "bikc,bjkc->bijc",
                a[:, start:end].float(),
                b.float(),
            )

            out_i = torch.einsum(
                "bkic,bkjc->bijc",
                a[:, :, start:end].float(),
                b.float(),
            )

            out = (out_o + out_i) * scale
            out = out.to(z.dtype) * g[:, start:end]
            outs.append(self.out(out))

        update = torch.cat(outs, dim=1)

        if pair_mask is not None:
            update = update * pair_mask.unsqueeze(-1).to(update.dtype)

        return update


# ==========================================
# 4. Pair Potential —— PDB 统计势 PMF 基座 + 可学习残差
# ==========================================
def load_pmf_potential(path):
    """加载 compute_pmf.py 输出的 pmf.npz，并断言 bin/AA 与 new.py 完全对齐。"""
    d = np.load(path)
    pmf = d["pmf"].astype(np.float32)                        # (N_BINS, 20, 20)
    assert pmf.shape == (N_BINS, 20, 20), \
        f"PMF shape 应为 ({N_BINS},20,20)，实为 {pmf.shape}"
    edges = d["bin_edges"].astype(np.float32)
    assert np.allclose(edges, BIN_EDGES.numpy().astype(np.float32), atol=1e-5), \
        "PMF 的 bin_edges 与 new.py 不一致！"
    aa = d["aa_order"]
    if isinstance(aa, np.ndarray):
        aa = aa.item() if aa.ndim == 0 else str(aa[0])
    assert str(aa).strip() == AA_ORDER, \
        f"PMF 的 AA_ORDER '{aa}' 与 new.py 不一致！"
    return torch.tensor(pmf)


class PairPotential(nn.Module):
    """
    成对统计势。核心公式（论文）:  E = scale·E_pmf + ΔE_learned
      mode='pmf':      E = exp(log_scale)·E_pmf(冻结) + delta(可学习)   ← 残差微调，默认
      mode='pmf_only': E = E_pmf（全冻结，纯先验消融）
      mode='learned':  E = softplus(log_scale)·delta（无可学习基座，旧方法，作对比）

    pmf 输入: (N_BINS, 20, 20) 张量（load_pmf_potential 输出，37-bin Cβ）。
    符号约定: E 越低越有利。attention bias 里用 -E（见 ProteinPredictor）。
    """
    def __init__(self, n_aa=20, n_bins=N_BINS, mode='pmf', pmf=None, shuffle_seed=None):
        super().__init__()
        self.mode = mode
        self.n_bins = n_bins
        self.register_buffer("bin_edges", BIN_EDGES.float())
        self.register_buffer("bin_centers", BIN_CENTERS.float())

        if mode in ('pmf', 'pmf_only'):
            assert pmf is not None, f"mode='{mode}' 需要 pmf 张量（--pmf 指向 pmf.npz）"
            pmf = pmf.float()
            if shuffle_seed is not None:
                # 扰动实验：随机置换氨基酸标签 → 破坏 aa 特异性但保留距离谱
                rng = np.random.default_rng(shuffle_seed)
                perm = torch.tensor(rng.permutation(n_aa), dtype=torch.long)
                pmf = pmf[:, perm][:, :, perm]
                print(f"[PairPotential] PMF 氨基酸标签已打乱 (seed={shuffle_seed})")
            pmf = 0.5 * (pmf + pmf.transpose(1, 2))           # 对称化
            self.register_buffer("e_pmf", pmf)                # 冻结基座
            if mode == 'pmf':
                self.delta = nn.Parameter(torch.zeros(n_bins, n_aa, n_aa))
                self.log_scale = nn.Parameter(torch.tensor(0.0))   # exp(0)=1 → 初始即 E_pmf
            else:
                self.register_buffer("delta", torch.zeros(n_bins, n_aa, n_aa))
                self.register_buffer("log_scale", torch.tensor(0.0))
        else:  # learned —— 旧的可学习势（随机 init，作对比）
            self.register_buffer("e_pmf", torch.zeros(n_bins, n_aa, n_aa))
            init_delta = torch.zeros(n_bins, n_aa, n_aa)
            for b in range(n_bins):
                init_delta[b] = -5.0 if self.bin_centers[b].item() < 8.0 else 5.0
            self.delta = nn.Parameter(init_delta)
            self.log_scale = nn.Parameter(torch.tensor(np.log(0.5)))

    def pot(self):
        """返回当前 (n_bins, 20, 20) 势能张量（含梯度）。"""
        if self.mode == 'pmf':
            return torch.exp(self.log_scale) * self.e_pmf + self.delta
        if self.mode == 'pmf_only':
            return self.e_pmf
        return F.softplus(self.log_scale) * self.delta

    def static_bias(self, aa_idx):
        """静态 aa 对 bias: -min_{bin<far} E[b, aa, aa]（最有利接触亲和力）。
        不依赖预测距离 → 无软查表稀释，信号最强。
        返回 (B, L, L)：亲和力强的 aa 对 → 大正 bias → 高注意力。"""
        pot = self.pot()
        min_e = pot[:-1].min(dim=0).values          # (20,20)，排除 far bin（≈0）
        bias = -min_e                               # 亲和力强 → 正
        L = aa_idx.shape[1]
        aa_i = aa_idx.unsqueeze(2).expand(-1, -1, L)
        aa_j = aa_idx.unsqueeze(1).expand(-1, L, -1)
        return bias[aa_i, aa_j]                     # (B, L, L)

    def forward(self, aa_idx, dist_or_logits, force_symmetric=True):
        """
        aa_idx: (B, L)
        dist_or_logits: 4D logits (B, n_bins, L, L) → 可微软查表（训练）
                        3D 距离 (B, L, L)            → 硬查表（eval/debug）
        返回: (B, L, L) 能量值
        """
        L = aa_idx.shape[1]
        pot = self.pot()
        pot_flat = pot.reshape(self.n_bins, -1)              # (N_BINS, 400)
        aa_i = aa_idx.unsqueeze(2).expand(-1, -1, L)
        aa_j = aa_idx.unsqueeze(1).expand(-1, L, -1)
        pair_idx = aa_i * 20 + aa_j                          # (B, L, L)

        if dist_or_logits.dim() == 4:
            pred_prob = F.softmax(dist_or_logits.float(), dim=1)     # (B, N_BINS, L, L)
            e = pot_flat[:, pair_idx]                                # (N_BINS, B, L, L)
            energy = (pred_prob.transpose(0, 1).float() * e).sum(dim=0)  # (B, L, L)
        else:
            bin_idx = true_dist_to_bin_idx(dist_or_logits, self.bin_edges.to(dist_or_logits.device))
            energy = pot_flat[bin_idx, pair_idx]                     # (B, L, L)

        if force_symmetric:
            energy = (energy + energy.transpose(1, 2)) / 2
        return energy


# ==========================================
# 4.5. Relative Position Encoding & Triangle Block
# ==========================================
class RelativePositionEncoding(nn.Module):
    """给成对表示加入可学习的序列距离偏置。"""
    def __init__(self, hidden=64, max_rel_dist=32):
        super().__init__()
        self.max_dist = max_rel_dist
        self.bias = nn.Parameter(torch.randn(2 * max_rel_dist + 1, hidden) * 0.1)

    def forward(self, z):
        """z: (B, L, L, hidden) → (B, L, L, hidden)"""
        L = z.shape[1]
        pos = torch.arange(L, device=z.device)
        rel_pos = pos[:, None] - pos[None, :]                     # (L, L)
        rel_pos = rel_pos.clamp(-self.max_dist, self.max_dist) + self.max_dist
        return z + self.bias[rel_pos]                             # broadcast B


class TriangleBlock(nn.Module):
    """
    单层 Evoformer-style 三角模块:
      LayerNorm → TriangleMultiplication → LayerNorm → Transition(FFN)
    """
    def __init__(self, c=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(c)
        self.tri_mul = TriangleMultiplication(c)
        self.norm2 = nn.LayerNorm(c)
        self.transition = nn.Sequential(
            nn.Linear(c, c * 2),
            nn.ReLU(),
            nn.Linear(c * 2, c),
        )

    def forward(self, z, pair_mask=None):
        z = z + self.tri_mul(self.norm1(z), pair_mask=pair_mask)
        z = z + self.transition(self.norm2(z))
        return z


# ==========================================
# 5. Main Prediction Head
# ==========================================
class AxialSelfAttention(nn.Module):
    """Axial attention with optional learned potential bias."""
    def __init__(self, dim=64, n_heads=4, dropout=0.1, bias_scale=0.05):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.bias_scale = bias_scale
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def _normalized_bias(self, bias):
        """返回归一化的 (B, L, L) 物理偏置；bias=None 返回 None。"""
        if bias is None:
            return None
        b = bias.float()
        b = b - b.mean(dim=(-1, -2), keepdim=True)
        b = b / (b.std(dim=(-1, -2), keepdim=True, unbiased=False) + 1e-6)
        return self.bias_scale * b

    def _pass_mask(self, bias_norm, key_mask, B, L):
        """构造 (B*L, 1, L, L) 加法注意力掩码（bias 全矩阵 + key padding）。
        bias 对同一 batch 的所有行/列序列共享同一 (L,L) 矩阵，故先按 (B,L,L) 组合再广播。"""
        if bias_norm is None and key_mask is None:
            return None
        dev = bias_norm.device if bias_norm is not None else key_mask.device
        base = bias_norm if bias_norm is not None else torch.zeros(B, L, L, device=dev)
        if key_mask is not None:
            base = torch.where(key_mask > 0, base, torch.full_like(base, -1e4))
        return base.unsqueeze(1).expand(B, L, L, L).reshape(B * L, 1, L, L)

    def forward(self, x, bias=None, mask_1d=None):
        """
        x: (B, L, L, dim)
        bias: (B, L, L) or None（物理势能，低能 → 高注意力）
        mask_1d: (B, L) or None
        使用 FlashAttention（F.scaled_dot_product_attention），避免物化 (B*L, H, L, L) 注意力矩阵。
        """
        B, H, W, D = x.shape
        assert H == W
        L = H
        bias_norm = self._normalized_bias(bias)

        key_mask = None
        if mask_1d is not None:
            key_mask = mask_1d[:, None, :].expand(-1, L, -1)   # [query, key]=mask[key]

        # ---- 行 pass: 每行一个序列 ----
        attn_mask = self._pass_mask(bias_norm, key_mask, B, L)
        x_r = x.reshape(B * L, L, D)
        q, k, v = self.qkv(x_r).chunk(3, dim=-1)
        q = q.view(B * L, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B * L, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B * L, L, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, L, L, D)

        # ---- 列 pass: 每列一个序列 ----
        attn_mask = self._pass_mask(bias_norm, key_mask, B, L)
        x_c = out.permute(0, 2, 1, 3).reshape(B * L, L, D)
        q, k, v = self.qkv(x_c).chunk(3, dim=-1)
        q = q.view(B * L, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B * L, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B * L, L, self.n_heads, self.head_dim).transpose(1, 2)
        out_c = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0)
        out_c = out_c.transpose(1, 2).reshape(B, L, L, D).permute(0, 2, 1, 3)

        return x + self.norm(self.out(out_c))


class ProteinPredictor(nn.Module):
    """
    完整模型。
    可消融: n-gram, 三角块堆叠, 特征门控, 注意力bias
    """
    def __init__(self, esm_dim=1280, hidden=64, n_bins=N_BINS,
                 use_ngram=True, use_triangle=True,
                 use_feature_gate=False, use_attn_bias=False,
                 use_static_bias=False, use_attn_no_bias=False,
                 pot_mode='pmf', pmf=None, pot_perturb_seed=None,
                 num_blocks=4, bias_scale=0.05):
        super().__init__()
        self.use_ngram = use_ngram
        self.use_triangle = use_triangle
        self.use_feature_gate = use_feature_gate
        self.use_attn_bias = use_attn_bias
        self.use_static_bias = use_static_bias
        self.use_attn_no_bias = use_attn_no_bias
        self.hidden = hidden
        self.n_bins = n_bins

        # n-gram
        if use_ngram:
            self.ngram = NgramEncoder(esm_dim, 32)
            self.fuse_1d = nn.Linear(esm_dim + 32, hidden)
        else:
            self.fuse_1d = nn.Linear(esm_dim, hidden)

        # 2D pair construction
        self.pair_i = nn.Linear(hidden, hidden)
        self.pair_j = nn.Linear(hidden, hidden)

        # Relative position encoding
        self.rel_pos = RelativePositionEncoding(hidden)

        # Feature-level gating
        if use_feature_gate:
            self.mj_proj = nn.Linear(1, hidden)
            self.gate = nn.Linear(hidden * 2, 1)

        # Stacked TriangleBlocks (每个: LN → TriMul → LN → FFN)
        if use_triangle:
            self.blocks = nn.ModuleList([
                TriangleBlock(hidden) for _ in range(num_blocks)
            ])
        else:
            self.blocks = nn.ModuleList()

        # Axial attention with bias
        if use_attn_bias or use_attn_no_bias:
            self.axial = AxialSelfAttention(hidden, bias_scale=bias_scale)

        # Pair potential: PDB-PMF 基座 + 可学习残差（mode='learned' 时为旧可学习势）
        self.pair_pot = PairPotential(n_aa=20, n_bins=n_bins, mode=pot_mode,
                                      pmf=pmf, shuffle_seed=pot_perturb_seed)
        self.pot_mode = pot_mode

        # Initial distance bin head (用于生成初始距离，计算物理约束)
        self.init_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, n_bins, 3, padding=1),
        )

        # Final distance bin head (用于最终预测，计算损失)
        self.final_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, n_bins, 3, padding=1),
        )

        # Auxiliary geometry heads (for direct bond/angle prediction)
        self.bond_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.bond_head[2].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.bond_head[2].bias, 3.8)

        self.angle_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.angle_head[2].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.angle_head[2].bias, 1.92)

        # Log lambda for decay (used in feature gating)
        self.log_lambda = nn.Parameter(torch.tensor(-4.6))

    def forward(self, esm_emb, aa_idx, mask_1d=None):
        """
        esm_emb:  (B, L, 1280)  来自 ESM2 cache
        aa_idx:   (B, L)        氨基酸索引
        mask_1d:  (B, L)        有效残基mask（用于attention padding）
        """
        B, L = esm_emb.shape[:2]

        # ---- 1D 特征 ----
        if self.use_ngram:
            ngram_feat = self.ngram(esm_emb)
            feat_1d = torch.cat([esm_emb, ngram_feat], dim=-1)
        else:
            feat_1d = esm_emb
        feat_1d = self.fuse_1d(feat_1d)                   # (B, L, hidden)

        # ---- 2D 成对 ----
        z = (
            self.pair_i(feat_1d).unsqueeze(2)
            + self.pair_j(feat_1d).unsqueeze(1)
        )  # (B, L, L, hidden)

        # 相对位置编码
        z = self.rel_pos(z)

        # ---- 第一次前向：获取初始距离预测 ----
        z_init = z.permute(0, 3, 1, 2).contiguous()
        init_logits = self.init_head(z_init)
        init_logits = 0.5 * (init_logits + init_logits.transpose(-1, -2))

        # 可学习势能（使用logits，走可微路径）——静态 bias 时不需要（避免软查表稀释）
        if self.use_feature_gate or (self.use_attn_bias and not self.use_static_bias):
            pair_energy = self.pair_pot(aa_idx, init_logits.detach())  # (B, L, L)

            if mask_1d is not None:
                pm = mask_1d.unsqueeze(-1) * mask_1d.unsqueeze(1)
                pair_energy = pair_energy * pm.to(pair_energy.dtype)

        # ---- Feature-level gating (物理注入特征) ----
        if self.use_feature_gate:
            sep = build_sep_map(L, esm_emb.device).unsqueeze(0)
            lam = torch.exp(self.log_lambda)
            decay = 1.0 / (1.0 + lam * sep.pow(2))
            energy_decayed = pair_energy * decay           # (B, L, L)

            z_mj = self.mj_proj(energy_decayed.unsqueeze(-1))  # (B,L,L,hidden)
            gate_in = torch.cat([z, z_mj], dim=-1)
            alpha = torch.sigmoid(self.gate(gate_in))
            z = alpha * z + (1 - alpha) * z_mj

        # ---- Build pair_mask from mask_1d ----
        pair_mask = None
        if mask_1d is not None:
            pair_mask = mask_1d.unsqueeze(-1) * mask_1d.unsqueeze(1)
            pair_mask = pair_mask.to(z.dtype)
            z = z * pair_mask.unsqueeze(-1)

        # ---- Stacked TriangleBlocks ----
        for block in self.blocks:
            z = block(z, pair_mask=pair_mask)
            if pair_mask is not None:
                z = z * pair_mask.unsqueeze(-1)

        if pair_mask is not None:
            z = z * pair_mask.unsqueeze(-1)

        # ---- Axial Attention (可选 bias) ----
        if self.use_attn_bias or self.use_attn_no_bias:
            if self.use_attn_no_bias:
                # 关键对照：只有注意力、无物理 bias（隔离架构与物理的贡献）
                z = self.axial(z, bias=None, mask_1d=mask_1d)
            elif self.use_static_bias:
                # 方案 B: 静态 aa 对亲和力 bias（不依赖预测距离，信号最强）
                z = self.axial(z, bias=self.pair_pot.static_bias(aa_idx), mask_1d=mask_1d)
            else:
                z = self.axial(z, bias=-pair_energy, mask_1d=mask_1d)  # 软查表（原方案）
            if pair_mask is not None:
                z = z * pair_mask.unsqueeze(-1)

        # ---- Distance bin head (最终预测) ----
        z = z.permute(0, 3, 1, 2).contiguous()             # (B, hidden, L, L)
        logits = self.final_head(z)                         # (B, n_bins, L, L)
        logits = 0.5 * (logits + logits.transpose(-1, -2))

        # ---- Auxiliary geometry heads ----
        z_2d = z.permute(0, 2, 3, 1).contiguous()          # (B, L, L, hidden)
        if mask_1d is not None:
            col_mask = mask_1d[:, None, :, None].to(z_2d.dtype)
            denom = col_mask.sum(dim=2).clamp(min=1.0)
            feat_1d = (z_2d * col_mask).sum(dim=2) / denom  # (B, L, hidden)
        else:
            feat_1d = z_2d.mean(dim=2)                         # (B, L, hidden)

        bond_pred = None
        if L > 1:
            bond_input = torch.cat([feat_1d[:, :-1], feat_1d[:, 1:]], dim=-1)
            bond_pred = self.bond_head(bond_input).squeeze(-1)  # (B, L-1)

        angle_pred = None
        if L > 2:
            angle_input = torch.cat([feat_1d[:, :-2], feat_1d[:, 1:-1], feat_1d[:, 2:]], dim=-1)
            angle_pred = self.angle_head(angle_input).squeeze(-1)  # (B, L-2)

        return logits, init_logits, bond_pred, angle_pred

# ==========================================
# 7. Physics Losses
# ==========================================
class PhysicsLoss(nn.Module):
    def __init__(self, n_bins=N_BINS,
                 lambda_dist=1.0, lambda_stat=0.0,
                 lambda_clash=0.0, lambda_tri=0.0,
                 lambda_bond_aux=0.0, lambda_angle_aux=0.0,
                 lambda_bond_dist=0.0, lambda_angle_dist=0.0,
                 pair_pot=None, use_stat=False, tri_max_len=128,
                 stat_margin=0.5, use_phys=False, lambda_phys=0.0,
                 phys_margin=1.0):
        super().__init__()
        self.n_bins = n_bins
        self.register_buffer("bin_edges", BIN_EDGES.float())
        self.register_buffer("bin_centers", BIN_CENTERS.float())
        self.lambda_dist = lambda_dist
        self.lambda_stat = lambda_stat
        self.lambda_clash = lambda_clash
        self.lambda_tri = lambda_tri
        self.tri_max_len = tri_max_len
        self.stat_margin = stat_margin
        self.lambda_bond_aux = lambda_bond_aux
        self.lambda_angle_aux = lambda_angle_aux
        self.lambda_bond_dist = lambda_bond_dist
        self.lambda_angle_dist = lambda_angle_dist
        self.pair_pot = pair_pot
        self.use_stat = use_stat
        self.use_phys = use_phys
        self.lambda_phys = lambda_phys
        self.phys_margin = phys_margin
        self.pair_pot = pair_pot
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, bin_logits, true_dist, mask_1d=None, mask_2d=None, aa_idx=None,
                bond_pred=None, angle_pred=None, bond_labels=None, angle_labels=None):
        """
        bin_logits: (B, n_bins, L, L)  预测的距离分桶logits
        true_dist: (B, L, L)          真值距离
        mask_1d: (B, L)              有效残基mask（1表示有效，0表示padding）
        mask_2d: (B, L, L)            有效区域mask（1表示有效，0表示padding）
        aa_idx: (B, L)                氨基酸索引（用于 L_stat）
        bond_pred: (B, L-1)           辅助head预测的键长
        angle_pred: (B, L-2)          辅助head预测的键角
        bond_labels: (B, L-1)         真实键长标签
        angle_labels: (B, L-2)        真实键角标签
        """
        B, L = true_dist.shape[:2]
        device = bin_logits.device
        bin_logits = bin_logits.float()   # 损失在 fp32 计算，避免 autocast fp16 精度问题
        loss_dict = {}

        if mask_1d is None:
            mask_1d = torch.ones(B, L, device=device)
        if mask_2d is None:
            mask_2d = mask_1d.unsqueeze(-1) * mask_1d.unsqueeze(1)

        sep = torch.arange(L, device=device)
        sep = (sep[:, None] - sep[None, :]).abs()

        dist_mask = mask_2d * (sep >= 1).float()
        stat_mask = mask_2d * (sep >= 6).float()
        clash_mask = mask_2d * (sep >= 3).float()
        tri_mask_base = mask_2d * (sep >= 6).float()

        valid_dist = dist_mask.sum() + 1e-6
        valid_stat = stat_mask.sum() + 1e-6
        valid_clash = clash_mask.sum() + 1e-6

        # ---- L_dist: 距离分桶交叉熵 ----
        # 注意: Rosettosa `dist` 字段 0.0 是 far 哨兵（≥20Å）→ FAR_BIN，否则会误判为接触
        true_bin = true_dist_to_bin_idx(true_dist, self.bin_edges.to(device))
        loss_dist = self.ce(bin_logits, true_bin)
        loss_dict["L_dist"] = (loss_dist * dist_mask).sum() / valid_dist * self.lambda_dist

        # ---- L_stat: 可学习统计势 ----
        if self.use_stat and self.pair_pot is not None and aa_idx is not None:
            energy = self.pair_pot(aa_idx, bin_logits)
            contact_label = ((true_dist > 0) & (true_dist < 8.0)).float()   # 0 是 far 哨兵，非接触
            contact = contact_label * stat_mask
            noncontact = (1.0 - contact_label) * stat_mask

            loss_contact = contact * F.relu(energy + self.stat_margin)
            loss_noncontact = noncontact * F.relu(self.stat_margin - energy)
            loss_dict["L_stat"] = (loss_contact + loss_noncontact).sum() / valid_stat * self.lambda_stat

            if hasattr(self.pair_pot, "delta") and any(p.requires_grad for p in self.pair_pot.parameters()):
                loss_dict["L_pot_reg"] = 1e-4 * self.pair_pot.delta.pow(2).mean()

        # ---- L_phys: 冻结 PDB-PMF 能量约束（预测分布不应对应物理高能态）----
        if self.use_phys and self.pair_pot is not None and aa_idx is not None:
            pred_prob = F.softmax(bin_logits, dim=1)               # (B,N_BINS,L,L)
            e_pmf = self.pair_pot.e_pmf                            # (N_BINS,20,20) 冻结，无梯度
            e_flat = e_pmf.reshape(self.n_bins, -1)                # (N_BINS,400)
            aa_i = aa_idx.unsqueeze(2).expand(-1, -1, L)
            aa_j = aa_idx.unsqueeze(1).expand(-1, L, -1)
            pair_idx = aa_i * 20 + aa_j
            e_soft = (pred_prob.transpose(0, 1) * e_flat[:, pair_idx]).sum(dim=0)  # (B,L,L) 期望能量
            e_true = e_flat[true_bin, pair_idx]                    # (B,L,L) 真值 bin 能量
            phys_viol = F.relu(e_soft - e_true - self.phys_margin)
            phys_mask = mask_2d * (sep >= 6).float()
            loss_dict["L_phys"] = (phys_viol * phys_mask).sum() / (phys_mask.sum() + 1e-6) * self.lambda_phys

        # ---- L_clash: 位阻（概率版本）----
        if self.lambda_clash > 0:
            pred_prob = F.softmax(bin_logits, dim=1)
            clash_bins = (self.bin_centers < 3.5).float().view(1, -1, 1, 1).to(device)
            p_clash = (pred_prob * clash_bins).sum(dim=1)
            loss_dict["L_clash"] = (p_clash * clash_mask).sum() / valid_clash * self.lambda_clash

        # ---- L_tri: 三角不等式损失（采样版本）----
        if self.lambda_tri > 0:
            pred_dist = logits_to_pred_dist(bin_logits)
            
            if L > self.tri_max_len:
                n_samples = min(L * 32, 8192)
                
                i = torch.randint(0, L, (n_samples,), device=device)
                j = torch.randint(0, L, (n_samples,), device=device)
                k = torch.randint(0, L, (n_samples,), device=device)
                
                sep_valid = ((i - j).abs() >= 6) & ((i - k).abs() >= 6) & ((j - k).abs() >= 6)
                
                d_ij = pred_dist[:, i, j]
                d_ik = pred_dist[:, i, k]
                d_jk = pred_dist[:, j, k]
                
                tri_viol = (
                    F.relu(d_ij - d_ik - d_jk + 0.5)
                    + F.relu(d_ik - d_ij - d_jk + 0.5)
                    + F.relu(d_jk - d_ij - d_ik + 0.5)
                ) / 3.0
                
                sample_mask = sep_valid.float().unsqueeze(0)
                if mask_1d is not None:
                    sample_mask = sample_mask * mask_1d[:, i] * mask_1d[:, j] * mask_1d[:, k]
                
                loss_dict["L_tri"] = (tri_viol * sample_mask).sum() / (sample_mask.sum() + 1e-6) * self.lambda_tri
            else:
                d_ij = pred_dist.unsqueeze(-1)
                d_ik = pred_dist.unsqueeze(2)
                d_jk = pred_dist.unsqueeze(1)
                tri_viol = (
                    F.relu(d_ij - d_ik - d_jk + 0.5)
                    + F.relu(d_ik - d_ij - d_jk + 0.5)
                    + F.relu(d_jk - d_ij - d_ik + 0.5)
                ) / 3.0
                tri_mask = tri_mask_base.unsqueeze(-1) * tri_mask_base.unsqueeze(2) * tri_mask_base.unsqueeze(1)
                loss_dict["L_tri"] = (tri_viol * tri_mask).sum() / (tri_mask.sum() + 1e-6) * self.lambda_tri

        # ---- L_bond_aux: 辅助head键长损失 ----
        if self.lambda_bond_aux > 0 and bond_pred is not None and bond_labels is not None:
            if mask_1d is not None:
                bond_mask = mask_1d[:, 1:] * mask_1d[:, :-1]
                loss_bond_aux = ((bond_pred - bond_labels).pow(2) * bond_mask).sum() / (bond_mask.sum() + 1e-6)
            else:
                loss_bond_aux = (bond_pred - bond_labels).pow(2).mean()
            loss_dict["L_bond_aux"] = loss_bond_aux * self.lambda_bond_aux

        # ---- L_angle_aux: 辅助head键角损失 ----
        if self.lambda_angle_aux > 0 and angle_pred is not None and angle_labels is not None:
            if mask_1d is not None:
                angle_mask = mask_1d[:, :-2] * mask_1d[:, 1:-1] * mask_1d[:, 2:]
                loss_angle_aux = ((angle_pred - angle_labels).pow(2) * angle_mask).sum() / (angle_mask.sum() + 1e-6)
            else:
                loss_angle_aux = (angle_pred - angle_labels).pow(2).mean()
            loss_dict["L_angle_aux"] = loss_angle_aux * self.lambda_angle_aux

        # ---- L_bond_dist: distogram相邻残基距离约束（CE版本）----
        if self.lambda_bond_dist > 0 and bond_labels is not None:
            idx = torch.arange(L - 1, device=device)
            bond_logits = bin_logits[:, :, idx, idx + 1]
            
            bond_bin_idx = true_dist_to_bin_idx(bond_labels, self.bin_edges.to(device))
            
            if mask_1d is not None:
                bond_mask = mask_1d[:, 1:] * mask_1d[:, :-1]
                loss_bond_dist = (self.ce(bond_logits, bond_bin_idx) * bond_mask).sum() / (bond_mask.sum() + 1e-6)
            else:
                loss_bond_dist = self.ce(bond_logits, bond_bin_idx).mean()
            loss_dict["L_bond_dist"] = loss_bond_dist * self.lambda_bond_dist

        # ---- L_angle_dist: distogram相邻残基键角约束（cos loss）----
        if self.lambda_angle_dist > 0 and angle_labels is not None:
            pred_dist = logits_to_pred_dist(bin_logits)
            idx = torch.arange(L - 2, device=device)
            d_prev = pred_dist[:, idx, idx + 1]
            d_next = pred_dist[:, idx + 1, idx + 2]
            d_skip = pred_dist[:, idx, idx + 2]
            
            cos_angle_pred = (d_prev.pow(2) + d_next.pow(2) - d_skip.pow(2)) / (2 * d_prev * d_next + 1e-8)
            cos_angle_pred = cos_angle_pred.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            
            cos_angle_labels = torch.cos(angle_labels)
            
            if mask_1d is not None:
                angle_mask = mask_1d[:, :-2] * mask_1d[:, 1:-1] * mask_1d[:, 2:]
                loss_angle_dist = ((cos_angle_pred - cos_angle_labels).pow(2) * angle_mask).sum() / (angle_mask.sum() + 1e-6)
            else:
                loss_angle_dist = (cos_angle_pred - cos_angle_labels).pow(2).mean()
            loss_dict["L_angle_dist"] = loss_angle_dist * self.lambda_angle_dist

        return loss_dict, sum(loss_dict.values())


# ==========================================
# 8. Data Loading
# ==========================================
def load_pdnet_list(lst_name):
    lst_path = os.path.join(DATA_DIR, lst_name)
    items = []
    with open(lst_path) as f:
        for line in f:
            p = line.strip().split()
            items.append((p[0], int(p[1]) if len(p) >= 2 else None))
    print(f"   {lst_name}: {len(items)} proteins")
    return items


def load_dist(pdb_id, dtype):
    # 统一使用 Cβ 距离图（trRosetta 约定；Gly 以 Cα 兜底）
    path = os.path.join(DATA_DIR, dtype, "distance", f"{pdb_id}-cb.npy")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


def normalize_seq(seq):
    if isinstance(seq, bytes):
        return seq.decode()
    if isinstance(seq, np.ndarray):
        if seq.ndim == 0:
            return normalize_seq(seq.item())
        return "".join(x.decode() if isinstance(x, bytes) else str(x) for x in seq.tolist())
    return str(seq)


def load_fasta(path):
    """读取 fasta -> {header: seq}。"""
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


def _id_of(x):
    """列表元素可能是 (pid, len) 元组（PDNet）或纯字符串 ID（trRosetta）。"""
    return x[0] if isinstance(x, (tuple, list)) else x



class ContactDataset(Dataset):
    """从距离矩阵派生 bond/angle 标签的 Dataset"""
    def __init__(self, pdb_list, dtype="deepcov", crop=256):
        self.samples = []
        for pid, _ in tqdm(pdb_list, desc=f"{dtype}"):
            r = load_dist(pid, dtype)
            if r is None:
                continue

            L = int(r[0])
            seq_s = normalize_seq(r[1])
            dmap = r[2].astype(np.float32)

            if L < 20 or L > 1024:
                continue

            if len(seq_s) != L:
                continue

            if dmap.shape != (L, L):
                continue

            bond_len = np.diag(dmap, k=1).astype(np.float32)
            if L >= 3:
                d1 = bond_len[:-1]
                d2 = bond_len[1:]
                d3 = np.diag(dmap, k=2).astype(np.float32)
                denom = 2.0 * d1 * d2
                cos_angle = (d1**2 + d2**2 - d3**2) / (denom + 1e-8)
                cos_angle = np.clip(cos_angle, -1.0 + 1e-6, 1.0 - 1e-6)
                bond_angle = np.arccos(cos_angle).astype(np.float32)
            else:
                bond_angle = None
            
            bond_len = np.nan_to_num(bond_len, nan=3.8, posinf=3.8, neginf=3.8)
            if bond_angle is not None:
                bond_angle = np.nan_to_num(bond_angle, nan=1.92, posinf=1.92, neginf=1.92)
            
            sample = {
                "pdb_id": pid, "seq": seq_s, "L": L,
                "dist": dmap,
                "bond_len": bond_len, "bond_angle": bond_angle,
            }
            self.samples.append(sample)
        self.crop = crop
        print(f"   {dtype}: {len(self.samples)} proteins")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        L = s["L"]
        full_seq = s["seq"]
        seq_crop = full_seq
        dist = s["dist"]
        cs = 0
        if self.crop and L > self.crop:
            cs = np.random.randint(0, L - self.crop + 1)
            seq_crop = full_seq[cs:cs+self.crop]
            dist = dist[cs:cs+self.crop, cs:cs+self.crop]
        
        bond_labels = None
        angle_labels = None
        if "bond_len" in s and s["bond_len"] is not None:
            bond_labels = s["bond_len"].astype(np.float32)
            if self.crop and L > self.crop:
                bond_labels = bond_labels[cs:cs+self.crop-1]
        if "bond_angle" in s and s["bond_angle"] is not None:
            angle_labels = s["bond_angle"].astype(np.float32)
            if self.crop and L > self.crop:
                angle_labels = angle_labels[cs:cs+self.crop-2]
        
        return {"pdb_id": s["pdb_id"], "full_seq": full_seq, "seq": seq_crop,
                "dist": dist,
                "bond_labels": bond_labels, "angle_labels": angle_labels,
                "crop": cs, "len": len(seq_crop)}


class TRRosettaDataset(ContactDataset):
    """
    trRosetta 训练集（Cβ 37-bin 距离，far 哨兵）。样本结构同 ContactDataset，
    复用其 __getitem__（crop）与模块级 collate。

    每个 npz:
      dist    (L,L) Cβ-Cβ 距离，0-20Å 为真实值，≥20Å 全编码成 0.0（far 哨兵）
      mask1d  (L,)  有效残基 mask
    序列来自 15051.fasta；只保留 mask1d=True 的有效残基子序列。
    训练时 dist==0 由 true_dist_to_bin_idx 映射到 FAR_BIN（见 PhysicsLoss）。
    """
    def __init__(self, ids, npz_dir, fasta_seq, crop=256, max_L=1024):
        self.samples = []
        for pid in tqdm(ids, desc="trRosetta"):
            try:
                x = np.load(os.path.join(npz_dir, f"{pid}.npz"), allow_pickle=True)
            except Exception:
                continue
            seq_full = fasta_seq.get(pid, "")
            mask = x["mask1d"].astype(bool)
            dmap = x["dist"].astype(np.float32)             # Cβ，0 = far 哨兵
            if mask.sum() < 20:
                continue
            seq_s = "".join(c for c, m in zip(seq_full, mask) if m)
            dmap = dmap[mask][:, mask]
            L = dmap.shape[0]
            if L != len(seq_s) or not (20 <= L <= max_L):
                continue
            if dmap.shape != (L, L):
                continue
            # bond/angle 从距离矩阵派生（同 ContactDataset）
            bond_len = np.diag(dmap, k=1).astype(np.float32)
            if L >= 3:
                d1, d2 = bond_len[:-1], bond_len[1:]
                d3 = np.diag(dmap, k=2).astype(np.float32)
                denom = 2.0 * d1 * d2
                cos_angle = (d1 ** 2 + d2 ** 2 - d3 ** 2) / (denom + 1e-8)
                cos_angle = np.clip(cos_angle, -1.0 + 1e-6, 1.0 - 1e-6)
                bond_angle = np.arccos(cos_angle).astype(np.float32)
            else:
                bond_angle = None
            bond_len = np.nan_to_num(bond_len, nan=3.8, posinf=3.8, neginf=3.8)
            if bond_angle is not None:
                bond_angle = np.nan_to_num(bond_angle, nan=1.92, posinf=1.92, neginf=1.92)
            self.samples.append({
                "pdb_id": pid, "seq": seq_s, "L": L,
                "dist": dmap, "bond_len": bond_len, "bond_angle": bond_angle,
            })
        self.crop = crop
        print(f"   trRosetta: {len(self.samples)} proteins")


class TestNPZDataset(ContactDataset):
    """外部测试集（统一 npz: seq + dist(Cβ截断,0=far哨兵) + mask1d）。
    复用 ContactDataset.__getitem__ 与模块级 collate，供 --eval-external-test 评估。"""
    def __init__(self, ids, npz_dir, crop=None, max_L=1024):
        self.samples = []
        for pid in tqdm(ids, desc="Test"):
            try:
                x = np.load(os.path.join(npz_dir, f"{pid}.npz"), allow_pickle=True)
            except Exception:
                continue
            seq = str(x["seq"])
            dist = x["dist"].astype(np.float32)          # Cβ 截断 20Å，0 = far 哨兵
            L = len(seq)
            if not (20 <= L <= max_L) or dist.shape != (L, L):
                continue
            self.samples.append({"pdb_id": pid, "seq": seq, "L": L, "dist": dist})
        self.crop = crop
        print(f"   TestNPZDataset: {len(self.samples)} proteins")


def collate(batch):
    batch = sorted(batch, key=lambda x: x["len"], reverse=True)
    maxL = batch[0]["len"]
    dists, ids, lens, seqs, full_seqs, crops = [], [], [], [], [], []
    bond_labels_list, angle_labels_list = [], []
    
    for item in batch:
        lens.append(item["len"]); ids.append(item["pdb_id"])
        seqs.append(item["seq"]); full_seqs.append(item["full_seq"])
        crops.append(item["crop"])
        d = torch.from_numpy(item["dist"])
        if maxL > item["len"]:
            d = F.pad(d, (0, maxL-item["len"], 0, maxL-item["len"]))
        dists.append(d)
        
        if item["bond_labels"] is not None:
            bl = torch.from_numpy(item["bond_labels"])
            if maxL - 1 > len(bl):
                bl = F.pad(bl, (0, maxL - 1 - len(bl)))
            bond_labels_list.append(bl)
        else:
            bond_labels_list.append(torch.zeros(maxL - 1))
        
        if item["angle_labels"] is not None:
            al = torch.from_numpy(item["angle_labels"])
            if maxL - 2 > len(al):
                al = F.pad(al, (0, maxL - 2 - len(al)))
            angle_labels_list.append(al)
        else:
            angle_labels_list.append(torch.zeros(maxL - 2))
    
    dists = torch.stack(dists)
    bond_labels = torch.stack(bond_labels_list) if bond_labels_list else None
    angle_labels = torch.stack(angle_labels_list) if angle_labels_list else None
    
    mask_1d = (torch.arange(maxL)[None, :] < torch.tensor(lens)[:, None]).float()
    mask_2d = mask_1d.unsqueeze(-1) * mask_1d.unsqueeze(1)
    
    return {"pdb_id": ids, "lens": lens, "crops": crops, "seq": seqs,
            "full_seq": full_seqs,
            "dist": dists, "mask_1d": mask_1d, "mask_2d": mask_2d,
            "bond_labels": bond_labels, "angle_labels": angle_labels}


# ==========================================
# 9. Metrics
# ==========================================
def calc_precision(pred, true, sep_min=12, k_list=[1, 2, 5, 10]):
    L = len(pred)
    mask = np.triu(np.ones((L, L), dtype=bool), k=sep_min)
    pf, tf = pred[mask], true[mask]
    pf = np.nan_to_num(pf, nan=1e6, posinf=1e6, neginf=0.0)
    
    if len(pf) == 0:
        return {f"P@L/{k}": 0.0 for k in k_list}
    
    order = np.argsort(pf)
    return {
        f"P@L/{k}": float(tf[order[:min(max(1, L // k), len(order))]].mean())
        for k in k_list
    }

def batch_to_embeddings(batch):
    """从 batch 取出 ESM 嵌入，crop/pad 对齐到 maxL。返回 (emb, maxL)。"""
    maxL = max(batch["lens"])
    embs = []
    for i, pid in enumerate(batch["pdb_id"]):
        item = get_emb(pid, batch["full_seq"][i])
        if item is None:
            raise KeyError(f"Missing ESM cache for {pid}")
        et = torch.from_numpy(np.array(item["embedding"], dtype=np.float32))
        cs = batch["crops"][i]
        Lt = batch["lens"][i]
        et = et[cs:cs + Lt]
        if maxL > Lt:
            et = F.pad(et, (0, 0, 0, maxL - Lt))
        embs.append(et)
    return torch.stack(embs).to(DEVICE), maxL


# ==========================================
# 10. Trainer
# ==========================================
class Trainer:
    def __init__(self, model, lr=5e-4, wd=1e-5, init_loss_weight=0.1, max_eval_len=512,
                 n_epochs=30, **loss_kwargs):
        self.model = model
        self.init_loss_weight = init_loss_weight
        self.max_eval_len = max_eval_len

        other_params = []
        pot_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("pair_pot."):
                pot_params.append(p)
            else:
                other_params.append(p)

        params = [{"params": other_params}]
        if pot_params:
            params.append({"params": pot_params, "lr": 1e-4})
            print(f"   Learned potential params: {sum(p.numel() for p in pot_params):,}")

        self.opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=n_epochs,
                                                                 eta_min=1e-5)

        self.phys = PhysicsLoss(**loss_kwargs).to(DEVICE)
        self.n_bins = model.n_bins
        self.use_cuda = DEVICE == "cuda"
        self.scaler = torch.cuda.amp.GradScaler() if self.use_cuda else None

    def train_epoch(self, loader):
        self.model.train()
        total = 0
        n_steps = 0

        for batch in tqdm(loader, desc="Train"):
            emb, maxL = batch_to_embeddings(batch)
            td = batch["dist"].to(DEVICE)

            aa_idx = seq_to_idx(batch["seq"], maxL, DEVICE)

            mask_1d = batch["mask_1d"].to(DEVICE)
            mask_2d = batch["mask_2d"].to(DEVICE)
            
            bond_labels = batch["bond_labels"].to(DEVICE) if batch["bond_labels"] is not None else None
            angle_labels = batch["angle_labels"].to(DEVICE) if batch["angle_labels"] is not None else None
            
            if self.use_cuda:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    bin_logits, init_logits, bond_pred, angle_pred = self.model(emb, aa_idx, mask_1d=mask_1d)
                    ld_main, loss_main = self.phys(bin_logits, td, mask_1d=mask_1d, mask_2d=mask_2d,
                                                   aa_idx=aa_idx,
                                                   bond_pred=bond_pred, angle_pred=angle_pred,
                                                   bond_labels=bond_labels, angle_labels=angle_labels)
                    ld_init, loss_init = self.phys(init_logits, td, mask_1d=mask_1d, mask_2d=mask_2d,
                                                   aa_idx=aa_idx)
                    loss = loss_main + self.init_loss_weight * loss_init
            else:
                bin_logits, init_logits, bond_pred, angle_pred = self.model(emb, aa_idx, mask_1d=mask_1d)
                ld_main, loss_main = self.phys(bin_logits, td, mask_1d=mask_1d, mask_2d=mask_2d,
                                               aa_idx=aa_idx,
                                               bond_pred=bond_pred, angle_pred=angle_pred,
                                               bond_labels=bond_labels, angle_labels=angle_labels)
                ld_init, loss_init = self.phys(init_logits, td, mask_1d=mask_1d, mask_2d=mask_2d,
                                               aa_idx=aa_idx)
                loss = loss_main + self.init_loss_weight * loss_init

            if not torch.isfinite(loss):
                print("[DEBUG train] non-finite loss, skip step")
                self.opt.zero_grad(set_to_none=True)
                continue

            self.opt.zero_grad()
            if self.use_cuda:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
            total += loss.item()
            n_steps += 1

        return total / max(n_steps, 1)

    @torch.no_grad()
    def evaluate(self, loader, return_raw=False):
        self.model.eval()
        metrics = {}
        n_skipped = 0
        for batch in tqdm(loader, desc="Eval"):
            maxL = max(batch["lens"])
            if maxL > self.max_eval_len:
                n_skipped += 1
                continue
            emb, _ = batch_to_embeddings(batch)
            td = batch["dist"].to(DEVICE)
            aa_idx = seq_to_idx(batch["seq"], maxL, DEVICE)
            mask_1d = batch["mask_1d"].to(DEVICE)
            mask_2d = batch["mask_2d"].to(DEVICE)

            with torch.no_grad():
                if DEVICE == "cuda":
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        bin_logits, _, _, _ = self.model(emb, aa_idx, mask_1d=mask_1d)
                else:
                    bin_logits, _, _, _ = self.model(emb, aa_idx, mask_1d=mask_1d)
            pred_dist = logits_to_pred_dist(bin_logits)

            # ---- ① 接触精度 P@L/k（用预测距离排序，非二值化） ----
            pred_score = pred_dist.float().cpu().numpy()      # 值越小 = 接触置信度越高
            true_contact = ((td > 0) & (td < 8.0)).float().cpu().numpy()  # 0 是 far 哨兵，排除

            for i, L_i in enumerate(batch["lens"]):
                m = calc_precision(pred_score[i, :L_i, :L_i],
                                   true_contact[i, :L_i, :L_i])
                for k, v in m.items():
                    metrics.setdefault(k, []).append(v)

            # ---- ② MAE（排除对角线 + padding + far 哨兵 0；远距离只报分类准确率）----
            pred_bin = bin_logits.argmax(dim=1)                # (B, L, L) 预测 bin
            for i, L_i in enumerate(batch["lens"]):
                sep = (torch.arange(L_i, device=DEVICE)[:, None] -
                       torch.arange(L_i, device=DEVICE)[None, :]).abs() >= 1
                pd_i = pred_dist[i, :L_i, :L_i]
                td_i = td[i, :L_i, :L_i]
                valid = (td_i > 0) & sep                        # 真实子-20Å 距离
                far_cells = (td_i == 0) & sep                   # far 哨兵（≥20Å）
                diff = (pd_i - td_i).abs()

                def safe_mean(d, m):
                    cnt = m.sum().clamp(min=1)
                    return (d * m).sum() / cnt

                metrics.setdefault("MAE_all_valid", []).append(safe_mean(diff, valid).item())
                metrics.setdefault("MAE_contact_range", []).append(safe_mean(diff, valid & (td_i < 8.0)).item())
                metrics.setdefault("MAE_mid", []).append(safe_mean(diff, valid & (td_i >= 8.0)).item())
                # far 哨兵格子: 模型是否判成 far（argmax == FAR_BIN）→ 分类准确率
                far_hit = (pred_bin[i, :L_i, :L_i] == FAR_BIN) & far_cells
                metrics.setdefault("far_acc", []).append((far_hit.sum() / far_cells.sum().clamp(min=1)).item())

            # ---- ③ 局部几何误差（直接从 pred_dist 计算，无 MDS）----
            for i, L_i in enumerate(batch["lens"]):
                if L_i > self.max_eval_len:
                    continue
                pd_i = pred_dist[i, :L_i, :L_i]
                td_i = td[i, :L_i, :L_i]

                if L_i > 1:
                    idx = torch.arange(L_i - 1, device=DEVICE)
                    pred_bond = pd_i[idx, idx + 1]
                    true_bond = td_i[idx, idx + 1]
                    bm = (true_bond > 0).float()                # 排除 far 哨兵
                    metrics.setdefault("bond_err_dist", []).append(
                        (((pred_bond - true_bond).abs() * bm).sum() / bm.sum().clamp(min=1)).item()
                    )

                if L_i > 2:
                    idx = torch.arange(L_i - 2, device=DEVICE)
                    d_prev = pd_i[idx, idx + 1]
                    d_next = pd_i[idx + 1, idx + 2]
                    d_skip = pd_i[idx, idx + 2]
                    cos_pred = (d_prev.pow(2) + d_next.pow(2) - d_skip.pow(2)) / (2 * d_prev * d_next + 1e-8)
                    cos_pred = cos_pred.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                    pred_angle = torch.acos(cos_pred)

                    true_d_prev = td_i[idx, idx + 1]
                    true_d_next = td_i[idx + 1, idx + 2]
                    true_d_skip = td_i[idx, idx + 2]
                    cos_true = (true_d_prev.pow(2) + true_d_next.pow(2) - true_d_skip.pow(2)) / (2 * true_d_prev * true_d_next + 1e-8)
                    cos_true = cos_true.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                    true_angle = torch.acos(cos_true)

                    am = ((true_d_prev > 0) & (true_d_next > 0) & (true_d_skip > 0)).float()  # 排除 far 哨兵
                    metrics.setdefault("angle_err_dist", []).append(
                        (((pred_angle - true_angle).abs() * am).sum() / am.sum().clamp(min=1)).item()
                    )

            # ---- ④ 位阻违例率 ----
            sep_mask = (torch.arange(maxL, device=DEVICE).unsqueeze(1) -
                        torch.arange(maxL, device=DEVICE).unsqueeze(0)).abs() >= 3
            for i, L_i in enumerate(batch["lens"]):
                pd_i = pred_dist[i, :L_i, :L_i]
                mask = sep_mask[:L_i, :L_i]
                clash_hits = ((pd_i < 3.0) & mask).float().sum()
                clash = (clash_hits / mask.float().sum().clamp(min=1)).item()
                metrics.setdefault("clash_rate", []).append(clash)
                
                pred_prob_i = F.softmax(bin_logits[i, :, :L_i, :L_i], dim=0)
                clash_bins = (BIN_CENTERS < 3.5).float().to(DEVICE)
                p_clash_i = (pred_prob_i * clash_bins.view(-1, 1, 1)).sum(dim=0)
                p_clash_mean = (p_clash_i * mask).sum() / mask.sum().clamp(min=1)
                metrics.setdefault("p_clash_mean", []).append(p_clash_mean.item())

            # ---- ⑤ 三角不等式违例率（采样版本，sep>=6） ----
            for i, L_i in enumerate(batch["lens"]):
                d = pred_dist[i, :L_i, :L_i]
                
                if L_i > 128:
                    n_samples = min(L_i * 16, 4096)
                    idx_i = torch.randint(0, L_i, (n_samples,), device=DEVICE)
                    idx_j = torch.randint(0, L_i, (n_samples,), device=DEVICE)
                    idx_k = torch.randint(0, L_i, (n_samples,), device=DEVICE)
                    
                    sep_valid = ((idx_i - idx_j).abs() >= 6) & ((idx_i - idx_k).abs() >= 6) & ((idx_j - idx_k).abs() >= 6)
                    
                    d_ij = d[idx_i, idx_j]
                    d_ik = d[idx_i, idx_k]
                    d_jk = d[idx_j, idx_k]
                    
                    tri_v = (
                        F.relu(d_ij - d_ik - d_jk + 0.5)
                        + F.relu(d_ik - d_ij - d_jk + 0.5)
                        + F.relu(d_jk - d_ij - d_ik + 0.5)
                    ) / 3.0
                    
                    sample_mask = sep_valid.float()
                    n_tri = sample_mask.sum().clamp(min=1)
                    metrics.setdefault("tri_viol", []).append(((tri_v * sample_mask).sum() / n_tri).item())
                else:
                    tri_sep = ((torch.arange(L_i, device=DEVICE)[:, None] -
                                torch.arange(L_i, device=DEVICE)[None, :]).abs() >= 6).float()
                    d_ij = (d * tri_sep).unsqueeze(-1)
                    d_ik = (d * tri_sep).unsqueeze(2)
                    d_jk = (d * tri_sep).unsqueeze(1)
                    tri_mask = tri_sep.unsqueeze(-1) * tri_sep.unsqueeze(2) * tri_sep.unsqueeze(1)
                    tri_v = (
                        F.relu(d_ij - d_ik - d_jk + 0.5)
                        + F.relu(d_ik - d_ij - d_jk + 0.5)
                        + F.relu(d_jk - d_ij - d_ik + 0.5)
                    ) / 3.0 * tri_mask
                    n_tri = tri_mask.sum().clamp(min=1)
                    metrics.setdefault("tri_viol", []).append((tri_v.sum() / n_tri).item())

        if n_skipped > 0:
            print(f"   [Eval] Skipped {n_skipped} proteins exceeding max_eval_len={self.max_eval_len}")
        if return_raw:
            return metrics
        return {k: float(np.nanmean(v)) for k, v in metrics.items()}


# ==========================================
# 11. CSV Logger
# ==========================================
class CSVLogger:
    def __init__(self, filepath, cmd=""):
        self.filepath = filepath
        self.cmd = cmd
        self.header_written = False
        self.fields = []

    def log(self, **kwargs):
        import csv
        import os

        if not self.header_written:
            self.fields = list(kwargs.keys())
            file_exists = os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0
            with open(self.filepath, 'a', newline='') as f:
                if file_exists:
                    f.write("\n")
                if self.cmd:
                    f.write(f"# command: {self.cmd}\n")
                writer = csv.writer(f)
                writer.writerow(self.fields)
            self.header_written = True

        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([kwargs.get(k, '') for k in self.fields])
    
    def __str__(self):
        return f"CSVLogger({self.filepath})"


# ==========================================
# 12. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--init-loss-weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-eval-len", type=int, default=512,
                        help="评估时跳过长于此的蛋白，防止OOM")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--pot-mode", type=str, default="pmf",
                        choices=["pmf", "pmf_only", "learned"],
                        help="成对势基座: pmf(冻结PDB先验+残差) / pmf_only(全冻结纯先验) / learned(旧可学习势)")
    parser.add_argument("--pmf", type=str, default="",
                        help="compute_pmf.py 输出的 PMF 路径（默认 ../new1/pmf37_cb.npz，37-bin Cβ）")
    parser.add_argument("--pot-perturb-seed", type=int, default=None,
                        help="若设置，打乱 PMF 氨基酸标签（随机先验扰动实验）")
    parser.add_argument("--phys-loss", action="store_true",
                        help="冻结 PDB-PMF 能量约束损失 L_phys")
    parser.add_argument("--lambda-phys", type=float, default=0.1,
                        help="L_phys 权重")
    parser.add_argument("--phys-margin", type=float, default=1.0,
                        help="L_phys hinge margin")
    parser.add_argument("--num-blocks", type=int, default=4,
                        help="Number of stacked TriangleBlocks (Evoformer-style)")

    # 消融开关
    parser.add_argument("--no-ngram", action="store_true", help="Disable n-gram")
    parser.add_argument("--no-triangle", action="store_true",
                        help="Disable triangle multiplication")

    # 物理注入方式
    parser.add_argument("--learned-pot", action="store_true",
                        help="Enable L_stat with learned potential")
    parser.add_argument("--freeze-pot", action="store_true",
                        help="Freeze learned potential parameters (for ablation)")
    parser.add_argument("--stat-margin", type=float, default=0.5,
                        help="Margin for L_stat hinge loss (default: 0.5)")
    parser.add_argument("--lambda-stat", type=float, default=0.05,
                        help="Weight for L_stat loss (default: 0.05)")
    parser.add_argument("--feature-gate", action="store_true",
                        help="Enable feature-level gating")
    parser.add_argument("--attn-bias", action="store_true",
                        help="Enable attention bias")
    parser.add_argument("--attn-bias-static", action="store_true",
                        help="用静态 aa 对亲和力 bias（-min E，不依赖预测距离），避免软查表稀释")
    parser.add_argument("--attn-no-bias", action="store_true",
                        help="关键对照：只有轴向注意力、无物理 bias（隔离架构与物理的贡献）")
    parser.add_argument("--bias-scale", type=float, default=0.05,
                        help="Scale factor for attention bias")
    parser.add_argument("--esm-model", default="esm2_t33_650M_UR50D",
                        choices=list(ESM_MODELS.keys()),
                        help="ESM 骨干: esm2_t33_650M_UR50D(默认) / esm2_t12_35M_UR50D")
    parser.add_argument("--esm-weights", default=None,
                        help="本地 ESM 权重目录（含 config.json 等）；缺省走 ModelScope 下载")
    parser.add_argument("--eval-per-protein", default=None,
                        help="外部测试评估时额外保存逐蛋白 P@L/5 到 pkl（target-level 统计用）")
    parser.add_argument("--eval-external-test", type=str, default=None,
                        help="评估外部测试集（npz 目录，--weights 必须给定）")
    parser.add_argument("--eval-test-cache", type=str, default=None,
                        help="外部测试集 ESM 缓存 pkl（prepare_test_sets.py 生成）")

    # 物理损失
    parser.add_argument("--clash", action="store_true", help="L_clash")
    parser.add_argument("--tri", action="store_true", help="L_tri")
    parser.add_argument("--tri-max-len", type=int, default=128,
                        help="Skip L_tri for sequences longer than this (default: 128)")
    
    # 几何约束新方案
    parser.add_argument("--geometry-aux", action="store_true",
                        help="Enable auxiliary bond/angle prediction heads")
    parser.add_argument("--local-dist-geometry", action="store_true",
                        help="Enable bond/angle loss from distogram local distances")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of DataLoader workers (0 for Windows compatibility)")
    parser.add_argument("--exp-name", type=str, default="",
                        help="Experiment name (used in history filename)")
    parser.add_argument("--train-size", type=int, default=None,
                        help="Number of training proteins to use (None for all)")
    parser.add_argument("--test-set", type=str, default="cameo",
                        choices=["cameo", "casp14"],
                        help="Test set to use (仅 pdnet 数据源)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (default: D:\\数据集蛋白质\\PDNET\\data)")
    parser.add_argument("--data-source", type=str, default="trrosetta",
                        choices=["trrosetta", "pdnet"],
                        help="训练数据源: trrosetta(Cβ 37-bin,默认) / pdnet(Cβ)")
    parser.add_argument("--trr-npz-dir", type=str, default=r"D:\数据集蛋白质\Rosettosa\npz")
    parser.add_argument("--trr-fasta", type=str, default=r"D:\数据集蛋白质\Rosettosa\15051.fasta")
    parser.add_argument("--trr-cache", type=str, default=None,
                        help="trRosetta ESM 嵌入缓存 pkl（缺省按 --esm-model 自动命名）")
    parser.add_argument("--trr-valid-size", type=int, default=200)
    parser.add_argument("--trr-test-size", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    global DATA_DIR
    global CACHE_FILE
    global _EMBEDDING_CACHE
    global ESM_DIM, _ESM_MODEL_ID, _ESM_WEIGHTS_DIR
    _ESM_MODEL_ID = args.esm_model
    _ESM_WEIGHTS_DIR = args.esm_weights
    ESM_DIM = ESM_MODELS[args.esm_model]["dim"]
    if args.trr_cache is None:
        args.trr_cache = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "esm2_cache",
            f"esm2_trrosetta_{ESM_MODELS[args.esm_model]['tag']}_ca.pkl")
    if args.data_dir is not None:
        DATA_DIR = args.data_dir
    print(f"[INFO] Data directory: {DATA_DIR}")

    if not args.pmf:
        args.pmf = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "new1",
            "pmf37_cb.npz")
    print(f"[INFO] PMF: {args.pmf}")

    print("=" * 60)
    print("ESM2 + 物理约束距离预测")
    print(f"  {args.epochs}epochs, lr={args.lr}, crop={args.crop}, batch={args.batch}, seed={args.seed}")
    ablate = []
    if args.no_ngram: ablate.append("no-ngram")
    if args.no_triangle: ablate.append("no-triangle")
    physics = []
    if args.learned_pot: physics.append("L_stat")
    if args.feature_gate: physics.append("feature-gate")
    if args.attn_bias: physics.append("attn-bias")
    if args.clash: physics.append("L_clash")
    if args.tri: physics.append("L_tri")
    if args.geometry_aux: physics.append("geometry-aux")
    if args.local_dist_geometry: physics.append("local-dist-geometry")
    if args.phys_loss: physics.append("L_phys")
    print(f"  Ablations: {ablate if ablate else 'none'}")
    print(f"  Physics: {physics if physics else 'none (baseline)'}")
    print("=" * 60)

    # ---- A. Prepare model ----
    use_any_potential = (args.learned_pot or args.feature_gate or args.attn_bias
                         or args.attn_bias_static or args.phys_loss)

    # 加载 PDB-PMF（mode='learned' 时不需要）
    pmf_tensor = None
    if args.pot_mode in ("pmf", "pmf_only"):
        if not os.path.exists(args.pmf):
            parser.error(f"找不到 pmf.npz: {args.pmf}\n请先运行 compute_pmf.py 生成")
        print(f"[PMF] 加载 {args.pmf} ...")
        pmf_tensor = load_pmf_potential(args.pmf)
        print(f"   PMF 范围: [{pmf_tensor.min():.2f}, {pmf_tensor.max():.2f}]")

    model = ProteinPredictor(
        esm_dim=ESM_DIM, hidden=64, n_bins=N_BINS,
        use_ngram=not args.no_ngram,
        use_triangle=not args.no_triangle,
        use_feature_gate=args.feature_gate,
        use_attn_bias=args.attn_bias or args.attn_bias_static,
        use_static_bias=args.attn_bias_static,
        use_attn_no_bias=args.attn_no_bias,
        pot_mode=args.pot_mode,
        pmf=pmf_tensor,
        pot_perturb_seed=args.pot_perturb_seed,
        num_blocks=args.num_blocks,
        bias_scale=args.bias_scale,
    ).to(DEVICE)

    if args.freeze_pot:
        for p in model.pair_pot.parameters():
            p.requires_grad = False
        print("   [INFO] Pair potential frozen by --freeze-pot")
    elif not use_any_potential:
        for p in model.pair_pot.parameters():
            p.requires_grad = False
        print("   [INFO] Pair potential frozen (baseline mode)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n   Model params: {trainable:,}")

    # ---- A.5 外部测试集评估（--eval-external-test + --weights）----
    if args.eval_external_test:
        if not args.weights:
            parser.error("--eval-external-test 需要 --weights 指定训练好的 checkpoint")
        if not args.eval_test_cache or not os.path.exists(args.eval_test_cache):
            parser.error(f"找不到 ESM 缓存: {args.eval_test_cache}（用 prepare_test_sets.py 生成）")
        model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
        with open(args.eval_test_cache, "rb") as f:
            _EMBEDDING_CACHE = pickle.load(f)
        ids = sorted(f[:-4] for f in os.listdir(args.eval_external_test)
                     if f.endswith(".npz"))
        test_ds = TestNPZDataset(ids, args.eval_external_test)
        loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                            collate_fn=collate, num_workers=0)
        eval_trainer = Trainer(
            model, lr=args.lr, init_loss_weight=args.init_loss_weight,
            max_eval_len=args.max_eval_len, n_epochs=args.epochs,
            n_bins=N_BINS, lambda_dist=1.0,
            lambda_stat=args.lambda_stat if args.learned_pot else 0.0,
            lambda_clash=0.02 if args.clash else 0.0,
            lambda_tri=0.1 if args.tri else 0.0,
            lambda_bond_aux=0.1 if args.geometry_aux else 0.0,
            lambda_angle_aux=0.05 if args.geometry_aux else 0.0,
            lambda_bond_dist=0.05 if args.local_dist_geometry else 0.0,
            lambda_angle_dist=0.05 if args.local_dist_geometry else 0.0,
            pair_pot=model.pair_pot if use_any_potential else None,
            use_stat=args.learned_pot, tri_max_len=args.tri_max_len,
            stat_margin=args.stat_margin,
        )
        print(f"\n[External Test] {os.path.basename(args.eval_external_test)}: "
              f"{len(ids)} npz, 有效 {len(test_ds)}")
        raw_metrics = eval_trainer.evaluate(loader, return_raw=True)
        metrics = {k: float(np.nanmean(v)) for k, v in raw_metrics.items()}
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        if args.eval_per_protein:
            ids_eval = [s["pdb_id"] for s in test_ds.samples
                        if s["L"] <= eval_trainer.max_eval_len]
            vals = raw_metrics.get("P@L/5", [])
            if len(ids_eval) != len(vals):
                print(f"[warn] per-protein alignment mismatch ids={len(ids_eval)} p5={len(vals)}")
            with open(args.eval_per_protein, "wb") as f:
                pickle.dump({"ids": ids_eval, "P@L/5": vals}, f)
            print(f"[per-protein] saved {len(ids_eval)} targets -> {args.eval_per_protein}")
        return

    # ---- B. Load data ----
    rng = np.random.default_rng(args.seed)
    test_set_name = args.test_set

    if args.data_source == "trrosetta":
        CACHE_FILE = args.trr_cache
        trr_ids = sorted(os.path.splitext(os.path.basename(f))[0]
                         for f in glob.glob(os.path.join(args.trr_npz_dir, "*.npz")))
        fasta = load_fasta(args.trr_fasta)
        rng.shuffle(trr_ids)
        n_valid, n_test = args.trr_valid_size, args.trr_test_size
        vlist = trr_ids[:n_valid]
        test_ids = trr_ids[n_valid:n_valid + n_test]
        tlist = trr_ids[n_valid + n_test:]
        if args.train_size is not None:
            tlist = tlist[:args.train_size]
            print(f"[Data] Training set size limited to {args.train_size}")
        print(f"[trRosetta] total={len(trr_ids)} train={len(tlist)} "
              f"valid={len(vlist)} test={len(test_ids)}")
        print("\n[Data] Loading trRosetta datasets ...")
        datasets = {
            "train": TRRosettaDataset(tlist, args.trr_npz_dir, fasta, crop=args.crop),
            "valid": TRRosettaDataset(vlist, args.trr_npz_dir, fasta, crop=None),
        }
        if test_ids:
            datasets[test_set_name] = TRRosettaDataset(
                test_ids, args.trr_npz_dir, fasta, crop=None)
        cache_list = [datasets["train"], datasets["valid"]] + \
                     ([datasets[test_set_name]] if test_ids else [])
    else:
        print("\n[Data] Loading lists ...")
        all_dc = load_pdnet_list("deepcov.lst")
        rng.shuffle(all_dc)
        vlist, tlist = all_dc[:100], all_dc[100:]
        if args.train_size is not None:
            tlist = tlist[:args.train_size]
            print(f"[Data] Training set size limited to {args.train_size}")
        split_suffix = f"_{args.exp_name}" if args.exp_name else ""
        split_path = os.path.join(CACHE_DIR, f"split_seed{args.seed}{split_suffix}.txt")
        with open(split_path, 'w') as f:
            f.write(f"# Seed: {args.seed}\n")
            f.write(f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# Valid set:\n")
            for pid in vlist:
                f.write(f"valid\t{_id_of(pid)}\n")
            f.write("# Train set:\n")
            for pid in tlist:
                f.write(f"train\t{_id_of(pid)}\n")
        print(f"   [Data] Split saved to: {split_path}")
        test_psicov = load_pdnet_list("psicov.lst")
        if args.test_set == "casp14":
            test_set_data = load_pdnet_list("casp14.lst")
            test_set_dtype = "casp14"
        else:
            test_set_data = load_pdnet_list("cameo-hard.lst")
            test_set_dtype = "cameo"
        print("\n[Data] Loading datasets ...")
        datasets = {}
        for name, lst, dtype in [
            ("train", tlist, "deepcov"),
            ("valid", vlist, "deepcov"),
            ("psicov", test_psicov, "psicov"),
            (test_set_name, test_set_data, test_set_dtype),
        ]:
            datasets[name] = ContactDataset(
                lst, dtype, crop=args.crop if name == "train" else None)
        cache_list = [datasets["train"], datasets["valid"],
                      datasets["psicov"], datasets[test_set_name]]

    # ---- C. Cache ESM2 embeddings ----
    cache_embeddings(cache_list)

    print(f"\n[Data] Final dataset sizes:")
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds)}")

    # ---- D. Build loaders ----
    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch, shuffle=True,
        collate_fn=collate, num_workers=args.num_workers)
    valid_loader = DataLoader(
        datasets["valid"], batch_size=1, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers)

    # ---- E. Init trainer ----
    trainer = Trainer(
        model, lr=args.lr,
        init_loss_weight=args.init_loss_weight,
        max_eval_len=args.max_eval_len,
        n_epochs=args.epochs,
        n_bins=N_BINS,
        lambda_dist=1.0,
        lambda_stat=args.lambda_stat if args.learned_pot else 0.0,
        lambda_clash=0.02 if args.clash else 0.0,
        lambda_tri=0.1 if args.tri else 0.0,
        lambda_bond_aux=0.1 if args.geometry_aux else 0.0,
        lambda_angle_aux=0.05 if args.geometry_aux else 0.0,
        lambda_bond_dist=0.05 if args.local_dist_geometry else 0.0,
        lambda_angle_dist=0.05 if args.local_dist_geometry else 0.0,
        pair_pot=model.pair_pot if (args.learned_pot or args.feature_gate or args.attn_bias or args.phys_loss) else None,
        use_stat=args.learned_pot,
        tri_max_len=args.tri_max_len,
        stat_margin=args.stat_margin,
        use_phys=args.phys_loss,
        lambda_phys=args.lambda_phys,
        phys_margin=args.phys_margin,
    )

    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    if args.eval_only:
        print(trainer.evaluate(valid_loader))
        return

    # ---- F. Train ----
    print(f"\n[Train] {args.epochs} epochs ...")
    
    name_base = f"esm2{args.exp_name}" if args.exp_name else "esm2"
    history_path = os.path.join(OUTPUT_DIR, f"{name_base}_history_seed{args.seed}.csv")
    best_path = os.path.join(OUTPUT_DIR, f"{name_base}_best_seed{args.seed}.pt")
    cmd_str = "python " + " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    logger = CSVLogger(history_path, cmd=cmd_str)
    print(f"   [LOG] Training history → {history_path}")
    print(f"   [LOG] Best checkpoint → {best_path}")
    
    best_p50 = -1.0
    patience = 0
    early_stop_patience = 8
    for epoch in range(1, args.epochs + 1):
        loss = trainer.train_epoch(train_loader)
        trainer.sched.step()

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            val = trainer.evaluate(valid_loader)
            print(f"   Epoch {epoch}: loss={loss:.4f}  {val}")
            
            logger.log(epoch=epoch, loss=loss, **val)
            
            if val.get("P@L/5", 0) > best_p50:
                best_p50 = val["P@L/5"]
                patience = 0
                torch.save(model.state_dict(), best_path)
                print(f"   [SAVE] P@L/5={best_p50:.4f}  →  {best_path}")
            else:
                patience += 1
                if patience >= early_stop_patience:
                    print(f"   [EARLY STOP] No improvement for {early_stop_patience} evaluations")
                    break
        else:
            print(f"   Epoch {epoch}: loss={loss:.4f}")
            logger.log(epoch=epoch, loss=loss)

    print(f"\n[DONE] Best P@L/5={best_p50:.4f}")
    print(f"   [LOG] Training history saved to {history_path}")

    # ---- G. Test ----
    if os.path.exists(best_path):
        best_w = best_path
        model.load_state_dict(torch.load(best_w, map_location=DEVICE))
    
    test_set_display_name = "CASP14" if test_set_name == "casp14" else "CAMEO"
    test_pairs = []
    if args.data_source == "pdnet":
        test_pairs = [("PSICOV", "psicov"), (test_set_display_name, test_set_name)]
    elif test_set_name in datasets:
        test_pairs = [(test_set_display_name, test_set_name)]
    for name, ds_name in test_pairs:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        loader = DataLoader(datasets[ds_name], batch_size=1, shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers)
        test_metrics = trainer.evaluate(loader)
        print(f"\n{name}: {test_metrics}")
        logger.log(epoch=f"test_{ds_name}", **test_metrics)


if __name__ == "__main__":
    main()
