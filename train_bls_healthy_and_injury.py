#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
train_bls_healthy_and_injury.py
健康→受伤 两阶段 BLS 管线（共享 CNN | 受伤仅更新 β）
+ 添加健康阶段权重检查：如果已存在则跳过健康训练，直接进行受伤阶段
"""

import os, re, math, random, glob, warnings
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch, joblib
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    precision_recall_fscore_support
)
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt, iirnotch

from dataset import EMGSlidingDataset
from model import EMGCNN, BLSIncrementalFast1
from utils import extract_features
from data_utils import load_splits, get_data_split

# ======================== 配置区 ========================
# 数据与CNN权重
HEALTHY_DATA_DIR  = r'G:\cbl-\CBAL\data'
INJURY_DATA_DIR   = r'G:\cbl-\CBAL\受伤\1'
CNN_MODEL_DIR     = r'cnn_model_output_STA_pointnorm_wuSTA'

# 输出目录
HEALTHY_OUT_DIR   = r'bls_out_healthy_final_exref1'
INJURY_OUT_DIR    = r'bls_out_injury_final_exref1'

# 通用参数
NUM_CH            = 12
SEG_LEN           = 1000
STRIDE            = 500
SKIP              = 3
BS                = 64
NUM_WORKERS       = 4
PIN_MEMORY        = True

FUSION_HIDDEN     = 960
FUSION_DROPOUT    = 0.2
FUSION_EPOCHS     = 20
LR                = 1e-3

BLS_N1            = 6000
BLS_REG           = 1e-5
BLS_NF            = BLS_N1 // 2
BLS_NH            = BLS_N1 - BLS_NF

SEED              = 42
USE_SHARED_SPLIT  = False
HEALTHY_SPLITS    = os.path.join(CNN_MODEL_DIR, 'splits.pkl')
INJURY_SPLITS     = os.path.join(CNN_MODEL_DIR, 'splits.pkl')

DEVICE            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_FOLDS         = 10
MERGE_TRAIN_VAL_FOR_BLS  = True
USE_STANDARDIZE          = True

EXCLUDE_REF_ACTION_FROM_CLASSES = True

# MVC 设置
USE_MVC                 = True
USE_CNN_STYLE_MVC       = True
FS                      = 1000
WIN_SEC                 = 1.0
OVERLAP                 = 0.5
NOTCH                   = 0
TOPK                    = 10
SPIKE_FACTOR            = 1.5
MVC_REF_ACTION          = "Stretch_arm"
MVC_PERCENT_SCALE       = 100.0
MVC_CLIP                = None
MVC_REUSE_HEALTHY_BANK  = False

# 通道侧别映射
RIGHT_IDXS = list(range(0, 6))
LEFT_IDXS  = list(range(6, 12))

# 受伤侧声明
INJURY_ARM_MAP: Dict[str, str] = {
   'S12': 'L',
}

SUBJECT_ID_PATTERN = r"(S\d+|P\d+)"

# 受伤阶段先验
REUSE_HEALTHY_SCALER = True
PRIOR_MU             = 1.0

# 绘图参数
DPI, FIG_W_PER_COL, FIG_H_PER_ROW = 600, 3.4, 2.6
FONTSIZE_TITLE, FONTSIZE_LABEL, FONTSIZE_TICK = 10, 9, 7
LW_BOX, LW_WHISK_CAP, LW_MEDIAN, LW_MEAN = 1.1, 1.0, 1.8, 1.2
LS_MEAN, FLIER_MS, WHIS, SHOW_FLIERS, ROTATE_X = "--", 2.0, 1.5, True, 0
MAX_COLS, CONNECT_MEDIAN, BREAK_EDGES = 3, True, [(5, 6)]
XLABEL, YLABEL = "Channel", "Contribution"
plt.rcParams["axes.unicode_minus"] = False

# 贡献度收集容器
HEALTHY_CONTRIB_LONG = []
INJURY_CONTRIB_LONG  = []

# ================= 新增：权重文件检查函数 =================
def check_weights_exist(out_dir: str, phase: str = "healthy") -> bool:
    """
    检查指定阶段的权重文件是否完整存在
    
    Args:
        out_dir: 输出目录
        phase: 阶段名称（用于日志输出）
    
    Returns:
        bool: 所有必需文件都存在返回True，否则False
    """
    if not os.path.exists(out_dir):
        print(f"[检查] {phase.upper()} 输出目录不存在: {out_dir}")
        return False
    
    # 检查每一折的必需文件
    missing_files = []
    for fold in range(1, NUM_FOLDS + 1):
        fold_dir = os.path.join(out_dir, f'fold{fold}')
        
        # 必需文件列表
        required_files = [
            'scaler.pkl',  # 标准化器
        ]
        
        # 每个通道的BLS模型
        for ch in range(NUM_CH):
            required_files.append(f'bls_ch{ch}.joblib')
        
        # 检查文件是否存在
        for fname in required_files:
            fpath = os.path.join(fold_dir, fname)
            if not os.path.exists(fpath):
                missing_files.append(fpath)
    
    if missing_files:
        print(f"[检查] {phase.upper()} 阶段缺失以下文件:")
        for f in missing_files[:5]:  # 只显示前5个
            print(f"  - {f}")
        if len(missing_files) > 5:
            print(f"  ... 还有 {len(missing_files) - 5} 个文件缺失")
        return False
    
    print(f"[检查] {phase.upper()} 阶段所有权重文件完整 ✓")
    return True


def load_healthy_mvc_bank(out_dir: str) -> Optional[dict]:
    """
    从健康阶段输出目录加载MVC字典
    
    Args:
        out_dir: 健康阶段输出目录
    
    Returns:
        dict: MVC字典，如果加载失败返回None
    """
    mvc_path = os.path.join(out_dir, "healthy_mvc.xlsx")
    
    if not os.path.exists(mvc_path):
        print(f"[加载] 未找到健康MVC文件: {mvc_path}")
        return None
    
    try:
        df = pd.read_excel(mvc_path)
        mvc_bank = {}
        
        # 从Excel重建MVC字典
        for subj in df['Subject'].unique():
            subj_data = df[df['Subject'] == subj].sort_values('Channel')
            mvc_vec = subj_data['MVC'].to_numpy()
            mvc_bank[subj] = mvc_vec
        
        print(f"[加载] 成功加载健康MVC字典: {len(mvc_bank)} 个被试")
        return mvc_bank
    
    except Exception as e:
        print(f"[加载] 加载健康MVC字典失败: {e}")
        return None

# ================= 工具函数（保持原有代码）=================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _safe_list_dirs(root):
    return [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

def _infer_actions_from_fs(data_dir, expect_C: int):
    actions = []
    subs = sorted([d for d in _safe_list_dirs(data_dir) if d.startswith('S') or d.startswith('P')])
    if len(subs) > 0:
        actions = sorted(_safe_list_dirs(os.path.join(data_dir, subs[0])))
    if len(actions) != expect_C:
        actions = [f"Class{i}" for i in range(expect_C)]
    return actions

def _ensure_exists(path, what="file"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {what}: {path}")

def _load_state_flex(model: torch.nn.Module, ckpt_path: str):
    sd = torch.load(ckpt_path, map_location='cpu')
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    new_sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:    print(f"[Warn] Missing keys: {missing[:8]}{'...' if len(missing)>8 else ''}")
    if unexpected: print(f"[Warn] Unexpected keys: {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")
    model.to(DEVICE)

def _action_from_path(fp: str) -> str:
    return os.path.basename(os.path.dirname(fp))

def _remap_split_drop_refaction(split: dict, ref_action: str):
    out = {}
    acts = set()
    for k in ('train','val','test'):
        items = split[k]
        kept: List[Tuple[str, str]] = []
        for fp, _y in items:
            act = _action_from_path(fp)
            if act == ref_action:
                continue
            kept.append((fp, act))
            acts.add(act)
        out[k] = kept
    actions = sorted(acts)
    act2id = {a:i for i,a in enumerate(actions)}
    for k in out:
        out[k] = [(fp, act2id[act]) for fp, act in out[k]]
    return out, actions


# ========= 原始TXT读取（仅用于 MVC 构建）=========
def _robust_read_df_for_mvc(path: str, num_channels: int, skip_header: int) -> pd.DataFrame:
    """
    读 Stretch_arm 原始 txt/csv：优先 numpy.loadtxt；兼容 'SYN1/SYN2' 尾列；返回 (T, num_channels) 的 DataFrame
    """
    # 首选：空白分隔的快速路径
    try:
        arr = np.loadtxt(path, dtype=float, ndmin=2, skiprows=max(3, skip_header))
    except Exception:
        # 退回 pandas 尝试
        arr = None
        for sepr in (None, r"[,\t;| ]+", r"\s+"):
            try:
                df = pd.read_csv(path, sep=sepr, engine="python", header=None,
                                 skiprows=max(3, skip_header), on_bad_lines="skip")
                df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
                if df.shape[1] > 0:
                    arr = df.values
                    break
            except Exception:
                continue
        if arr is None:
            raise

    A = np.asarray(arr, dtype=np.float64)
    # 若尾部两列为计数器（SYN1/SYN2），裁掉
    if A.shape[1] >= num_channels + 2:
        tail = A[:, -2:]
        if np.all(np.isfinite(tail)):
            frac = np.abs(tail - np.round(tail))
            is_int_like = (np.nanmax(frac) < 1e-6)
            mono = np.all(np.diff(tail, axis=0) >= -1e-12)
            if is_int_like and mono:
                A = A[:, :-2]

    if A.shape[1] < num_channels:
        raise ValueError(f"列数不足({A.shape[1]}<{num_channels})：{os.path.basename(path)}")

    return pd.DataFrame(A[:, :num_channels])

# ---------- 仅用于“预估段数”的快速读取 ----------
def _robust_loadtxt(path: str, num_channels: int, skip_header: int) -> np.ndarray:
    """
    仅为“预估段数”而快速读取前 num_channels 列，返回 (T, C) 的 float64。
    兼容空白/逗号/制表符/分号/竖线等分隔，自动丢弃全 NaN 列和含 NaN 的行。
    读失败抛异常（由调用方兜底占位）。
    """
    # 1) 空白分隔
    try:
        arr = np.loadtxt(path, skiprows=skip_header, usecols=range(num_channels), dtype=np.float64)
        if arr.ndim == 1: arr = arr.reshape(1, -1)
        if arr.shape[1] < num_channels:
            raise ValueError(f"{os.path.basename(path)} 列数不足: {arr.shape[1]} < {num_channels}")
        return arr[:, :num_channels]
    except Exception:
        pass
    # 2) 逗号
    try:
        arr = np.loadtxt(path, delimiter=",", skiprows=skip_header, usecols=range(num_channels), dtype=np.float64)
        if arr.ndim == 1: arr = arr.reshape(1, -1)
        if arr.shape[1] < num_channels:
            raise ValueError(f"{os.path.basename(path)} 列数不足: {arr.shape[1]} < {num_channels}")
        return arr[:, :num_channels]
    except Exception:
        pass
    # 3) pandas 兜底
    for sepr in (r"[,\t;|]+", r"[,\t;| ]+", r"\s+"):
        try:
            df = pd.read_csv(path, sep=sepr, engine="python", header=None,
                             skiprows=skip_header, on_bad_lines="skip")
            df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna(axis=0, how="any")
            if df.shape[1] == 0: continue
            if df.shape[1] < num_channels:
                raise ValueError(f"{os.path.basename(path)} 列数不足: {df.shape[1]} < {num_channels}")
            vals = df.iloc[:, :num_channels].to_numpy(dtype=np.float64, copy=False)
            if vals.ndim == 1: vals = vals.reshape(1, -1)
            return vals
        except Exception:
            continue
    raise RuntimeError(f"_robust_loadtxt 解析失败: {os.path.basename(path)}")

# ---------- 切片段数估计（与 EMGSlidingDataset 规则保持一致） ----------
def _slice_indices_len(T: int, seg_len: int, stride: int) -> int:
    """返回按 (seg_len, stride) 切片后产生的段数（与 EMGSlidingDataset 的策略一致）
       规则：若 T <= seg_len => 1 段；否则按 stride 滑动，末段若剩余 ≥ seg_len*0.5 追加一段（左对齐到最后）"""
    if T <= seg_len: return 1
    cnt, s = 0, 0
    while s + seg_len <= T:
        cnt += 1; s += stride
    if T - (s) >= int(round(seg_len * 0.5)): cnt += 1
    return cnt

# ========= CNN同款：滤波+abs+MAV(topK剔峰均值) =========
def _design_filters(fs: int, notch_freq: Optional[float]):
    b_bp, a_bp = butter(4, [20/(fs/2), 450/(fs/2)], btype="band")
    if notch_freq and notch_freq > 0:
        b_n, a_n = iirnotch(w0=notch_freq/(fs/2), Q=30.0)
        return (b_bp, a_bp), (b_n, a_n)
    return (b_bp, a_bp), None

def _preprocess_abs(x: np.ndarray, bp, notch) -> np.ndarray:
    """带通(+陷波)+全波整流，输入/输出均为 (T,C)"""
    b_bp, a_bp = bp
    padlen_bp = min(3*max(len(b_bp), len(a_bp)), max(x.shape[0]-1, 1))
    y = filtfilt(b_bp, a_bp, x, axis=0, padlen=padlen_bp)
    if notch is not None:
        b_n, a_n = notch
        padlen_n = min(3*max(len(b_n), len(a_n)), max(y.shape[0]-1, 1))
        y = filtfilt(b_n, a_n, y, axis=0, padlen=padlen_n)
    return np.abs(y)

def _sliding_mav(abs_sig: np.ndarray, win_len: int, hop: int) -> np.ndarray:
    """对整流后的 (T,C) 计算滑窗 MAV -> (N, C)"""
    T, C = abs_sig.shape
    if T < win_len:
        return abs_sig.mean(axis=0, keepdims=True)
    s_idx = np.arange(0, T - win_len + 1, hop, dtype=int)
    out = np.empty((len(s_idx), C), dtype=float)
    for i, s in enumerate(s_idx):
        out[i] = abs_sig[s:s+win_len, :].mean(axis=0)
    return out

def _baseline_topk_spike(mavs: np.ndarray, k: int = TOPK, spike_factor: float = SPIKE_FACTOR, eps: float = 1e-12) -> np.ndarray:
    """每通道取TopK；若max1>spike_factor*max2剔除max1；其余求均值。返回 (C,)"""
    N, C = mavs.shape
    out = np.zeros(C, dtype=float)
    for c in range(C):
        w = mavs[:, c]
        w = w[np.isfinite(w)]
        if w.size == 0:
            out[c] = eps; continue
        k_eff = min(k, w.size)
        topk = np.partition(w, -k_eff)[-k_eff:]
        topk.sort()
        if topk.size >= 2 and topk[-1] > spike_factor * max(topk[-2], eps):
            topk = topk[:-1]
        out[c] = max(float(np.mean(topk)), eps)
    return out

# =========（兼容旧签名）MVC 构建 =========
def compute_mvc_for_subject(subj_dir: str, num_channels: int,
                            ref_action: str, skip_header: int,
                            win_samples: int, stride_samples: int,
                            stat: str) -> np.ndarray:
    """
    兼容旧接口：内部按 CNN 同款 MAV+TopK 方案计算 per-channel MVC。
    """
    action_dir = os.path.join(subj_dir, ref_action)
    files = []
    if os.path.isdir(action_dir):
        for ext in ("*.txt", "*.csv", "*.dat", "*.tsv"):
            files += glob.glob(os.path.join(action_dir, ext))
    if not files:
        warnings.warn(f"[{os.path.basename(subj_dir)}] 未找到 {ref_action} 数据，MVC 回退为全1。")
        return np.ones((num_channels,), dtype=np.float64)

    (bp), notch_ba = _design_filters(FS, NOTCH if NOTCH and NOTCH > 0 else None)
    win_len = max(1, int(round(WIN_SEC * FS)))
    hop     = max(1, int(round(win_len * (1 - OVERLAP))))

    mavs_all = []
    for p in sorted(files):
        try:
            df = _robust_read_df_for_mvc(p, num_channels=num_channels, skip_header=skip_header)
            x  = df.iloc[:, :num_channels].to_numpy(dtype=float)
            abs_emg = _preprocess_abs(x, (bp), notch_ba)   # (T,C)
            mavs_all.append(_sliding_mav(abs_emg, win_len, hop))
        except Exception as e:
            warnings.warn(f"[{os.path.basename(subj_dir)}] Stretch_arm 文件失败 {os.path.basename(p)}: {e}")
            continue
    if not mavs_all:
        warnings.warn(f"[{os.path.basename(subj_dir)}] {ref_action} 无有效窗，MVC 回退为全1。")
        return np.ones((num_channels,), dtype=np.float64)

    mavs_all = np.vstack(mavs_all)  # (N, C)
    mvc = _baseline_topk_spike(mavs_all, k=TOPK, spike_factor=SPIKE_FACTOR, eps=1e-12)  # (C,)
    mvc = np.where(np.isfinite(mvc) & (mvc > 0), mvc, 1.0)
    return mvc.astype(np.float64)

def build_mvc_bank(data_root: str, num_channels: int,
                   ref_action: str, skip_header: int,
                   win_samples: int, stride_samples: int, stat: str) -> dict:
    """
    保留原签名，但内部走 CNN 同款 MVC 计算。
    """
    bank = {}
    subs = sorted([d for d in _safe_list_dirs(data_root) if d.startswith("S") or d.startswith("P")])
    for s in subs:
        subj_dir = os.path.join(data_root, s)
        mvc = compute_mvc_for_subject(
            subj_dir, num_channels, ref_action, skip_header,
            win_samples, stride_samples, stat
        )
        bank[s] = mvc
    print(f"[MVC] 构建完成：{len(bank)} 个被试。规则=CNN同款(MAV+TopK剔峰)")
    return bank

def _extract_subject_from_path(path: str) -> str:
    p = path.replace("\\", "/")
    m = re.search(SUBJECT_ID_PATTERN, p)
    if m: return m.group(1)
    parts = p.split("/")
    for i in range(len(parts)-1, -1, -1):
        if parts[i].startswith("S") or parts[i].startswith("P"):
            return parts[i]
    return "UNKNOWN"

def mirror_mvc_by_injury_side(mvc_vec: np.ndarray, side: str,
                              right_idxs=RIGHT_IDXS, left_idxs=LEFT_IDXS) -> np.ndarray:
    mvc_vec = np.asarray(mvc_vec, dtype=np.float64).copy()
    if side == 'R':
        for r_i, l_i in zip(right_idxs, left_idxs):
            mvc_vec[r_i] = mvc_vec[l_i]
    elif side == 'L':
        for r_i, l_i in zip(right_idxs, left_idxs):
            mvc_vec[l_i] = mvc_vec[r_i]
    return mvc_vec

# ========= %MVC 数据集包装器 =========
class EMGSlidingDatasetMVC(Dataset):
    """
    在构造时预展开“段级别”的被试索引，保证与 base 的 __len__ 对齐，
    避免 __getitem__ 用 self.items[idx] 造成越界。
    """
    def __init__(self, items, num_ch, seg_len, stride, skip_header,
                 mvc_bank: dict, percent_scale: float = 100.0, clip=None):
        super().__init__()
        self.num_ch = num_ch
        self.seg_len = seg_len
        self.stride = stride
        self.skip = skip_header
        self.scale = float(percent_scale)
        self.clip = clip

        # 底层原始数据集（负责真正的读文件&切片）
        self.base = EMGSlidingDataset(items, num_ch, seg_len, stride, skip_header)

        # 1) 预先把每个文件的“段数”算出来，展开出与 base.__len__ 等长的 subject 列表
        subj_per_seg = []
        for fp, _y in items:
            try:
                subj = _extract_subject_from_path(fp)
                arr = _robust_loadtxt(fp, num_channels=num_ch, skip_header=skip_header)  # (T,C)
                T = arr.shape[0]
                nseg = _slice_indices_len(T, seg_len, stride)
                subj_per_seg.extend([subj] * nseg)
            except Exception as e:
                # 读失败：至少占位 1 段
                warnings.warn(f"[MVC] 预展开失败 {os.path.basename(fp)}，占位1段。({e})")
                subj_per_seg.append(_extract_subject_from_path(fp))

        # 2) 和 base 实际长度对齐（极少数情况下，base 的切片策略或过滤导致长度不同）
        L_base = len(self.base)
        if len(subj_per_seg) < L_base:
            fill = subj_per_seg[-1] if subj_per_seg else "UNKNOWN"
            subj_per_seg.extend([fill] * (L_base - len(subj_per_seg)))
        elif len(subj_per_seg) > L_base:
            subj_per_seg = subj_per_seg[:L_base]

        self.subj_per_seg = subj_per_seg
        self.mvc_bank = mvc_bank

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]           # x: [C, T]
        subj = self.subj_per_seg[idx] if idx < len(self.subj_per_seg) else "UNKNOWN"
        mvc = self.mvc_bank.get(subj, None)

        if mvc is None:
            warnings.warn(f"[MVC] {subj} 未在 MVC 字典中，回退为全 1。")
            denom = torch.ones((self.num_ch,), dtype=x.dtype, device=x.device)
        else:
            denom = torch.tensor(mvc, dtype=x.dtype, device=x.device)
            denom = torch.clamp(denom, min=1e-9)

        x = (x / denom.view(-1, 1)) * self.scale
        if self.clip is not None:
            lo, hi = self.clip
            x = torch.clamp(x, lo, hi)

        return x, y

# ========= Loader（自动切换 MVC/非 MVC）=========
def _build_loader(items, shuffle: bool, data_root: str, mvc_bank_cache: dict):
    if USE_MVC:
        if data_root not in mvc_bank_cache:
            mvc_bank_cache[data_root] = build_mvc_bank(
                data_root, NUM_CH, MVC_REF_ACTION, SKIP,
                SEG_LEN, STRIDE, "mean"
            )
        ds = EMGSlidingDatasetMVC(items, NUM_CH, SEG_LEN, STRIDE, SKIP,
                                  mvc_bank=mvc_bank_cache[data_root],
                                  percent_scale=MVC_PERCENT_SCALE,
                                  clip=MVC_CLIP)
    else:
        ds = EMGSlidingDataset(items, NUM_CH, SEG_LEN, STRIDE)
    return DataLoader(ds, batch_size=BS, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

# ========= 导出：MVC 字典 & 健康-受伤对照 =========
def export_mvc_bank(mvc_bank: dict, out_path: str):
    rows = []
    for subj, vec in mvc_bank.items():
        for ch, v in enumerate(vec):
            rows.append({"Subject": subj, "Channel": f"Ch{ch+1}", "MVC": float(v)})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_excel(out_path, index=False)
    print("[保存]", out_path)

def export_mvc_comparison(healthy_bank: Optional[dict],
                          injury_bank_before: Optional[dict],
                          injury_bank_after: Optional[dict],
                          out_path: str):
    """
    合并三套 MVC：健康 / 受伤前镜像 / 受伤后镜像；主键=Subject×Channel
    """
    subjects = set()
    if healthy_bank: subjects |= set(healthy_bank.keys())
    if injury_bank_before: subjects |= set(injury_bank_before.keys())
    if injury_bank_after: subjects |= set(injury_bank_after.keys())
    rows = []
    for subj in sorted(subjects):
        for ch in range(NUM_CH):
            vh = float(healthy_bank.get(subj, np.array([np.nan]*NUM_CH))[ch]) if healthy_bank and subj in healthy_bank and len(healthy_bank[subj])>ch else np.nan
            vi_b = float(injury_bank_before.get(subj, np.array([np.nan]*NUM_CH))[ch]) if injury_bank_before and subj in injury_bank_before and len(injury_bank_before[subj])>ch else np.nan
            vi_a = float(injury_bank_after.get(subj, np.array([np.nan]*NUM_CH))[ch]) if injury_bank_after and subj in injury_bank_after and len(injury_bank_after[subj])>ch else np.nan
            ratio = (vi_a / vh) if (isinstance(vh, float) and np.isfinite(vh) and vh>0 and np.isfinite(vi_a)) else np.nan
            rows.append({
                "Subject": subj,
                "Channel": f"Ch{ch+1}",
                "MVC_healthy": vh,
                "MVC_injury_before": vi_b,
                "MVC_injury_after": vi_a,
                "after_to_healthy_ratio": ratio
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_excel(out_path, index=False)
    print("[保存]", out_path)

# ========= 受伤先验相关 =========
import torch.nn.functional as F
def _one_hot(y_long: torch.Tensor, C: int) -> torch.Tensor:
    return F.one_hot(y_long, num_classes=C).float()

def _align_beta_cols(beta0: torch.Tensor, C: int) -> torch.Tensor:
    D, C0 = beta0.shape
    if C0 == C: return beta0
    beta_new = torch.zeros((D, C), device=beta0.device, dtype=beta0.dtype)
    beta_new[:, :min(C0, C)] = beta0[:, :min(C0, C)]
    return beta_new

@torch.no_grad()
def _proximal_beta_update(H: torch.Tensor, Y_onehot: torch.Tensor,
                          beta0: torch.Tensor, lam: float, mu: float) -> torch.Tensor:
    H = H.to(dtype=torch.float32)
    Y_onehot = Y_onehot.to(dtype=torch.float32)
    beta0 = beta0.to(dtype=torch.float32)
    D = H.shape[1]
    I = torch.eye(D, device=H.device, dtype=H.dtype)
    A = H.T @ H + (lam + mu) * I
    B = H.T @ Y_onehot + mu * beta0
    if hasattr(torch.linalg, "solve"):
        try: return torch.linalg.solve(A, B)
        except Exception: pass
    for eps in (0.0, 1e-8, 1e-7, 1e-6, 1e-5):
        try:
            A_eps = A + eps * I
            L = torch.linalg.cholesky(A_eps) if hasattr(torch.linalg, "cholesky") else torch.cholesky(A_eps)
            if hasattr(torch, "cholesky_solve"): return torch.cholesky_solve(B, L)
            else: return torch.linalg.cholesky_solve(B, L)
        except Exception: continue
    try:
        X, _ = torch.solve(B, A); return X
    except Exception:
        return torch.pinverse(A) @ B

# ========= 箱线图 =========
def _apply_axes_style(ax, y_label: str, show_ylabel: bool):
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(axis="both", which="both", direction="in", labelsize=FONTSIZE_TICK)
    ax.set_xlabel(XLABEL, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(y_label if show_ylabel else "", fontsize=FONTSIZE_LABEL)

def _break_segments(xs, ys, tick_labels, breaks):
    nums = [int(re.search(r"(\d+)", str(l)).group(1)) if re.search(r"(\d+)", str(l)) else None
            for l in tick_labels]
    break_idx = set()
    for a, b in breaks:
        if a in nums and b in nums:
            ia, ib = nums.index(a), nums.index(b)
            if ib == ia + 1: break_idx.add(ia)
    segs, start = [], 0
    for i in range(len(xs) - 1):
        if i in break_idx: segs.append((xs[start:i+1], ys[start:i+1])); start = i + 1
    segs.append((xs[start:], ys[start:])); return segs

def _boxplot_with_labels(ax, data, labels_list, **kwargs):
    try: return ax.boxplot(data, tick_labels=labels_list, **kwargs)
    except TypeError: return ax.boxplot(data, labels=labels_list, **kwargs)

def export_contrib_boxplots(all_contribs, actions, out_dir, *, tag: str, y_label: str, as_percent: bool):
    if len(all_contribs) == 0: return
    arr = np.stack(all_contribs, axis=0)     # (F, C, CH)
    F, C, CH = arr.shape
    channels = [f"Ch{i+1}" for i in range(CH)]
    rows = []
    for f in range(F):
        for ci, cls in enumerate(actions):
            row = {"Class": str(cls)}
            for j in range(CH):
                row[f"Ch{j+1}"] = float(arr[f, ci, j])
            rows.append(row)
    df_wide = pd.DataFrame(rows)
    excel_path = os.path.join(out_dir, f"boxplot_input_across_folds_{tag}.xlsx")
    df_wide.to_excel(excel_path, index=False); print("[保存]", excel_path)

    long_df = df_wide.melt(id_vars=["Class"], var_name="Channel", value_name="Value")
    classes = list(pd.unique(long_df["Class"]))
    n = len(classes); ncols = min(MAX_COLS, n); nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_W_PER_COL * ncols, FIG_H_PER_ROW * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for i, cls in enumerate(classes):
        ax = axes[i]
        sub = long_df[long_df["Class"] == cls]
        series = [sub.loc[sub["Channel"] == ch, "Value"].values for ch in channels]
        _boxplot_with_labels(
            ax, series, channels,
            showmeans=True, meanline=True, notch=False,
            whis=WHIS, showfliers=SHOW_FLIERS,
            boxprops=dict(linewidth=LW_BOX),
            whiskerprops=dict(linewidth=LW_WHISK_CAP),
            capprops=dict(linewidth=LW_WHISK_CAP),
            medianprops=dict(linewidth=LW_MEDIAN),
            meanprops=dict(linewidth=LW_MEAN, linestyle=LS_MEAN),
            flierprops=dict(marker="o", markersize=FLIER_MS,
                            markerfacecolor="none", markeredgewidth=0.8),
        )
        if CONNECT_MEDIAN:
            med = sub.groupby("Channel")["Value"].median()
            medians = [med.get(ch, np.nan) for ch in channels]
            xs = np.arange(1, len(channels) + 1)
            for sx, sy in _break_segments(xs, medians, channels, BREAK_EDGES):
                ax.plot(sx, sy, marker="o", linewidth=1, zorder=0.1)
        ax.set_title(str(cls), fontsize=FONTSIZE_TITLE, pad=6)
        ax.set_xticklabels(channels, rotation=ROTATE_X)
        _apply_axes_style(ax, y_label, show_ylabel=(i % ncols == 0))
        if as_percent: ax.set_ylim(0, 100)

    for j in range(i + 1, nrows * ncols): fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"box_all_classes_facet_{tag}.png"), dpi=DPI, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f"box_all_classes_facet_{tag}.pdf"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# =============== 健康阶段（训练并保存） ===============
def run_phase_healthy(mvc_bank_cache: dict):
    print("\n==== Phase A: HEALTHY training ====")
    os.makedirs(HEALTHY_OUT_DIR, exist_ok=True)

    splits = load_splits(HEALTHY_SPLITS) if (USE_SHARED_SPLIT and os.path.isfile(HEALTHY_SPLITS)) \
             else get_data_split(HEALTHY_DATA_DIR, NUM_CH, NUM_FOLDS)

    # 健康 MVC 字典并导出
    healthy_mvc_bank = None
    if USE_MVC:
        healthy_mvc_bank = build_mvc_bank(
            HEALTHY_DATA_DIR, NUM_CH, MVC_REF_ACTION, SKIP,
            SEG_LEN, STRIDE, "mean"
        )
        mvc_bank_cache[HEALTHY_DATA_DIR] = healthy_mvc_bank
        export_mvc_bank(healthy_mvc_bank, os.path.join(HEALTHY_OUT_DIR, "healthy_mvc.xlsx"))

    all_contribs_raw, all_contribs_norm = [], []
    bls_metrics_all, bls_classwise_all, all_metrics = [], [], []

    for fold in range(NUM_FOLDS):
        print(f"\n[HEALTHY] Fold {fold+1}/{NUM_FOLDS}")

        # === 关键：每折先拿原始split，再按开关过滤 & 重映射 ===
        split_raw = splits[fold]
        if EXCLUDE_REF_ACTION_FROM_CLASSES:
            split, actions = _remap_split_drop_refaction(split_raw, MVC_REF_ACTION)
        else:
            split = split_raw
            C_guess = max(max(l for _, l in sum(split.values(), [])) for _ in [0]) + 1
            actions = _infer_actions_from_fs(HEALTHY_DATA_DIR, C_guess)
        C = len(actions)

        items_tr = (split['train'] + split['val']) if MERGE_TRAIN_VAL_FOR_BLS else split['train']
        ld_tr = _build_loader(items_tr, True, HEALTHY_DATA_DIR, mvc_bank_cache)
        ld_te = _build_loader(split['test'], False, HEALTHY_DATA_DIR, mvc_bank_cache)

        ckpt = os.path.join(CNN_MODEL_DIR, f'fold{fold+1}', 'cnn_model_best.pth')
        _ensure_exists(ckpt, f"CNN checkpoint fold{fold+1}")
        cnn = EMGCNN(C, SEG_LEN, NUM_CH); _load_state_flex(cnn, ckpt); cnn.eval()

        fold_dir = os.path.join(HEALTHY_OUT_DIR, f'fold{fold+1}'); os.makedirs(fold_dir, exist_ok=True)

        X_tr, Y_tr = extract_features(cnn, ld_tr, DEVICE)
        X_te, Y_te = extract_features(cnn, ld_te, DEVICE)

        if USE_STANDARDIZE:
            scaler = StandardScaler().fit(X_tr)
            X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
            joblib.dump(scaler, os.path.join(fold_dir, 'scaler.pkl'))
        else:
            X_tr_s, X_te_s = X_tr, X_te

        feat_per_ch = X_tr_s.shape[1] // NUM_CH
        bls_list = []
        Xtr_t = torch.tensor(X_tr_s, dtype=torch.float32, device=DEVICE)
        Ytr_t = torch.tensor(Y_tr, dtype=torch.long, device=DEVICE)

        for c_idx in range(NUM_CH):
            Xc = Xtr_t[:, c_idx*feat_per_ch:(c_idx+1)*feat_per_ch]
            bls = BLSIncrementalFast1(feat_per_ch, BLS_NF, BLS_NH, C, reg=BLS_REG, device=DEVICE)
            bls.fit(Xc, Ytr_t)
            beta = bls.beta.detach().float().cpu().numpy()
            np.save(os.path.join(fold_dir, f'beta_ch{c_idx}.npy'), beta)
            bls_to_save = bls
            try:
                if hasattr(bls, 'to'): bls_to_save = bls.to('cpu')
            except Exception as e:
                print(f"[Warn] to('cpu') failed for bls_ch{c_idx}: {e}")
            # 提醒：磁盘空间不足时可加 compress=3
            joblib.dump(bls_to_save, os.path.join(fold_dir, f'bls_ch{c_idx}.joblib'))
            bls_list.append(bls)
            print(f"[HEALTHY|Fold{fold+1}|Ch{c_idx}] saved β & BLS.")

        def build_Z(Xs):
            Xt = torch.tensor(Xs, dtype=torch.float32, device=DEVICE); parts=[]
            for c_idx, bls in enumerate(bls_list):
                Xc = Xt[:, c_idx*feat_per_ch:(c_idx+1)*feat_per_ch]
                Hc = bls._map(Xc); parts.append(Hc @ bls.beta)
            return torch.cat(parts, dim=1)

        with torch.no_grad():
            Z_tr = build_Z(X_tr_s); Z_te = build_Z(X_te_s)

        fusion = torch.nn.Sequential(
            torch.nn.Linear(NUM_CH*C, FUSION_HIDDEN),
            torch.nn.ReLU(True),
            torch.nn.Dropout(FUSION_DROPOUT),
            torch.nn.Linear(FUSION_HIDDEN, C)
        ).to(DEVICE)
        opt = torch.optim.AdamW(fusion.parameters(), lr=LR)
        crit = torch.nn.CrossEntropyLoss()
        fusion.train()
        for _ in range(FUSION_EPOCHS):
            opt.zero_grad(set_to_none=True)
            loss = crit(fusion(Z_tr), Ytr_t)
            loss.backward(); opt.step()
        fusion.eval()
        with torch.no_grad():
            preds_te_fusion = fusion(Z_te).argmax(1).cpu().numpy()
        acc_fusion = accuracy_score(Y_te, preds_te_fusion)
        f1_fusion  = f1_score(Y_te, preds_te_fusion, average='macro')

        with torch.no_grad():
            logits_parts = []
            Xt = torch.tensor(X_te_s, dtype=torch.float32, device=DEVICE)
            for c_idx, bls in enumerate(bls_list):
                Xc = Xt[:, c_idx*feat_per_ch:(c_idx+1)*feat_per_ch]
                Hc = bls._map(Xc)
                logits_parts.append((Hc @ bls.beta).cpu().numpy())
            logits = np.sum(logits_parts, axis=0)
            preds_te_bls = logits.argmax(1)

        acc_bls  = accuracy_score(Y_te, preds_te_bls)
        prec_bls = precision_score(Y_te, preds_te_bls, average='macro', zero_division=0)
        rec_bls  = recall_score(Y_te, preds_te_bls,   average='macro', zero_division=0)
        f1_bls   = f1_score(Y_te,    preds_te_bls,    average='macro')

        print(f"[HEALTHY|Fold{fold+1}] BLS acc={acc_bls:.4f}  F1={f1_bls:.4f} | Fusion acc={acc_fusion:.4f}  F1={f1_fusion:.4f}")

        df_bls = pd.DataFrame([{"fold": fold+1, "acc": acc_bls, "precision": prec_bls, "recall": rec_bls, "f1": f1_bls}])
        df_bls.to_csv(os.path.join(fold_dir, "bls_metrics.csv"), index=False, encoding="utf-8-sig")
        bls_metrics_all.append(df_bls)

        prec_c, rec_c, f1_c, supp_c = precision_recall_fscore_support(
            Y_te, preds_te_bls, labels=list(range(C)), average=None, zero_division=0
        )
        df_bls_per_class = pd.DataFrame({
            "fold": fold+1,
            "class_id": np.arange(C, dtype=int),
            "action": [actions[i] for i in range(C)],
            "precision": prec_c.astype(float),
            "recall":    rec_c.astype(float),
            "f1":        f1_c.astype(float),
            "support":   supp_c.astype(int),
        })
        df_bls_per_class.to_csv(os.path.join(fold_dir, "bls_metrics_per_class.csv"),
                                index=False, encoding="utf-8-sig")
        bls_classwise_all.append(df_bls_per_class)

        # === 贡献度（RAW + RowNorm100） ===
        contrib = np.zeros((C, NUM_CH), dtype=float)
        for c_idx, bls in enumerate(bls_list):
            beta_c = bls.beta.detach().cpu().numpy()
            contrib[:, c_idx] = np.abs(beta_c).sum(axis=0)
        df_raw  = pd.DataFrame(contrib, index=actions, columns=[f"Ch{i+1}" for i in range(NUM_CH)])
        df_raw.to_excel(os.path.join(fold_dir, 'channel_to_class_contrib_raw.xlsx'))

        # —— 追加到“每折长表”：列= fold, action, channel, value（原始 |β| 通道和）
        for act_idx, act_name in enumerate(actions):
            for ch in range(NUM_CH):
                HEALTHY_CONTRIB_LONG.append({
                    "fold": fold+1,
                    "action": str(act_name),
                    "channel": f"Ch{ch+1}",
                    "value": float(contrib[act_idx, ch])
                })

        X = df_raw.to_numpy(dtype=float)
        row_sum = np.nansum(X, axis=1, keepdims=True)
        X_norm = np.zeros_like(X)
        ok = (row_sum > 0).squeeze()
        X_norm[ok, :] = (X[ok, :] / row_sum[ok]) * 100.0
        tails = ~np.isclose(X_norm.sum(axis=1), 100.0, rtol=0, atol=1e-10)
        for r in np.where(tails)[0]:
            diff = 100.0 - X_norm[r].sum()
            X_norm[r, -1] += diff
        df_norm = pd.DataFrame(X_norm, index=actions, columns=[f"Ch{i+1}" for i in range(NUM_CH)])
        df_norm.to_excel(os.path.join(fold_dir, 'channel_to_class_contrib_rowNorm100.xlsx'))
        verify = pd.DataFrame({"Action": df_norm.index,
                               "RowSum": df_norm.sum(axis=1).round(10),
                               "Equal_100": (df_norm.sum(axis=1).round(10) == 100.0)})
        verify.to_csv(os.path.join(fold_dir, 'verify_row_sum_100.csv'), index=False, encoding="utf-8-sig")

        all_contribs_raw.append(df_raw.to_numpy())
        all_contribs_norm.append(df_norm.to_numpy())
        all_metrics.append({'fold': fold+1, 'acc_fusion': acc_fusion, 'f1_fusion': f1_fusion,
                            'acc_bls': acc_bls, 'prec_bls': prec_bls, 'rec_bls': rec_bls, 'f1_bls': f1_bls})

    # 汇总（健康）
    df_m = pd.DataFrame(all_metrics)
    df_m.to_csv(os.path.join(HEALTHY_OUT_DIR, 'summary_metrics.csv'), index=False, encoding='utf-8-sig')
    if bls_metrics_all:
        df_bls_all = pd.concat(bls_metrics_all, ignore_index=True)
        mean_row = {"fold":"mean","acc":df_bls_all["acc"].mean(),"precision":df_bls_all["precision"].mean(),
                    "recall":df_bls_all["recall"].mean(),"f1":df_bls_all["f1"].mean()}
        df_bls_all = pd.concat([df_bls_all, pd.DataFrame([mean_row])], ignore_index=True)
        df_bls_all.to_csv(os.path.join(HEALTHY_OUT_DIR, "summary_bls_metrics.csv"),
                          index=False, encoding="utf-8-sig")
    if bls_classwise_all:
        df_bls_cls_all = pd.concat(bls_classwise_all, ignore_index=True)
        df_cls_mean = (df_bls_cls_all.groupby(["class_id","action"], as_index=False)
                       .agg(precision=("precision","mean"), recall=("recall","mean"),
                            f1=("f1","mean"), support=("support","sum")))
        df_cls_mean.to_csv(os.path.join(HEALTHY_OUT_DIR,"summary_bls_metrics_per_class_mean.csv"),
                           index=False, encoding="utf-8-sig")
        def _wavg(g, col):
            w=g["support"].to_numpy(float); x=g[col].to_numpy(float); s=w.sum()
            return float(np.average(x, weights=w)) if s>0 else float(np.nan)
        rows=[]
        for (cid,act),g in df_bls_cls_all.groupby(["class_id","action"]):
            rows.append({"class_id":cid,"action":act,
                         "precision":_wavg(g,"precision"),
                         "recall":_wavg(g,"recall"),
                         "f1":_wavg(g,"f1"),
                         "support":int(g["support"].sum())})
        pd.DataFrame(rows).to_csv(os.path.join(HEALTHY_OUT_DIR,"summary_bls_metrics_per_class_weighted.csv"),
                                  index=False, encoding="utf-8-sig")
    if len(all_contribs_raw):
        mean_raw = np.mean(np.stack(all_contribs_raw, axis=0), axis=0)
        pd.DataFrame(mean_raw, index=actions, columns=[f"Ch{i+1}" for i in range(NUM_CH)])\
            .to_excel(os.path.join(HEALTHY_OUT_DIR,'channel_to_class_contrib_mean_RAW.xlsx'))
        export_contrib_boxplots(all_contribs_raw, actions,
                                HEALTHY_OUT_DIR, tag="raw_healthy",
                                y_label="Contribution (a.u.)", as_percent=False)
    if len(all_contribs_norm):
        mean_norm = np.mean(np.stack(all_contribs_norm, axis=0), axis=0)
        pd.DataFrame(mean_norm, index=actions, columns=[f"Ch{i+1}" for i in range(NUM_CH)])\
            .to_excel(os.path.join(HEALTHY_OUT_DIR,'channel_to_class_contrib_mean_rowNorm100.xlsx'))
        export_contrib_boxplots(all_contribs_norm, actions,
                                HEALTHY_OUT_DIR, tag="rowNorm100_healthy",
                                y_label="Contribution (%)", as_percent=True)

    # ===== 导出：健康阶段 |β| 绝对值通道求和（每折合在一起 + 跨折求和） =====
    if HEALTHY_CONTRIB_LONG:
        df_long = pd.DataFrame(HEALTHY_CONTRIB_LONG)
        out_long_csv = os.path.join(HEALTHY_OUT_DIR, "contrib_abs_by_fold_healthy_long.csv")
        df_long.to_csv(out_long_csv, index=False, encoding="utf-8-sig")
        print("[保存]", out_long_csv)

        df_sum = (df_long
                  .groupby(["action", "channel"], as_index=False)["value"]
                  .sum())
        out_sum_csv = os.path.join(HEALTHY_OUT_DIR, "contrib_abs_sum_over_folds_healthy_long.csv")
        df_sum.to_csv(out_sum_csv, index=False, encoding="utf-8-sig")
        print("[保存]", out_sum_csv)

        df_sum_wide = df_sum.pivot(index="action", columns="channel", values="value").fillna(0.0)
        out_sum_xlsx = os.path.join(HEALTHY_OUT_DIR, "contrib_abs_sum_over_folds_healthy.xlsx")
        df_sum_wide.to_excel(out_sum_xlsx)
        print("[保存]", out_sum_xlsx)

    print("\n[HEALTHY] Done.")
    return healthy_mvc_bank

# =============== 受伤阶段（复用健康映射并更新 β） ===============
def run_phase_injury(mvc_bank_cache: dict, healthy_mvc_bank: Optional[dict]):
    print("\n==== Phase B: INJURY adaptation (healthy prior) ====")
    os.makedirs(INJURY_OUT_DIR, exist_ok=True)

    # 检查健康阶段的权值文件是否存在
    if not check_weights_exist(HEALTHY_OUT_DIR, phase="healthy"):
        print("[INJURY] 健康阶段权值不存在，跳过受伤阶段。")
        return  # 跳过受伤阶段

    splits = load_splits(INJURY_SPLITS) if (USE_SHARED_SPLIT and os.path.isfile(INJURY_SPLITS)) \
             else get_data_split(INJURY_DATA_DIR, NUM_CH, NUM_FOLDS)

    # 在受伤目录内独立计算 MVC（或复用健康），并保存“镜像前/后”
    injury_mvc_before = None
    injury_mvc_after  = None
    if USE_MVC:
        if MVC_REUSE_HEALTHY_BANK and healthy_mvc_bank is not None:
            injury_mvc_before = dict(healthy_mvc_bank)
            print("[MVC|Injury] 复用健康 MVC 字典作为受伤镜像前值。")
        else:
            injury_mvc_before = build_mvc_bank(
                INJURY_DATA_DIR, NUM_CH, MVC_REF_ACTION, SKIP,
                SEG_LEN, STRIDE, "mean"
            )
            print("[MVC|Injury] 已在受伤目录内独立计算 MVC（镜像前）。")

        # 镜像后
        injury_mvc_after = {subj: vec.copy() for subj, vec in injury_mvc_before.items()}
        for subj, side in INJURY_ARM_MAP.items():
            if subj in injury_mvc_after and side in ('R','L'):
                injury_mvc_after[subj] = mirror_mvc_by_injury_side(injury_mvc_after[subj], side,
                                                                   right_idxs=RIGHT_IDXS, left_idxs=LEFT_IDXS)
                print(f"[MVC|Injury] {subj} side={side} → 已应用镜像。")

        # 导出三张表 + 对照表
        export_mvc_bank(injury_mvc_before, os.path.join(INJURY_OUT_DIR, "injury_mvc_before_mirror.xlsx"))
        export_mvc_bank(injury_mvc_after,  os.path.join(INJURY_OUT_DIR, "injury_mvc_after_mirror.xlsx"))
        export_mvc_comparison(
            healthy_bank=healthy_mvc_bank,
            injury_bank_before=injury_mvc_before,
            injury_bank_after=injury_mvc_after,
            out_path=os.path.join(INJURY_OUT_DIR, "mvc_compare_healthy_vs_injury.xlsx")
        )
        mvc_bank_cache[INJURY_DATA_DIR] = injury_mvc_after

    all_contribs_raw, all_contribs_norm = [], []
    bls_metrics_all, bls_classwise_all, all_metrics = [], [], []

    def _infer_injury_side(items) -> Optional[str]:
        """推断当前折训练集对应的受伤侧别（若唯一）。"""
        sides = set()
        for fp, _ in items:
            subj = _extract_subject_from_path(fp)
            side = INJURY_ARM_MAP.get(subj)
            if side in ("L", "R"):
                sides.add(side)
        if len(sides) == 1:
            return next(iter(sides))
        return None

    for fold in range(NUM_FOLDS):
        print(f"\n[INJURY] Fold {fold+1}/{NUM_FOLDS}")

        # === 关键：每折先拿原始split，再按开关过滤 & 重映射 ===
        split_raw = splits[fold]
        if EXCLUDE_REF_ACTION_FROM_CLASSES:
            split, actions_inj = _remap_split_drop_refaction(split_raw, MVC_REF_ACTION)
        else:
            split = split_raw
            C_guess = max(max(l for _, l in sum(split.values(), [])) for _ in [0]) + 1
            actions_inj = _infer_actions_from_fs(INJURY_DATA_DIR, C_guess)
        C_inj = len(actions_inj)

        items_tr = (split['train'] + split['val']) if MERGE_TRAIN_VAL_FOR_BLS else split['train']
        injury_side = _infer_injury_side(items_tr)
        ld_tr = _build_loader(items_tr, True, INJURY_DATA_DIR, mvc_bank_cache)
        ld_te = _build_loader(split['test'], False, INJURY_DATA_DIR, mvc_bank_cache)

        ckpt = os.path.join(CNN_MODEL_DIR, f'fold{fold+1}', 'cnn_model_best.pth')
        _ensure_exists(ckpt, f"CNN checkpoint fold{fold+1}")
        cnn = EMGCNN(C_inj, SEG_LEN, NUM_CH); _load_state_flex(cnn, ckpt); cnn.eval()

        fold_dir_inj = os.path.join(INJURY_OUT_DIR, f'fold{fold+1}')
        os.makedirs(fold_dir_inj, exist_ok=True)
        fold_dir_hlt = os.path.join(HEALTHY_OUT_DIR, f'fold{fold+1}')
        _ensure_exists(fold_dir_hlt, f"healthy fold dir fold{fold+1}")

        X_tr, Y_tr = extract_features(cnn, ld_tr, DEVICE)
        X_te, Y_te = extract_features(cnn, ld_te, DEVICE)

        if REUSE_HEALTHY_SCALER:
            sc_path = os.path.join(fold_dir_hlt, 'scaler.pkl')
            _ensure_exists(sc_path, "healthy scaler")
            scaler = joblib.load(sc_path)
            print(f"[INJURY|Fold{fold+1}] Reused healthy scaler.")
        else:
            scaler = StandardScaler().fit(X_tr)
            print(f"[INJURY|Fold{fold+1}] Fitted new scaler on injury train(+val).")

        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
        joblib.dump(scaler, os.path.join(fold_dir_inj, 'scaler.pkl'))

        feat_per_ch = X_tr_s.shape[1] // NUM_CH
        Xtr_t = torch.tensor(X_tr_s, dtype=torch.float32, device=DEVICE)
        Ytr_t = torch.tensor(Y_tr, dtype=torch.long, device=DEVICE)
        Ytr_onehot = _one_hot(Ytr_t, C_inj)

        bls_list = []
        for c_idx in range(NUM_CH):
            healthy_bls_path = os.path.join(fold_dir_hlt, f'bls_ch{c_idx}.joblib')
            _ensure_exists(healthy_bls_path, f"healthy BLS ch{c_idx}")
            bls = joblib.load(healthy_bls_path)
            if hasattr(bls, 'to'):
                try: bls.to(DEVICE)
                except: pass

            Xc = Xtr_t[:, c_idx*feat_per_ch:(c_idx+1)*feat_per_ch]
            Hc = bls._map(Xc)
            beta0 = bls.beta if isinstance(bls.beta, torch.Tensor) else torch.tensor(bls.beta, dtype=torch.float32, device=DEVICE)
            beta0 = _align_beta_cols(beta0.to(DEVICE), C_inj)

            freeze_channel = False
            if injury_side == 'L' and c_idx in RIGHT_IDXS:
                freeze_channel = True
            elif injury_side == 'R' and c_idx in LEFT_IDXS:
                freeze_channel = True

            if freeze_channel:
                bls.beta = beta0
                bls_list.append(bls)
                print(f"[INJURY|Fold{fold+1}|Ch{c_idx}] Healthy-side channel detected, β kept frozen.")
                continue

            beta_upd = _proximal_beta_update(Hc, Ytr_onehot, beta0, lam=BLS_REG, mu=PRIOR_MU)
            bls.beta = beta_upd
            bls_list.append(bls)

            print(f"[INJURY|Fold{fold+1}|Ch{c_idx}] β updated in-memory (μ={PRIOR_MU}).")

        def build_Z(Xs):
            Xt = torch.tensor(Xs, dtype=torch.float32, device=DEVICE); parts=[]
            for c_idx, bls in enumerate(bls_list):
                Xc = Xt[:, c_idx*feat_per_ch:(c_idx+1)*feat_per_ch]
                Hc = bls._map(Xc); parts.append(Hc @ bls.beta)
            return torch.cat(parts, dim=1)

        with torch.no_grad():
            Z_tr = build_Z(X_tr_s); Z_te = build_Z(X_te_s)

        fusion = torch.nn.Sequential(
            torch.nn.Linear(NUM_CH*C_inj, FUSION_HIDDEN),
            torch.nn.ReLU(True),
            torch.nn.Dropout(FUSION_DROPOUT),
            torch.nn.Linear(FUSION_HIDDEN, C_inj)
        ).to(DEVICE)
        opt = torch.optim.AdamW(fusion.parameters(), lr=LR)
        crit = torch.nn.CrossEntropyLoss()
        fusion.train()
        for _ in range(FUSION_EPOCHS):
            opt.zero_grad(set_to_none=True)
            loss = crit(fusion(Z_tr), Ytr_t)
            loss.backward(); opt.step()

        fusion.eval()
        with torch.no_grad():
            preds_te_fusion = fusion(Z_te).argmax(1).cpu().numpy()
        acc_fusion = accuracy_score(Y_te, preds_te_fusion)
        f1_fusion  = f1_score(Y_te, preds_te_fusion, average='macro')

        with torch.no_grad():
            logits_parts = []
            Xt = torch.tensor(X_te_s, dtype=torch.float32, device=DEVICE)
            for c_idx, bls in enumerate(bls_list):
                Xc = Xt[:, c_idx*feat_per_ch:(c_idx+1)*feat_per_ch]
                Hc = bls._map(Xc)
                logits_parts.append((Hc @ bls.beta).cpu().numpy())
            logits = np.sum(logits_parts, axis=0)
            preds_te_bls = logits.argmax(1)

        acc_bls  = accuracy_score(Y_te, preds_te_bls)
        prec_bls = precision_score(Y_te, preds_te_bls, average='macro', zero_division=0)
        rec_bls  = recall_score(Y_te, preds_te_bls,   average='macro', zero_division=0)
        f1_bls   = f1_score(Y_te,    preds_te_bls,    average='macro')

        print(f"[INJURY |Fold{fold+1}] BLS acc={acc_bls:.4f}  F1={f1_bls:.4f} | Fusion acc={acc_fusion:.4f}  F1={f1_fusion:.4f}")

        df_bls = pd.DataFrame([{"fold": fold+1, "acc": acc_bls, "precision": prec_bls, "recall": rec_bls, "f1": f1_bls}])
        df_bls.to_csv(os.path.join(fold_dir_inj, "bls_metrics.csv"), index=False, encoding="utf-8-sig")
        bls_metrics_all.append(df_bls)

        prec_c, rec_c, f1_c, supp_c = precision_recall_fscore_support(
            Y_te, preds_te_bls, labels=list(range(C_inj)), average=None, zero_division=0
        )
        df_bls_per_class = pd.DataFrame({
            "fold": fold+1,
            "class_id": np.arange(C_inj, dtype=int),
            "action": [actions_inj[i] for i in range(C_inj)],
            "precision": prec_c.astype(float),
            "recall":    rec_c.astype(float),
            "f1":        f1_c.astype(float),
            "support":   supp_c.astype(int),
        })
        df_bls_per_class.to_csv(os.path.join(fold_dir_inj, "bls_metrics_per_class.csv"),
                                index=False, encoding="utf-8-sig")
        bls_classwise_all.append(df_bls_per_class)

        # === 贡献度（RAW + RowNorm100） ===
        contrib = np.zeros((C_inj, NUM_CH), dtype=float)
        for c_idx, bls in enumerate(bls_list):
            beta_c = bls.beta.detach().cpu().numpy()
            contrib[:, c_idx] = np.abs(beta_c).sum(axis=0)
        df_raw  = pd.DataFrame(contrib, index=actions_inj, columns=[f"Ch{i+1}" for i in range(NUM_CH)])
        df_raw.to_excel(os.path.join(fold_dir_inj, 'channel_to_class_contrib_raw.xlsx'))

        # —— 追加到“每折长表”：列= fold, action, channel, value（原始 |β| 通道和）
        for act_idx, act_name in enumerate(actions_inj):
            for ch in range(NUM_CH):
                INJURY_CONTRIB_LONG.append({
                    "fold": fold+1,
                    "action": str(act_name),
                    "channel": f"Ch{ch+1}",
                    "value": float(contrib[act_idx, ch])
                })

        X = df_raw.to_numpy(dtype=float)
        row_sum = np.nansum(X, axis=1, keepdims=True)
        X_norm = np.zeros_like(X)
        ok = (row_sum > 0).squeeze()
        X_norm[ok, :] = (X[ok, :] / row_sum[ok]) * 100.0
        tails = ~np.isclose(X_norm.sum(axis=1), 100.0, rtol=0, atol=1e-10)
        for r in np.where(tails)[0]:
            diff = 100.0 - X_norm[r].sum()
            X_norm[r, -1] += diff
        df_norm = pd.DataFrame(X_norm, index=actions_inj, columns=[f"Ch{i+1}" for i in range(NUM_CH)])
        df_norm.to_excel(os.path.join(fold_dir_inj, 'channel_to_class_contrib_rowNorm100.xlsx'))
        verify = pd.DataFrame({"Action": df_norm.index,
                               "RowSum": df_norm.sum(axis=1).round(10),
                               "Equal_100": (df_norm.sum(axis=1).round(10) == 100.0)})
        verify.to_csv(os.path.join(fold_dir_inj, 'verify_row_sum_100.csv'), index=False, encoding="utf-8-sig")

        all_contribs_raw.append(df_raw.to_numpy())
        all_contribs_norm.append(df_norm.to_numpy())
        all_metrics.append({'fold': fold+1, 'acc_fusion': acc_fusion, 'f1_fusion': f1_fusion,
                            'acc_bls': acc_bls, 'prec_bls': prec_bls, 'rec_bls': rec_bls, 'f1_bls': f1_bls})

    # 汇总（受伤）
    df_m = pd.DataFrame(all_metrics)
    df_m.to_csv(os.path.join(INJURY_OUT_DIR, 'summary_metrics.csv'), index=False, encoding='utf-8-sig')
    if bls_metrics_all:
        df_bls_all = pd.concat(bls_metrics_all, ignore_index=True)
        mean_row = {"fold":"mean","acc":df_bls_all["acc"].mean(),"precision":df_bls_all["precision"].mean(),
                    "recall":df_bls_all["recall"].mean(),"f1":df_bls_all["f1"].mean()}
        df_bls_all = pd.concat([df_bls_all, pd.DataFrame([mean_row])], ignore_index=True)
        df_bls_all.to_csv(os.path.join(INJURY_OUT_DIR, "summary_bls_metrics.csv"),
                          index=False, encoding="utf-8-sig")
    if bls_classwise_all:
        df_bls_cls_all = pd.concat(bls_classwise_all, ignore_index=True)
        df_cls_mean = (df_bls_cls_all.groupby(["class_id","action"], as_index=False)
                       .agg(precision=("precision","mean"), recall=("recall","mean"),
                            f1=("f1","mean"), support=("support","sum")))
        df_cls_mean.to_csv(os.path.join(INJURY_OUT_DIR,"summary_bls_metrics_per_class_mean.csv"),
                           index=False, encoding="utf-8-sig")
        def _wavg(g, col):
            w=g["support"].to_numpy(float); x=g[col].to_numpy(float); s=w.sum()
            return float(np.average(x, weights=w)) if s>0 else float(np.nan)
        rows=[]
        for (cid,act),g in df_bls_cls_all.groupby(["class_id","action"]):
            rows.append({"class_id":cid,"action":act,
                         "precision":_wavg(g,"precision"),
                         "recall":_wavg(g,"recall"),
                         "f1":_wavg(g,"f1"),
                         "support":int(g["support"].sum())})
        pd.DataFrame(rows).to_csv(os.path.join(INJURY_OUT_DIR,"summary_bls_metrics_per_class_weighted.csv"),
                                  index=False, encoding="utf-8-sig")
    if len(all_contribs_raw):
        mean_raw = np.mean(np.stack(all_contribs_raw, axis=0), axis=0)
        pd.DataFrame(mean_raw, index=actions_inj, columns=[f"Ch{i+1}" for i in range(NUM_CH)]) \
          .to_excel(os.path.join(INJURY_OUT_DIR,'channel_to_class_contrib_mean_RAW.xlsx'))
        export_contrib_boxplots(all_contribs_raw, actions_inj, INJURY_OUT_DIR,
                                tag="raw_injury", y_label="Contribution (a.u.)", as_percent=False)
    if len(all_contribs_norm):
        mean_norm = np.mean(np.stack(all_contribs_norm, axis=0), axis=0)
        pd.DataFrame(mean_norm, index=actions_inj, columns=[f"Ch{i+1}" for i in range(NUM_CH)]) \
          .to_excel(os.path.join(INJURY_OUT_DIR,'channel_to_class_contrib_mean_rowNorm100.xlsx'))
        export_contrib_boxplots(all_contribs_norm, actions_inj, INJURY_OUT_DIR,
                                tag="rowNorm100_injury", y_label="Contribution (%)", as_percent=True)

    # ===== 导出：受伤阶段 |β| 绝对值通道求和（每折合在一起 + 跨折求和） =====
    if INJURY_CONTRIB_LONG:
        df_long = pd.DataFrame(INJURY_CONTRIB_LONG)
        out_long_csv = os.path.join(INJURY_OUT_DIR, "contrib_abs_by_fold_injury_long.csv")
        df_long.to_csv(out_long_csv, index=False, encoding="utf-8-sig")
        print("[保存]", out_long_csv)

        df_sum = (df_long
                  .groupby(["action", "channel"], as_index=False)["value"]
                  .sum())
        out_sum_csv = os.path.join(INJURY_OUT_DIR, "contrib_abs_sum_over_folds_injury_long.csv")
        df_sum.to_csv(out_sum_csv, index=False, encoding="utf-8-sig")
        print("[保存]", out_sum_csv)

        df_sum_wide = df_sum.pivot(index="action", columns="channel", values="value").fillna(0.0)
        out_sum_xlsx = os.path.join(INJURY_OUT_DIR, "contrib_abs_sum_over_folds_injury.xlsx")
        df_sum_wide.to_excel(out_sum_xlsx)
        print("[保存]", out_sum_xlsx)

    print("\n[INJURY] Done.")

# =============== 主函数（修改版）===============
def main():
    set_seed(SEED)
    mvc_bank_cache = {}
    
    print("="*60)
    print("BLS 两阶段训练：健康 → 受伤")
    print("="*60)
    
    # ========== 检查健康阶段权重是否存在 ==========
    healthy_weights_exist = check_weights_exist(HEALTHY_OUT_DIR, phase="healthy")
    
    if healthy_weights_exist:
        print("\n" + "="*60)
        print("检测到健康阶段权重已存在，跳过健康阶段训练")
        print("="*60)
        
        # 尝试加载健康MVC字典
        healthy_mvc_bank = load_healthy_mvc_bank(HEALTHY_OUT_DIR)
        
        if healthy_mvc_bank is None and USE_MVC:
            print("[警告] 未能加载健康MVC字典，将在受伤阶段重新计算")
        
    else:
        print("\n" + "="*60)
        print("健康阶段权重不完整，开始健康阶段训练")
        print("="*60)
        
        # 阶段A：健康训练（完整运行）
        # [此处插入完整的 run_phase_healthy 函数代码]
        healthy_mvc_bank = run_phase_healthy(mvc_bank_cache)
    
    # ========== 阶段B：受伤阶段（总是运行）==========
    print("\n" + "="*60)
    print("开始受伤阶段训练")
    print("="*60)
    
    run_phase_injury(mvc_bank_cache, healthy_mvc_bank)
    
    print("\n" + "="*60)
    print("所有流程完成！")
    print("="*60)

if __name__ == "__main__":
    main()
