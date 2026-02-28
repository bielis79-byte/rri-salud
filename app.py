import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.signal import welch
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


# ============================================================
# IO
# ============================================================

def load_rri_txt(file) -> np.ndarray:
    df = pd.read_csv(file, header=None)
    rri = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=float)

    if len(rri) == 0:
        raise ValueError("Archivo vacío o no numérico.")

    # seconds vs ms heuristic
    if np.nanmedian(rri) < 10.0:
        rri *= 1000.0

    return rri


# ============================================================
# Cleaning
# ============================================================

def mad_outlier_mask(x: np.ndarray, thresh: float = 4.5) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.ones_like(x, dtype=bool)
    robust_z = 0.6745 * (x - med) / mad
    return np.abs(robust_z) <= thresh


def clean_rri_ms(rri_ms: np.ndarray, mad_thresh: float = 4.5) -> dict:
    rri_ms = rri_ms.astype(float)

    # plausibility
    plausible = (rri_ms >= 300) & (rri_ms <= 2000)
    rri_ms = rri_ms[plausible]

    if len(rri_ms) < 50:
        return {"rri_clean": rri_ms, "removed_pct": np.nan}

    good = mad_outlier_mask(rri_ms, thresh=mad_thresh)
    removed_pct = 100.0 * (1.0 - good.mean())

    if good.all():
        return {"rri_clean": rri_ms, "removed_pct": removed_pct}

    idx = np.arange(len(rri_ms))
    f = interp1d(idx[good], rri_ms[good], kind="linear", fill_value="extrapolate")
    rri_interp = f(idx)

    return {"rri_clean": rri_interp, "removed_pct": removed_pct}


# ============================================================
# Window selection (last N minutes)
# ============================================================

def last_n_minutes_rri(rri_ms: np.ndarray, minutes: float = 5.0) -> np.ndarray:
    if len(rri_ms) == 0:
        return rri_ms
    t = np.cumsum(rri_ms / 1000.0)
    start = t[-1] - minutes * 60.0
    return rri_ms[t >= start]


# ============================================================
# Time-domain HRV
# ============================================================

def time_domain_metrics(rri_ms: np.ndarray) -> dict:
    x = np.asarray(rri_ms, dtype=float)
    if len(x) < 3:
        return {k: np.nan for k in [
            "MeanRR", "SDNN", "MeanHR", "STDHR", "RMSSD",
            "NN50", "pNN50", "TriIndex", "TINN"
        ]}

    mean_rr = float(np.mean(x))
    sdnn = float(np.std(x, ddof=1))

    hr = 60000.0 / x
    mean_hr = float(np.mean(hr))
    std_hr = float(np.std(hr, ddof=1))

    diff = np.diff(x)
    rmssd = float(np.sqrt(np.mean(diff**2)))

    nn50 = int(np.sum(np.abs(diff) > 50.0))
    pnn50 = float(100.0 * nn50 / len(diff)) if len(diff) > 0 else np.nan

    tri, tinn = triangular_index_and_tinn(x, bin_ms=7.8125)

    return {
        "MeanRR": mean_rr,
        "SDNN": sdnn,
        "MeanHR": mean_hr,
        "STDHR": std_hr,
        "RMSSD": rmssd,
        "NN50": nn50,
        "pNN50": pnn50,
        "TriIndex": tri,
        "TINN": tinn,
    }


def triangular_index_and_tinn(rri_ms: np.ndarray, bin_ms: float = 7.8125) -> tuple[float, float]:
    """
    RR Triangular Index = N / max(hist)
    TINN = baseline width of best-fitting triangle (approx implementation)
    """
    x = np.asarray(rri_ms, dtype=float)
    if len(x) < 10:
        return np.nan, np.nan

    # Histogram
    x_min = np.min(x)
    x_max = np.max(x)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return np.nan, np.nan

    nbins = int(np.ceil((x_max - x_min) / bin_ms))
    nbins = max(nbins, 10)
    counts, edges = np.histogram(x, bins=nbins, range=(x_min, x_max))
    centers = (edges[:-1] + edges[1:]) / 2.0

    hmax = np.max(counts) if np.max(counts) > 0 else np.nan
    tri_index = float(len(x) / hmax) if np.isfinite(hmax) else np.nan

    # TINN: coarse triangle fit search
    peak_idx = int(np.argmax(counts))
    if counts[peak_idx] <= 0:
        return tri_index, np.nan

    left_candidates = range(0, peak_idx)
    right_candidates = range(peak_idx + 1, len(counts))

    if peak_idx < 2 or peak_idx > len(counts) - 3:
        return tri_index, np.nan

    best_err = np.inf
    best_L = None
    best_R = None
    peak_x = centers[peak_idx]
    peak_y = counts[peak_idx]

    stepL = max(1, len(left_candidates)//40)
    stepR = max(1, len(right_candidates)//40)

    for Li in list(left_candidates)[::stepL]:
        for Ri in list(right_candidates)[::stepR]:
            if Ri <= Li:
                continue
            Lx, Rx = centers[Li], centers[Ri]
            if not (Lx < peak_x < Rx):
                continue

            yhat = np.zeros_like(counts, dtype=float)
            left_mask = (centers >= Lx) & (centers <= peak_x)
            yhat[left_mask] = peak_y * (centers[left_mask] - Lx) / (peak_x - Lx)

            right_mask = (centers >= peak_x) & (centers <= Rx)
            yhat[right_mask] = peak_y * (Rx - centers[right_mask]) / (Rx - peak_x)

            err = np.mean((counts - yhat) ** 2)
            if err < best_err:
                best_err = err
                best_L, best_R = Li, Ri

    if best_L is None or best_R is None:
        return tri_index, np.nan

    tinn = float(centers[best_R] - centers[best_L])
    return tri_index, tinn


# ============================================================
# Poincaré
# ============================================================

def poincare_sd1_sd2(rri_ms: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(rri_ms, dtype=float)
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    diff = np.diff(x)
    var_rr = np.var(x, ddof=1)
    var_diff = np.var(diff, ddof=1)
    sd1 = np.sqrt(var_diff / 2.0)
    sd2 = np.sqrt(max(0.0, 2.0 * var_rr - var_diff / 2.0))
    ratio = sd2 / sd1 if sd1 > 0 else np.nan
    return float(sd1), float(sd2), float(ratio)


# ============================================================
# ApEn / SampEn (Kubios-like defaults)
# ============================================================

def _phi_count(x: np.ndarray, m: int, r: float) -> float:
    n = len(x)
    if n <= m + 1:
        return np.nan
    X = np.array([x[i:i+m] for i in range(n - m + 1)])
    C = []
    for i in range(len(X)):
        d = np.max(np.abs(X - X[i]), axis=1)
        C.append(np.sum(d <= r) / len(d))
    C = np.asarray(C, dtype=float)
    C = C[C > 0]
    if len(C) == 0:
        return np.nan
    return float(np.mean(np.log(C)))


def apen(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < m + 5:
        return np.nan
    sd = np.std(x, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    r = r_factor * sd
    phi_m = _phi_count(x, m, r)
    phi_m1 = _phi_count(x, m+1, r)
    if not np.isfinite(phi_m) or not np.isfinite(phi_m1):
        return np.nan
    return float(phi_m - phi_m1)


def sampen(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < m + 5:
        return np.nan

    sd = np.std(x, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    r = r_factor * sd

    def _count(mm: int) -> int:
        count = 0
        for i in range(n - mm):
            xi = x[i:i+mm]
            for j in range(i+1, n - mm):
                xj = x[j:j+mm]
                if np.max(np.abs(xi - xj)) <= r:
                    count += 1
        return count

    B = _count(m)
    A = _count(m+1)
    if B == 0 or A == 0:
        return np.nan
    return float(-np.log(A / B))


# ============================================================
# DFA
# ============================================================

def dfa_alpha(rri_ms: np.ndarray, short_range=(4, 12), long_range=(13, 64)) -> tuple[float, float]:
    x = np.asarray(rri_ms, dtype=float)
    if len(x) < 200:
        return np.nan, np.nan

    x = x - np.mean(x)
    y = np.cumsum(x)

    def fluctuation(box_n: int) -> float:
        segments = len(y) // box_n
        if segments < 2:
            return np.nan
        rms = []
        for i in range(segments):
            seg = y[i*box_n:(i+1)*box_n]
            t = np.arange(box_n)
            coeffs = np.polyfit(t, seg, 1)
            trend = np.polyval(coeffs, t)
            rms.append(np.sqrt(np.mean((seg - trend)**2)))
        return float(np.mean(rms)) if len(rms) else np.nan

    scales = np.unique(np.logspace(np.log10(4), np.log10(64), 20).astype(int))
    F = np.array([fluctuation(s) for s in scales], dtype=float)

    valid = np.isfinite(F) & (F > 0)
    scales = scales[valid]
    F = F[valid]

    if len(scales) < 8:
        return np.nan, np.nan

    log_s = np.log(scales)
    log_F = np.log(F)

    mask_s = (scales >= short_range[0]) & (scales <= short_range[1])
    mask_l = (scales >= long_range[0]) & (scales <= long_range[1])

    if mask_s.sum() < 3 or mask_l.sum() < 3:
        return np.nan, np.nan

    alpha1 = np.polyfit(log_s[mask_s], log_F[mask_s], 1)[0]
    alpha2 = np.polyfit(log_s[mask_l], log_F[mask_l], 1)[0]
    return float(alpha1), float(alpha2)


# ============================================================
# Frequency domain: Smoothness priors + Welch (FFT-based)
# ============================================================

def rri_to_evenly_sampled(rri_ms: np.ndarray, fs: float = 4.0):
    rri_ms = np.asarray(rri_ms, dtype=float)
    t = np.cumsum(rri_ms / 1000.0)

    if len(t) < 10 or (t[-1] - t[0]) < 10:
        return None, None

    t0 = t - t[0]
    t_uniform = np.arange(0.0, t0[-1], 1.0 / fs)
    if len(t_uniform) < 10:
        return None, None

    kind = "cubic" if len(t0) >= 4 else "linear"
    f = interp1d(t0, rri_ms, kind=kind, fill_value="extrapolate")
    x = f(t_uniform)
    x = x - np.mean(x)
    return t_uniform, x


def smoothness_priors_detrend(x: np.ndarray, lam: float = 500.0):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 10:
        return x
    if n - 2 <= 0:
        return x

    D2 = diags(
        [np.ones(n - 2), -2*np.ones(n - 2), np.ones(n - 2)],
        [0, 1, 2],
        shape=(n - 2, n),
        format="csr",
    )
    A = diags([np.ones(n)], [0], format="csr") + lam * (D2.T @ D2)
    z = spsolve(A, x)
    return x - z


def psd_welch_kubios(x: np.ndarray, fs: float, win_sec: float = 256.0, overlap: float = 0.5, points_per_hz: int = 256):
    if x is None:
        return None, None

    n = len(x)
    nperseg_target = int(fs * win_sec)
    nperseg = min(n, nperseg_target)
    if nperseg < 16:
        return None, None

    noverlap = int(nperseg * overlap)

    nfft_target = int(fs * points_per_hz)
    nfft = max(nperseg, nfft_target)

    f, pxx = welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling="density"
    )
    return f, pxx


def bandpower_from_psd(f, pxx, fmin, fmax):
    if f is None or pxx is None:
        return np.nan
    mask = (f >= fmin) & (f < fmax)
    if not np.any(mask):
        return np.nan
    return float(trapezoid(pxx[mask], f[mask]))


def compute_freq_bands(rri_ms: np.ndarray, fs: float = 4.0, lam: float = 500.0):
    _, x = rri_to_evenly_sampled(rri_ms, fs=fs)
    if x is None:
        return None

    x_dt = smoothness_priors_detrend(x, lam=lam)
    f, pxx = psd_welch_kubios(x_dt, fs=fs, win_sec=256.0, overlap=0.5, points_per_hz=256)
    if f is None:
        return None

    vlf = bandpower_from_psd(f, pxx, 0.0033, 0.04)
    lf  = bandpower_from_psd(f, pxx, 0.04,   0.15)
    hf  = bandpower_from_psd(f, pxx, 0.15,   0.40)
    total = vlf + lf + hf if np.isfinite(vlf) and np.isfinite(lf) and np.isfinite(hf) else np.nan
    lfhf = lf / hf if np.isfinite(lf) and np.isfinite(hf) and hf > 0 else np.nan

    denom = lf + hf if np.isfinite(lf) and np.isfinite(hf) and (lf + hf) > 0 else np.nan
    lfnu = 100.0 * lf / denom if np.isfinite(denom) else np.nan
    hfnu = 100.0 * hf / denom if np.isfinite(denom) else np.nan

    return {
        "VLF": float(vlf),
        "LF": float(lf),
        "HF": float(hf),
        "TOTAL": float(total) if np.isfinite(total) else np.nan,
        "LF_HF": float(lfhf) if np.isfinite(lfhf) else np.nan,
        "LFnu": float(lfnu) if np.isfinite(lfnu) else np.nan,
        "HFnu": float(hfnu) if np.isfinite(hfnu) else np.nan,
        "f": f,
        "pxx": pxx
    }


# ============================================================
# RQA (REC/DET/Lmean/Lmax/ShanEn) + Correlation Dimension D2
# ============================================================

def takens_embed(x: np.ndarray, m: int, tau: int):
    x = np.asarray(x, dtype=float)
    n = len(x)
    M = n - (m - 1) * tau
    if M <= 10:
        return None
    return np.array([x[i:i + m * tau:tau] for i in range(M)], dtype=float)


def rqa_metrics(rri_ms: np.ndarray, m: int = 10, tau: int = 1, eps_factor: float = 3.1623, lmin: int = 2):
    x = np.asarray(rri_ms, dtype=float)
    emb = takens_embed(x, m=m, tau=tau)
    if emb is None:
        return {"Lmean": np.nan, "Lmax": np.nan, "REC": np.nan, "DET": np.nan, "ShanEn": np.nan}

    M = emb.shape[0]
    if M < 50:
        return {"Lmean": np.nan, "Lmax": np.nan, "REC": np.nan, "DET": np.nan, "ShanEn": np.nan}

    d = squareform(pdist(emb, metric="euclidean"))
    eps = eps_factor * np.std(x, ddof=1)
    if not np.isfinite(eps) or eps <= 0:
        return {"Lmean": np.nan, "Lmax": np.nan, "REC": np.nan, "DET": np.nan, "ShanEn": np.nan}

    R = d < eps
    np.fill_diagonal(R, False)

    rec = 100.0 * (np.sum(R) / (M * (M - 1)))

    diag_lengths = []
    for k in range(1, M):
        diag = np.diagonal(R, offset=k)
        run = 0
        for v in diag:
            if v:
                run += 1
            else:
                if run > 0:
                    diag_lengths.append(run)
                    run = 0
        if run > 0:
            diag_lengths.append(run)

    diag_lengths = np.asarray(diag_lengths, dtype=int)
    if len(diag_lengths) == 0:
        return {"Lmean": np.nan, "Lmax": np.nan, "REC": float(rec), "DET": np.nan, "ShanEn": np.nan}

    lmax = int(np.max(diag_lengths))

    det_mask = diag_lengths >= lmin
    det_points = np.sum(diag_lengths[det_mask])
    all_diag_points = np.sum(diag_lengths)
    det = 100.0 * det_points / all_diag_points if all_diag_points > 0 else np.nan

    lmean = float(np.mean(diag_lengths[det_mask])) if np.any(det_mask) else np.nan

    shanen = np.nan
    if np.any(det_mask):
        lens = diag_lengths[det_mask]
        vals, counts = np.unique(lens, return_counts=True)
        p = counts / np.sum(counts)
        shanen = float(-np.sum(p * np.log(p + 1e-12)))

    return {"Lmean": lmean, "Lmax": float(lmax), "REC": float(rec), "DET": float(det), "ShanEn": shanen}


def correlation_dimension_d2(rri_ms: np.ndarray, m: int = 10, tau: int = 1, n_radii: int = 12, max_points: int = 800):
    x = np.asarray(rri_ms, dtype=float)
    emb = takens_embed(x, m=m, tau=tau)
    if emb is None:
        return np.nan

    N = emb.shape[0]
    if N < 200:
        return np.nan

    if N > max_points:
        idx = np.linspace(0, N - 1, max_points).astype(int)
        emb = emb[idx]
        N = emb.shape[0]

    d = pdist(emb, metric="euclidean")
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan

    dmin = np.percentile(d, 5)
    dmax = np.percentile(d, 95)
    if dmax <= dmin or dmin <= 0:
        return np.nan

    radii = np.logspace(np.log10(dmin), np.log10(dmax), n_radii)
    Cr = np.array([np.mean(d < r) for r in radii], dtype=float)

    valid = (Cr > 0) & np.isfinite(Cr)
    radii = radii[valid]
    Cr = Cr[valid]
    if len(Cr) < 6:
        return np.nan

    log_r = np.log(radii)
    log_C = np.log(Cr)

    lo = int(len(log_r) * 0.25)
    hi = int(len(log_r) * 0.75)
    if hi - lo < 3:
        return np.nan

    slope = np.polyfit(log_r[lo:hi], log_C[lo:hi], 1)[0]
    return float(slope)


# ============================================================
# IVA SEMÁFORO (criterios pactados)
#   - RMSSD: >20 verde
#   - SDNN:  >50 verde
#   - SampEn: 1.5–2 verde
#   - DFA1: 0.70–1.3 verde
#   - DFA2: 0.90–1.2 verde
#   - LFnu: 1–30 verde   (AJUSTE PEDIDO)
#   - HFnu: 70–99 verde  (AJUSTE PEDIDO)
#   - LMAX: >300 rojo (rigidez)
#   - NO usar LF/HF en el semáforo
# ============================================================

def _status_color_label(status: str):
    if status == "VERDE":
        return "🟢 VERDE"
    if status == "AMARILLO":
        return "🟡 AMARILLO"
    return "🔴 ROJO"


def _grade(value, green_range=None, yellow_range=None, red_rule=None, higher_is_better=None):
    if not np.isfinite(value):
        return "ROJO", "NA"

    if red_rule is not None:
        if red_rule(value):
            return "ROJO", ""
        if green_range is None and yellow_range is None and higher_is_better is None:
            return "AMARILLO", ""

    if higher_is_better is not None:
        green_min, yellow_min = higher_is_better
        if value >= green_min:
            return "VERDE", f"≥{green_min}"
        if value >= yellow_min:
            return "AMARILLO", f"{yellow_min}–{green_min}"
        return "ROJO", f"<{yellow_min}"

    if green_range is not None and (green_range[0] <= value <= green_range[1]):
        return "VERDE", f"{green_range[0]}–{green_range[1]}"
    if yellow_range is not None and (yellow_range[0] <= value <= yellow_range[1]):
        return "AMARILLO", f"{yellow_range[0]}–{yellow_range[1]}"
    return "ROJO", "fuera de rango"


def _grade_lfnu(lfnu: float):
    if not np.isfinite(lfnu):
        return "ROJO", "NA"
    # Verde 1–30 (nuevo)
    if 1.0 <= lfnu <= 30.0:
        return "VERDE", "1–30"
    # Amarillo: 0.5–1 o 30–40
    if (0.5 <= lfnu < 1.0) or (30.0 < lfnu <= 40.0):
        return "AMARILLO", "0.5–1 o 30–40"
    # Rojo: <0.5 o >40
    return "ROJO", "<0.5 o >40"


def _grade_hfnu(hfnu: float):
    if not np.isfinite(hfnu):
        return "ROJO", "NA"
    # Verde 70–99 (nuevo)
    if 70.0 <= hfnu <= 99.0:
        return "VERDE", "70–99"
    # Amarillo: 60–70 o 99–100
    if (60.0 <= hfnu < 70.0) or (99.0 < hfnu <= 100.0):
        return "AMARILLO", "60–70 o 99–100"
    # Rojo: <60 o >100
    return "ROJO", "<60 o >100"


def compute_iva_semaforo(mean_hr, rmssd, sdnn, lfnu, hfnu, sampen_val, dfa1, dfa2, lmax):
    rows = []

    st_hr, _ = _grade(mean_hr, green_range=(50, 90), yellow_range=(40, 100),
                      red_rule=lambda v: (v < 40) or (v > 100))
    rows.append(("FC (MeanHR, bpm)", mean_hr, st_hr, "Verde 50–90; Amarillo 40–50/90–100; Rojo <40 o >100"))

    st_rmssd, _ = _grade(rmssd, higher_is_better=(20.0, 15.0))
    rows.append(("RMSSD (ms)", rmssd, st_rmssd, "Verde >20; Amarillo 15–20; Rojo <15"))

    st_sdnn, _ = _grade(sdnn, higher_is_better=(50.0, 30.0))
    rows.append(("SDNN (ms)", sdnn, st_sdnn, "Verde >50; Amarillo 30–50; Rojo <30"))

    st_se, _ = _grade(sampen_val, green_range=(1.5, 2.0), yellow_range=(1.2, 2.5))
    rows.append(("SampEn", sampen_val, st_se, "Verde 1.5–2.0; Amarillo 1.2–1.5 o 2.0–2.5; Rojo <1.2 (o muy fuera)"))

    st_dfa1, _ = _grade(dfa1, green_range=(0.70, 1.30), yellow_range=(0.60, 1.40))
    rows.append(("DFA α1", dfa1, st_dfa1, "Verde 0.70–1.30; Amarillo 0.60–0.70 o 1.30–1.40; Rojo <0.60 o >1.40"))

    st_dfa2, _ = _grade(dfa2, green_range=(0.90, 1.20), yellow_range=(0.80, 1.30))
    rows.append(("DFA α2", dfa2, st_dfa2, "Verde 0.90–1.20; Amarillo 0.80–0.90 o 1.20–1.30; Rojo <0.80 o >1.30"))

    st_lfnu, lf_ref = _grade_lfnu(lfnu)
    rows.append(("LFnu (%)", lfnu, st_lfnu, f"Verde {lf_ref} (ajustado)"))

    st_hfnu, hf_ref = _grade_hfnu(hfnu)
    rows.append(("HFnu (%)", hfnu, st_hfnu, f"Verde {hf_ref} (ajustado)"))

    if not np.isfinite(lmax):
        st_lmax = "ROJO"
        rows.append(("LMAX", lmax, st_lmax, "Rigidez alta si >300"))
    else:
        if lmax > 300:
            st_lmax = "ROJO"
        elif 250 <= lmax <= 300:
            st_lmax = "AMARILLO"
        else:
            st_lmax = "VERDE"
        rows.append(("LMAX", lmax, st_lmax, "Verde <250; Amarillo 250–300; Rojo >300 (rigidez)"))

    score_map = {"ROJO": 0, "AMARILLO": 1, "VERDE": 2}
    scores = [score_map[r[2]] for r in rows]
    avg = float(np.mean(scores)) if len(scores) else 0.0
    n_red = sum(1 for r in rows if r[2] == "ROJO")

    if n_red >= 2 or avg < 1.0:
        global_status = "ROJO"
    elif n_red == 0 and avg >= 1.6:
        global_status = "VERDE"
    else:
        global_status = "AMARILLO"

    df = pd.DataFrame(rows, columns=["Métrica", "Valor", "Estado", "Criterio"])
    return global_status, avg, n_red, df


# ============================================================
# Streamlit APP
# ============================================================

st.set_page_config(page_title="IVM — Kubios-like (RRi)", layout="wide")
st.title("IVM — Time/Freq/Nonlinear (Kubios-like) — últimos N min")
st.caption("Carga RRi (EliteHRV: 1 RR por línea), limpia, recorta a últimos N minutos y calcula métricas. (No diagnóstica).")

with st.sidebar:
    st.header("Parámetros")

    minutes = st.number_input("Ventana analizada (min)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
    mad_thresh = st.slider("Umbral outliers (MAD z)", 3.0, 8.0, 4.5, 0.1)

    st.subheader("DFA (latidos)")
    a1_lo = st.number_input("α1 mínimo", value=4, min_value=3, max_value=20)
    a1_hi = st.number_input("α1 máximo", value=12, min_value=8, max_value=40)
    a2_lo = st.number_input("α2 mínimo", value=13, min_value=8, max_value=64)
    a2_hi = st.number_input("α2 máximo", value=64, min_value=32, max_value=200)

    st.subheader("RQA (Kubios-like)")
    m = st.number_input("Embedding m", value=10, min_value=2, max_value=20)
    tau = st.number_input("Delay τ", value=1, min_value=1, max_value=10)
    eps_factor = st.number_input("Threshold (×SD)", value=3.1623, min_value=0.1, max_value=10.0, step=0.1, format="%.4f")
    lmin = st.number_input("Diagonal lmin (DET/Lmean)", value=2, min_value=2, max_value=10, step=1)

    st.subheader("Espectro (FFT/Welch + smoothness priors)")
    fs_spec = st.selectbox("Interpolación (Hz)", [2.0, 4.0], index=1)
    lam_sp = st.number_input("λ detrending (smoothness priors)", min_value=0.0, max_value=5000.0, value=500.0, step=50.0)
    show_units = st.radio("Mostrar bandas en:", ["Absoluto (ms²)", "Normalizado (nu)"], index=0)

uploaded = st.file_uploader("Sube tu archivo RRi (.txt o .csv con 1 columna)", type=["txt", "csv"])

if uploaded is None:
    st.info("Sube un .txt de Elite HRV (un RRi por línea).")
    st.stop()

try:
    rri_raw = load_rri_txt(uploaded)
    cleaned = clean_rri_ms(rri_raw, mad_thresh=mad_thresh)
    rri_clean = cleaned["rri_clean"]
    removed_pct = cleaned["removed_pct"]

    if len(rri_clean) < 100:
        st.warning("Muy pocos RRi tras limpieza. Revisa el archivo.")
        st.stop()

    rri_win = last_n_minutes_rri(rri_clean, minutes=float(minutes))
    if len(rri_win) < 50:
        st.warning("Ventana demasiado corta tras recorte. Sube la ventana o revisa el archivo.")
        st.stop()

    total_min = (np.sum(rri_clean) / 1000.0) / 60.0
    win_min = (np.sum(rri_win) / 1000.0) / 60.0

    td = time_domain_metrics(rri_win)
    spec = compute_freq_bands(rri_win, fs=float(fs_spec), lam=float(lam_sp))

    sd1, sd2, sd2sd1 = poincare_sd1_sd2(rri_win)
    a1, a2 = dfa_alpha(rri_win, short_range=(int(a1_lo), int(a1_hi)), long_range=(int(a2_lo), int(a2_hi)))
    ap = apen(rri_win, m=2, r_factor=0.2)
    se = sampen(rri_win, m=2, r_factor=0.2)
    rqa = rqa_metrics(rri_win, m=int(m), tau=int(tau), eps_factor=float(eps_factor), lmin=int(lmin))
    d2 = correlation_dimension_d2(rri_win, m=int(m), tau=int(tau), n_radii=12, max_points=800)

    lfnu = spec["LFnu"] if spec is not None else np.nan
    hfnu = spec["HFnu"] if spec is not None else np.nan
    lmax_for_semaforo = rqa["Lmax"] if rqa is not None else np.nan

    iva_status, iva_score, iva_nred, iva_df = compute_iva_semaforo(
        mean_hr=td["MeanHR"],
        rmssd=td["RMSSD"],
        sdnn=td["SDNN"],
        lfnu=lfnu,
        hfnu=hfnu,
        sampen_val=se,
        dfa1=a1,
        dfa2=a2,
        lmax=lmax_for_semaforo
    )

    c1, c2 = st.columns([1.05, 1.0])

    with c1:
        st.subheader("Semáforo — Índice de Vulnerabilidad Autonómica (IVA)")
        st.markdown(f"### {_status_color_label(iva_status)}")
        st.write(f"Score medio (0–2): **{iva_score:.2f}**  |  Nº ROJOS: **{iva_nred}**")
        st.dataframe(iva_df, use_container_width=True)

        st.divider()

        st.subheader("Control de calidad")
        st.write(f"- RRi totales (tras limpieza): **{len(rri_clean)}**")
        st.write(f"- Duración estimada total: **{total_min:.2f} min**")
        st.write(f"- RRi en ventana: **{len(rri_win)}**")
        st.write(f"- Duración estimada ventana: **{win_min:.2f} min** (objetivo {minutes:.1f} min)")
        st.write(f"- % puntos sospechosos/interpolados (MAD): **{removed_pct:.2f}%**" if np.isfinite(removed_pct) else "- % interpolado: NA")

        st.subheader("Time-domain")
        st.metric("Mean RR (ms)", f"{td['MeanRR']:.2f}" if np.isfinite(td["MeanRR"]) else "NA")
        st.metric("SDNN (ms)", f"{td['SDNN']:.2f}" if np.isfinite(td["SDNN"]) else "NA")
        st.metric("Mean HR (bpm)", f"{td['MeanHR']:.2f}" if np.isfinite(td["MeanHR"]) else "NA")
        st.metric("STD HR (bpm)", f"{td['STDHR']:.2f}" if np.isfinite(td["STDHR"]) else "NA")
        st.metric("RMSSD (ms)", f"{td['RMSSD']:.2f}" if np.isfinite(td["RMSSD"]) else "NA")
        st.metric("NN50", f"{td['NN50']}" if np.isfinite(td["NN50"]) else "NA")
        st.metric("pNN50 (%)", f"{td['pNN50']:.2f}" if np.isfinite(td["pNN50"]) else "NA")
        st.metric("RR Triangular Index", f"{td['TriIndex']:.3f}" if np.isfinite(td["TriIndex"]) else "NA")
        st.metric("TINN (ms)", f"{td['TINN']:.2f}" if np.isfinite(td["TINN"]) else "NA")

        st.subheader("Frequency-domain")
        if spec is None:
            st.warning("No se pudo calcular el espectro (señal insuficiente tras interpolación/Welch).")
        else:
            if show_units == "Absoluto (ms²)":
                st.metric("VLF (ms²)", f"{spec['VLF']:.3f}" if np.isfinite(spec["VLF"]) else "NA")
                st.metric("LF (ms²)", f"{spec['LF']:.3f}" if np.isfinite(spec["LF"]) else "NA")
                st.metric("HF (ms²)", f"{spec['HF']:.3f}" if np.isfinite(spec["HF"]) else "NA")
                st.metric("TOTAL (ms²)", f"{spec['TOTAL']:.3f}" if np.isfinite(spec["TOTAL"]) else "NA")
            else:
                st.metric("LF (nu)", f"{spec['LFnu']:.1f}" if np.isfinite(spec["LFnu"]) else "NA")
                st.metric("HF (nu)", f"{spec['HFnu']:.1f}" if np.isfinite(spec["HFnu"]) else "NA")

            # se muestra, pero NO se usa para el semáforo
            st.metric("LF/HF", f"{spec['LF_HF']:.3f}" if np.isfinite(spec["LF_HF"]) else "NA")

        st.subheader("Nonlinear")
        st.metric("SD1 (ms)", f"{sd1:.2f}" if np.isfinite(sd1) else "NA")
        st.metric("SD2 (ms)", f"{sd2:.2f}" if np.isfinite(sd2) else "NA")
        st.metric("SD2/SD1", f"{sd2sd1:.3f}" if np.isfinite(sd2sd1) else "NA")

        st.metric("Lmean", f"{rqa['Lmean']:.3f}" if np.isfinite(rqa["Lmean"]) else "NA")
        st.metric("Lmax", f"{rqa['Lmax']:.0f}" if np.isfinite(rqa["Lmax"]) else "NA")
        st.metric("REC (%)", f"{rqa['REC']:.2f}" if np.isfinite(rqa["REC"]) else "NA")
        st.metric("DET (%)", f"{rqa['DET']:.2f}" if np.isfinite(rqa["DET"]) else "NA")
        st.metric("ShanEn", f"{rqa['ShanEn']:.3f}" if np.isfinite(rqa["ShanEn"]) else "NA")

        st.metric("ApEn (m=2, r=0.2·SD)", f"{ap:.3f}" if np.isfinite(ap) else "NA")
        st.metric("SampEn (m=2, r=0.2·SD)", f"{se:.3f}" if np.isfinite(se) else "NA")
        st.metric("DFA α1", f"{a1:.3f}" if np.isfinite(a1) else "NA")
        st.metric("DFA α2", f"{a2:.3f}" if np.isfinite(a2) else "NA")
        st.metric("D2 (corr. dimension)", f"{d2:.3f}" if np.isfinite(d2) else "NA")

        out = {
            "MeanRR_ms": td["MeanRR"],
            "SDNN_ms": td["SDNN"],
            "MeanHR_bpm": td["MeanHR"],
            "STDHR_bpm": td["STDHR"],
            "RMSSD_ms": td["RMSSD"],
            "NN50": td["NN50"],
            "pNN50_pct": td["pNN50"],
            "RR_Triangular_Index": td["TriIndex"],
            "TINN_ms": td["TINN"],

            "VLF_ms2": spec["VLF"] if spec else np.nan,
            "LF_ms2": spec["LF"] if spec else np.nan,
            "HF_ms2": spec["HF"] if spec else np.nan,
            "TOTAL_ms2": spec["TOTAL"] if spec else np.nan,
            "LF_HF": spec["LF_HF"] if spec else np.nan,
            "LF_nu": spec["LFnu"] if spec else np.nan,
            "HF_nu": spec["HFnu"] if spec else np.nan,

            "SD1_ms": sd1,
            "SD2_ms": sd2,
            "SD2_SD1": sd2sd1,
            "Lmean": rqa["Lmean"],
            "Lmax": rqa["Lmax"],
            "REC_pct": rqa["REC"],
            "DET_pct": rqa["DET"],
            "ShanEn": rqa["ShanEn"],
            "ApEn": ap,
            "SampEn": se,
            "DFA_a1": a1,
            "DFA_a2": a2,
            "D2": d2,

            "IVA_status": iva_status,
            "IVA_score_mean_0_2": iva_score,
            "IVA_n_red": iva_nred,

            "N_clean_total": len(rri_clean),
            "N_window": len(rri_win),
            "minutes_total": total_min,
            "minutes_window": win_min,
            "removed_pct_MAD": removed_pct,
        }

        out_df = pd.DataFrame([out])
        st.download_button(
            "Descargar métricas (CSV)",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name="metricas_IVM_completo_con_IVA.csv",
            mime="text/csv",
        )

    with c2:
        st.subheader("RRi")
        fig1 = plt.figure()
        plt.plot(rri_clean, linewidth=1)
        plt.title("RRi limpio (toda la grabación)")
        plt.xlabel("Latido")
        plt.ylabel("ms")
        st.pyplot(fig1)

        fig2 = plt.figure()
        plt.plot(rri_win, linewidth=1)
        plt.title(f"RRi (ventana últimos {minutes:.1f} min)")
        plt.xlabel("Latido (ventana)")
        plt.ylabel("ms")
        st.pyplot(fig2)

        if spec is not None:
            st.subheader("PSD (Welch) — ventana")
            f = spec["f"]
            pxx = spec["pxx"]
            fig3 = plt.figure()
            plt.plot(f, pxx, linewidth=1)
            plt.xlim(0, 0.5)
            plt.xlabel("Hz")
            plt.ylabel("PSD (ms²/Hz)")
            plt.title("PSD (FFT/Welch) tras detrending λ (smoothness priors)")
            st.pyplot(fig3)

except Exception as e:

    st.error(f"Error procesando el archivo: {e}")
