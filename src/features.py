"""Feature extraction for the Panasonic NCR18650B ageing dataset (Zenodo 7896429).

One output row = one Reference Test Protocol (RTP) of one cell.
SOH target = 1C discharge capacity (4.2 V -> 2.5 V) / capacity at the cell's first RTP@25C.

Key facts about the raw files (verified on 2_009.h5):
* Group ``tests/test_XXXX`` corresponds to ``info.idtest == XXXX``. The ``info`` rows are NOT
  stored in test order, so zipping ``info`` with sorted group names mislabels tests.
* ``chargecap`` / ``dischargecap`` are counters that reset inside a test; capacities must be
  computed as differences within one step block.
* Capacity depends strongly on test temperature (RTP05..RTP45), so SOH is only defined on
  RTP@25C tests.

Usage:
    python -m src.features data/raw results/rtp_features.csv
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

C_NOM = 3.35  # A; the cycler's 1C current (see 1C discharge at -3.35 A)

# Load-profile design parameters (Table 2 of the dataset paper). T in C, currents in C-rate,
# F in Hz, SoC / dSoC in %. V212 is omitted because its row in the paper is malformed.
PROFILE_PARAMS = {
    "L35": (40, 0.2, 2, 3, 0.5, 55, 80), "L26": (40, 0.2, 0.2, 0.2, 0.0003, 55, 80),
    "C06": (40, 0, 0, 0, 0.0003, 95, 0.01), "L03": (40, 0.2, 2, 3, 0.2, 25, 2.5),
    "L09": (5, 0.2, 1, 3, 0.2, 80, 2.5), "L21": (5, 0.2, 0.2, 0.2, 0.0003, 80, 15),
    "L06": (20, 0.5, 0.2, 0.2, 0.0003, 25, 2.5), "L201": (0, 0, 0, 0, 0, 15, 0),
    "L202": (45, 0, 0, 0, 0, 15, 0), "L203": (0, 0, 0, 0, 0, 55, 0),
    "L204": (30, 0, 0, 0, 0, 55, 0), "L205": (20, 0, 0, 0, 0, 95, 0),
    "L206": (45, 0, 0, 0, 0, 95, 0), "L207": (20, 0.05, 0.9, 1.5, 0.25, 25, 2.5),
    "L208": (0, 0.05, 1.5, 1.5, 0, 55, 2.5), "L209": (45, 0.05, 0.9, 1.5, 0.25, 55, 2.5),
    "L210": (0, 0.05, 0.9, 1.5, 0.25, 85, 20), "L211": (0, 0.05, 1.5, 1.5, 0, 40, 50),
    "L212": (20, 0.05, 0.75, 1.5, 0.25, 40, 50), "L213": (30, 0.05, 1.5, 1.5, 0, 70, 50),
    "L214": (45, 0.05, 1.5, 1.5, 0, 70, 50), "L215": (0, 0.05, 0.3, 1.5, 0.25, 55, 80),
    "L216": (45, 0.05, 0.3, 1.5, 0.25, 55, 80), "L217": (20, 0.5, 1.5, 1.5, 0, 25, 2.5),
    "L218": (45, 0.5, 0.75, 1.5, 0.25, 25, 2.5), "L219": (45, 0.5, 0.3, 1.5, 0.25, 55, 2.5),
    "L220": (20, 0.5, 0.75, 1.5, 0.25, 85, 2.5), "L221": (45, 0.5, 1.5, 1.5, 0, 85, 2.5),
    "L222": (45, 0.5, 1.5, 1.5, 0, 40, 50), "L223": (20, 0.5, 0.3, 1.5, 0.25, 70, 50),
    "L224": (20, 0.5, 1.5, 1.5, 0, 55, 80), "L225": (45, 0.5, 0.75, 1.5, 0.25, 55, 80),
    "V206": (20, 0.2, 1, 1.4, 0.03, 25, 15),
}
PARAM_NAMES = ["lp_T_set", "lp_CC", "lp_ADC", "lp_PDC", "lp_F", "lp_SoC", "lp_dSoC"]

TEMP_SENTINEL = -273.0


def _decode(x):
    return x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else x


def read_info(f: h5py.File) -> pd.DataFrame:
    info = pd.DataFrame(f["info"][:])
    for c in ["idcell", "expname", "schedulefilename"]:
        info[c] = info[c].map(_decode)
    info["group"] = info["idtest"].map(lambda i: f"test_{i:04d}")
    info = info[info["group"].isin(f["tests"].keys())]
    return info.sort_values("startdatetime").reset_index(drop=True)


def load_test(g: h5py.Group) -> dict[str, np.ndarray]:
    d = {k: g[k][:] for k in ["voltage", "current", "temperature", "test_time",
                              "chargecap", "dischargecap", "stepindex"]}
    d["temperature"] = d["temperature"].astype(float)
    d["temperature"][d["temperature"] < TEMP_SENTINEL] = np.nan
    return d


def step_blocks(d) -> pd.DataFrame:
    """Contiguous runs of the same stepindex with summary stats."""
    s = d["stepindex"]
    edges = np.r_[0, np.flatnonzero(np.diff(s) != 0) + 1, len(s)]
    a, b = edges[:-1], edges[1:] - 1
    t, I, V = d["test_time"], d["current"], d["voltage"]
    csI = np.r_[0, np.cumsum(I, dtype=float)]
    return pd.DataFrame({
        "a": a, "b": b, "n": b - a + 1, "step": s[a],
        "dur": t[b] - t[a], "I": (csI[b + 1] - csI[a]) / (b - a + 1),
        "V0": V[a], "V1": V[b],
    })


# ----------------------------------------------------------------------------- RTP features
def rtp_features(d) -> dict:
    blk = step_blocks(d)
    t, I, V, T = d["test_time"], d["current"], d["voltage"], d["temperature"]
    dq, cq = d["dischargecap"], d["chargecap"]
    out: dict = {}

    # --- 1C full discharge to 2.5 V (target) ---------------------------------------------
    full = blk[(blk.I.between(-1.1 * C_NOM, -0.9 * C_NOM)) & (blk.V1 < 2.6) & (blk.dur > 900)]
    if full.empty:
        return {}
    caps = [dq[r.b] - dq[r.a] for r in full.itertuples()]
    out["cap_1C_Ah"] = float(np.mean(caps))
    out["cap_1C_spread_Ah"] = float(np.ptp(caps))
    r = full.iloc[-1]                       # last full discharge = best-conditioned one
    a, b = int(r.a), int(r.b)
    seg = slice(a, b + 1)
    Vd, Qd, Td = V[seg].astype(float), dq[seg] - dq[a], T[seg]
    out["dis1C_energy_Wh"] = float(np.trapezoid(Vd, Qd))  # leaky (≈ capacity * mean V)
    out["dis1C_meanV"] = out["dis1C_energy_Wh"] / max(Qd[-1], 1e-9)
    out["dis1C_dT"] = float(np.nanmax(Td) - Td[0]) if np.isfinite(Td[0]) else np.nan
    out["dis1C_Tstart"] = float(Td[0])
    out["dis1C_IRdrop_ohm"] = float((V[a - 1] - V[a]) / abs(I[a])) if a > 0 else np.nan
    # partial-window capacity: charge delivered between 3.8 V and 3.5 V (BMS-observable)
    out["dis1C_Q_3.8_3.5V"] = _q_window(Vd, Qd, 3.8, 3.5)
    out["dis1C_t_to_3.6V_s"] = _time_to(t[seg] - t[a], Vd, 3.6)
    # relaxation after the discharge (next block should be a rest)
    nxt = blk[(blk.a == b + 1)]
    if not nxt.empty and abs(nxt.iloc[0].I) < 0.01:
        n = nxt.iloc[0]
        out["relax_dV_after_dis"] = float(V[int(n.b)] - V[int(n.a)])

    # --- incremental capacity (dQ/dV) on the 1C discharge ---------------------------------
    out.update(_ica(Vd, Qd))

    # --- C/2 CC-CV charge to 4.2 V that precedes the last full discharge -----------------
    cc = blk[(blk.I.between(0.45 * C_NOM, 0.55 * C_NOM)) & (blk.V1 > 4.18) & (blk.b < a)]
    if not cc.empty:
        c = cc.iloc[-1]
        out["chg_CC_time_s"] = float(c.dur)
        out["chg_CC_Ah"] = float(cq[int(c.b)] - cq[int(c.a)])
        out["chg_CC_V0"] = float(c.V0)
        cv = blk[(blk.a > c.b) & (blk.b < a) & (blk.I > 0.005)]
        cv_ah = float(sum(cq[int(x.b)] - cq[int(x.a)] for x in cv.itertuples()))
        out["chg_CV_Ah"] = cv_ah
        out["chg_CV_time_s"] = float(cv.dur.sum())
        tot = out["chg_CC_Ah"] + cv_ah
        out["chg_CC_fraction"] = out["chg_CC_Ah"] / tot if tot > 0 else np.nan

    # --- C/3 partial discharge 4.11 -> 3.498 V (phase b) ---------------------------------
    c3 = blk[(blk.I.between(-0.38 * C_NOM, -0.28 * C_NOM)) & (blk.V1.between(3.45, 3.55))
             & (blk.dur > 1000)]
    if not c3.empty:
        x = c3.iloc[-1]
        out["dis_C3_CC_Ah"] = float(dq[int(x.b)] - dq[int(x.a)])

    # --- DC resistance from 10 s discharge pulses after >=800 s rest (phase d) -----------
    # Pulse amplitudes are C-rates relative to the cell's *latest measured* capacity, so they
    # shrink as the cell ages (e.g. 3.78 A -> 3.47 A). Classify by ratio to the largest pulse.
    pulses = []
    for p in blk[(blk.I < -0.3) & (blk.dur.between(8, 12))].itertuples():
        prev = blk[blk.b == p.a - 1]
        if prev.empty or abs(prev.iloc[0].I) > 0.01 or prev.iloc[0].dur < 800:
            continue
        amp = float(-I[p.a:p.b + 1].mean())
        pulses.append((amp, V[p.a - 1], V[p.a], V[p.b]))
    amax = max((p[0] for p in pulses), default=np.nan)
    classes = [("R_1p35C", 0.95, 1.05), ("R_1p13C", 0.75, 0.92), ("R_0p18C", 0.08, 0.2)]
    for tag, lo, hi in classes:
        sel = [p for p in pulses if lo <= p[0] / amax <= hi]
        rs_inst = [(vr - v0) / amp for amp, vr, v0, _ in sel]
        rs_10 = [(vr - v1) / amp for amp, vr, _, v1 in sel]
        if rs_10:
            out[f"{tag}_10s_mean"] = float(np.mean(rs_10))
            out[f"{tag}_inst_mean"] = float(np.mean(rs_inst))
            out[f"{tag}_n"] = len(rs_10)
            if len(rs_10) >= 4:      # 85,75,65,55,... % SOC -> index 3 ~ 55 %
                out[f"{tag}_10s_soc55"] = float(rs_10[3])
            out[f"{tag}_10s_lowSOC"] = float(rs_10[-1])

    out["rtp_T_mean"] = float(np.nanmean(T))
    return out


def _q_window(V, Q, v_hi, v_lo):
    if V.max() < v_hi or V.min() > v_lo:
        return np.nan
    i_hi = np.argmax(V <= v_hi)
    i_lo = np.argmax(V <= v_lo)
    return float(Q[i_lo] - Q[i_hi])


def _time_to(t, V, v):
    idx = np.flatnonzero(V <= v)
    return float(t[idx[0]]) if idx.size else np.nan


def _ica(V, Q, dv=0.01, lo=3.0, hi=4.15):
    """dQ/dV on a uniform voltage grid (discharge curve is monotone after smoothing)."""
    order = np.argsort(V)
    Vs, Qs = V[order], Q[order]
    Vs, keep = np.unique(Vs, return_index=True)
    Qs = Qs[keep]
    if len(Vs) < 20:
        return {}
    grid = np.arange(lo, hi, dv)
    Qg = np.interp(grid, Vs, Qs)
    ic = -np.gradient(Qg, dv)               # discharge: Q rises as V falls
    k = np.ones(5) / 5
    ic = np.convolve(ic, k, mode="same")
    out = {"ica_area_3.4_3.8V": float(np.interp(3.4, Vs, Qs) - np.interp(3.8, Vs, Qs))}
    # NCA 1C discharge shows three peaks: ~3.3 V, ~3.6 V, ~3.85 V
    for name, (wlo, whi) in {"p1": (3.15, 3.45), "p2": (3.5, 3.72), "p3": (3.75, 3.98)}.items():
        m = (grid >= wlo) & (grid <= whi)
        j = int(np.argmax(ic[m]))
        out[f"ica_{name}_height"] = float(ic[m][j])
        out[f"ica_{name}_V"] = float(grid[m][j])
    return out


# ----------------------------------------------------------------------------- LP features
def lp_features(d) -> dict:
    I, T, V, t = d["current"], d["temperature"], d["voltage"], d["test_time"]
    dq = np.diff(d["dischargecap"].astype(float))
    cq = np.diff(d["chargecap"].astype(float))
    dis_ah = float(dq[dq > 0].sum())
    chg_ah = float(cq[cq > 0].sum())
    dt = np.diff(t)
    dt = np.where((dt > 0) & (dt < 3600), dt, 0)
    w = dt / max(dt.sum(), 1e-9)          # time weights (logging is irregular)
    Ii, Vi, Ti = I[1:], V[1:], T[1:]
    dis = Ii < -0.01
    chg = Ii > 0.01
    return {
        "lp_days": float(dt.sum() / 86400),
        "lp_dis_Ah": dis_ah, "lp_chg_Ah": chg_ah,
        "lp_T_mean": float(np.nansum(w * np.nan_to_num(Ti))),
        "lp_T_max": float(np.nanmax(T)) if np.isfinite(T).any() else np.nan,
        "lp_V_mean": float(np.sum(w * Vi)),
        "lp_V_max": float(V.max()), "lp_V_min": float(V.min()),
        "lp_I_dis_mean": float(-np.sum(w[dis] * Ii[dis]) / max(w[dis].sum(), 1e-9)) if dis.any() else 0.0,
        "lp_I_chg_mean": float(np.sum(w[chg] * Ii[chg]) / max(w[chg].sum(), 1e-9)) if chg.any() else 0.0,
        "lp_I_dis_peak": float(-I.min()) if I.min() < 0 else 0.0,
        "lp_rest_frac": float(w[np.abs(Ii) <= 0.01].sum()),
    }


# ----------------------------------------------------------------------------- per cell
def process_cell(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    rows = []
    with h5py.File(path, "r") as f:
        info = read_info(f)
        cell = info.idcell.iloc[0]
        profile = next((e for e in info.expname if not e.startswith(("RTP", "SetSoC"))), None)
        cum = dict(cum_dis_Ah=0.0, cum_chg_Ah=0.0, cum_lp_days=0.0, cum_lp_Tdays=0.0)
        last_lp: dict = {}
        t0 = info.startdatetime.iloc[0]
        for m in info.itertuples():
            d = load_test(f["tests"][m.group])
            if m.expname.startswith("RTP"):
                feats = rtp_features(d)
                if not feats:
                    continue
                row = dict(cell_id=cell, profile=profile, idtest=m.idtest, expname=m.expname,
                           rtp_T_set=int(m.expname[3:]), schedule=m.schedulefilename,
                           days_since_start=m.startdatetime - t0, **cum,
                           cum_EFC=(cum["cum_dis_Ah"] + cum["cum_chg_Ah"]) / (2 * C_NOM),
                           cum_lp_T_mean=(cum["cum_lp_Tdays"] / cum["cum_lp_days"]
                                          if cum["cum_lp_days"] > 0 else np.nan),
                           **{f"prev_{k}": v for k, v in last_lp.items()}, **feats)
                rows.append(row)
            else:
                # SetSoC tests also move charge; count them in throughput but not as LP
                lf = lp_features(d)
                cum["cum_dis_Ah"] += lf["lp_dis_Ah"]
                cum["cum_chg_Ah"] += lf["lp_chg_Ah"]
                if not m.expname.startswith("SetSoC"):
                    cum["cum_lp_days"] += lf["lp_days"]
                    cum["cum_lp_Tdays"] += lf["lp_T_mean"] * lf["lp_days"]
                    last_lp = lf
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    params = PROFILE_PARAMS.get(profile, (np.nan,) * 7)
    for k, v in zip(PARAM_NAMES, params):
        df[k] = v
    ref = df.loc[df.rtp_T_set == 25, "cap_1C_Ah"]
    df["cap_ref_Ah"] = ref.iloc[0] if len(ref) else np.nan
    df["SOH"] = 100 * df.cap_1C_Ah / df.cap_ref_Ah
    df.loc[df.rtp_T_set != 25, "SOH"] = np.nan   # capacity at other temps is not comparable
    df["rtp_index"] = (df.rtp_T_set == 25).cumsum()
    return df


def main(raw_dir="data/raw", out="results/rtp_features.csv", workers=4):
    files = sorted(Path(raw_dir).glob("*.h5"))
    with ProcessPoolExecutor(int(workers)) as ex:
        dfs = list(ex.map(process_cell, files))
    for p, d in zip(files, dfs):
        print(f"{p.name}: {len(d)} RTP rows")
    df = pd.concat(dfs, ignore_index=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"saved {out}: {df.shape}")


if __name__ == "__main__":
    main(*sys.argv[1:])
