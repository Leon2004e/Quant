# -*- coding: utf-8 -*-
"""
4_Regime_Builder/Trend_Achse/filter_Analyse.py
(UPDATED: folder-based KPI + variant-aware + TF/variant filters + UNIQUE SYMBOL selection)

Problem-Fix:
- Vorher wurden pro TF immer die ersten Zeilen gewählt -> häufig immer EURUSD.
- Jetzt wird pro TF explizit auf UNIQUE symbols reduziert (drop_duplicates(["tf","symbol"])).

Input:
  <ROOT>/1_Data_Center/Data/Regime/Filter/Trend/passed.csv
  <ROOT>/1_Data_Center/Data/Regime/Filter/Trend/failed.csv
  optional: selected.csv

Overlay sources (variant-aware):
  States base:
    <ROOT>/1_Data_Center/Data/Regime/States/Trend/<TF>/<SYMBOL>.csv
  States variants:
    <ROOT>/1_Data_Center/Data/Regime/States/Trend/Variants/<variant_id>/<TF>/<SYMBOL>.csv
  OHLC:
    <ROOT>/1_Data_Center/Data/Regime/Ohcl/<TF>/<SYMBOL>.csv

Output:
  <ROOT>/1_Data_Center/Data/Regime/Filter/Trend/Analyse/
    pass_rate_by_tf.png
    pass_rate_by_variant.png
    top_failed_rules.png
    top_failed_timeframes.png
    top_failed_variants.png
    overlays/<variant_id>/<TF>/<SYMBOL>__IS.png
    overlays/<variant_id>/<TF>/<SYMBOL>__OOS.png
    overlays/<variant_id>/<TF>/<SYMBOL>__FULL.png (fallback)
    summary.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# ROOT / PATHS
# =========================

def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "1_Data_Center").exists():
            return p
    return start.resolve().parents[1]


ROOT = find_project_root(Path(__file__))

FILTER_DIR = ROOT / "1_Data_Center" / "Data" / "Regime" / "Filter" / "Trend"
PASSED_PATH = FILTER_DIR / "passed.csv"
FAILED_PATH = FILTER_DIR / "failed.csv"
SELECTED_PATH = FILTER_DIR / "selected.csv"

STATES_BASE_DIR = ROOT / "1_Data_Center" / "Data" / "Regime" / "States" / "Trend"
STATES_VAR_DIR = STATES_BASE_DIR / "Variants"
OHLC_DIR = ROOT / "1_Data_Center" / "Data" / "Regime" / "Ohcl"

ANALYSE_DIR = FILTER_DIR / "Analyse"
OVERLAY_DIR = ANALYSE_DIR / "overlays"
SUMMARY_PATH = ANALYSE_DIR / "summary.json"


# =========================
# TIMEFRAME ORDER + FILTER
# =========================

TIMEFRAMES_PREFERRED = ["M1", "M5", "M15", "H1", "H4", "H8", "H12", "D1", "W1", "MN1", "Q", "Y"]


def filter_tfs_min(tf_list: List[str], min_tf: str = "H4") -> List[str]:
    order = {tf: i for i, tf in enumerate(TIMEFRAMES_PREFERRED)}
    if min_tf not in order:
        return [tf for tf in tf_list if tf in order]
    min_i = order[min_tf]
    return [tf for tf in tf_list if tf in order and order[tf] >= min_i]


# =========================
# Helpers
# =========================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_time_utc(series: pd.Series) -> pd.Series:
    """
    Tries ISO8601 fast-path (pandas >=2.0 supports format="ISO8601"),
    falls back otherwise.
    """
    try:
        return pd.to_datetime(series, utc=True, errors="coerce", format="ISO8601")
    except TypeError:
        return pd.to_datetime(series, utc=True, errors="coerce")


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df if not df.empty else pd.DataFrame()


def _parse_split_time(row: pd.Series) -> Optional[pd.Timestamp]:
    if "split_time_utc" not in row.index:
        return None
    v = row.get("split_time_utc", None)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    if not s:
        return None
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts


def _variant_to_state_path(variant_id: str, tf: str, symbol: str) -> Path:
    vid = str(variant_id).strip()
    if vid.lower() == "base":
        return STATES_BASE_DIR / tf / f"{symbol}.csv"
    return STATES_VAR_DIR / vid / tf / f"{symbol}.csv"


def read_states(variant_id: str, tf: str, symbol: str) -> pd.DataFrame:
    p = _variant_to_state_path(variant_id, tf, symbol)
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p)
    if df.empty or "time" not in df.columns or "trend_state" not in df.columns:
        return pd.DataFrame()

    df["time"] = parse_time_utc(df["time"])
    df["trend_state"] = pd.to_numeric(df["trend_state"], errors="coerce")

    df = (
        df.dropna(subset=["time", "trend_state"])
          .sort_values("time")
          .drop_duplicates("time", keep="last")
          .reset_index(drop=True)
    )
    df["trend_state"] = df["trend_state"].astype(int).clip(-1, 1)
    return df


def read_ohlc(tf: str, symbol: str) -> pd.DataFrame:
    p = OHLC_DIR / tf / f"{symbol}.csv"
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p)
    if df.empty or "time" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()

    df["time"] = parse_time_utc(df["time"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = (
        df.dropna(subset=["time", "close"])
          .sort_values("time")
          .drop_duplicates("time", keep="last")
          .reset_index(drop=True)
    )
    return df


def merge_states_ohlc(states: pd.DataFrame, ohlc: pd.DataFrame) -> pd.DataFrame:
    if states.empty or ohlc.empty:
        return pd.DataFrame()
    m = pd.merge(states[["time", "trend_state"]], ohlc[["time", "close"]], on="time", how="inner")
    m = m.dropna(subset=["time", "trend_state", "close"]).sort_values("time").reset_index(drop=True)
    m["trend_state"] = m["trend_state"].astype(int).clip(-1, 1)
    return m


def count_failed_rules(failed_df: pd.DataFrame) -> pd.Series:
    if failed_df.empty or "failed_rules" not in failed_df.columns:
        return pd.Series(dtype=int)

    rules: List[str] = []
    for x in failed_df["failed_rules"].fillna("").astype(str).tolist():
        if not x.strip():
            continue
        parts = [p.strip() for p in x.split("|") if p.strip()]
        rules.extend(parts)

    if not rules:
        return pd.Series(dtype=int)
    return pd.Series(rules).value_counts()


def split_is_oos(df: pd.DataFrame, split_time: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df
    t = parse_time_utc(df["time"])
    is_df = df.loc[t <= split_time].copy()
    oos_df = df.loc[t > split_time].copy()
    return is_df, oos_df


def _col_or_fallback(df: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
    if primary in df.columns:
        return df[primary]
    if fallback in df.columns:
        return df[fallback]
    return pd.Series(dtype=str)


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "tf" not in out.columns and "timeframe" in out.columns:
        out["tf"] = out["timeframe"]
    if "variant_id" not in out.columns and "variant" in out.columns:
        out["variant_id"] = out["variant"]
    return out


# =========================
# Plotting
# =========================

def plot_pass_rate_by_tf(passed_df: pd.DataFrame, failed_df: pd.DataFrame, outpath: Path) -> Dict[str, float]:
    ensure_dir(outpath.parent)

    tf_p = _col_or_fallback(passed_df, "tf", "timeframe").astype(str) if not passed_df.empty else pd.Series(dtype=str)
    tf_f = _col_or_fallback(failed_df, "tf", "timeframe").astype(str) if not failed_df.empty else pd.Series(dtype=str)

    frames = sorted(set(tf_p.tolist() + tf_f.tolist()))
    frames = [x for x in frames if x and x.lower() != "nan"]
    if not frames:
        return {}

    rates: Dict[str, float] = {}
    for tf in frames:
        n_pass = int((tf_p == tf).sum()) if len(tf_p) else 0
        n_fail = int((tf_f == tf).sum()) if len(tf_f) else 0
        n = n_pass + n_fail
        rates[tf] = float(n_pass / n) if n > 0 else np.nan

    plt.figure(figsize=(12, 5))
    plt.bar(list(rates.keys()), list(rates.values()))
    plt.ylim(0, 1)
    plt.title("Pass-Rate pro Timeframe")
    plt.xlabel("Timeframe")
    plt.ylabel("Pass-Rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

    return rates


def plot_pass_rate_by_variant(passed_df: pd.DataFrame, failed_df: pd.DataFrame, outpath: Path) -> Dict[str, float]:
    ensure_dir(outpath.parent)

    v_p = _col_or_fallback(passed_df, "variant_id", "variant").astype(str) if not passed_df.empty else pd.Series(dtype=str)
    v_f = _col_or_fallback(failed_df, "variant_id", "variant").astype(str) if not failed_df.empty else pd.Series(dtype=str)

    vars_ = sorted(set(v_p.tolist() + v_f.tolist()))
    vars_ = [x for x in vars_ if x and x.lower() != "nan"]
    if not vars_:
        return {}

    rates: Dict[str, float] = {}
    for v in vars_:
        n_pass = int((v_p == v).sum()) if len(v_p) else 0
        n_fail = int((v_f == v).sum()) if len(v_f) else 0
        n = n_pass + n_fail
        rates[v] = float(n_pass / n) if n > 0 else np.nan

    plt.figure(figsize=(12, 5))
    plt.bar(list(rates.keys()), list(rates.values()))
    plt.ylim(0, 1)
    plt.title("Pass-Rate pro Variant")
    plt.xlabel("Variant")
    plt.ylabel("Pass-Rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

    return rates


def plot_top_failed_rules(failed_df: pd.DataFrame, outpath: Path, top_n: int = 20) -> Dict[str, int]:
    ensure_dir(outpath.parent)
    vc = count_failed_rules(failed_df)
    if vc.empty:
        return {}

    vc_top = vc.head(int(top_n))
    plt.figure(figsize=(12, 7))
    plt.barh(vc_top.index[::-1], vc_top.values[::-1])
    plt.title(f"Top {top_n} Fail-Regeln (gesamt)")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    return {str(k): int(v) for k, v in vc_top.to_dict().items()}


def plot_top_failed_timeframes(failed_df: pd.DataFrame, outpath: Path, top_n: int = 20) -> Dict[str, int]:
    ensure_dir(outpath.parent)
    tf = _col_or_fallback(failed_df, "tf", "timeframe")
    if failed_df.empty or tf.empty:
        return {}
    vc = tf.astype(str).value_counts().head(int(top_n))
    if vc.empty:
        return {}
    plt.figure(figsize=(12, 7))
    plt.barh(vc.index[::-1], vc.values[::-1])
    plt.title(f"Top {top_n} Timeframes nach Failed-Count")
    plt.xlabel("Failed Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    return {str(k): int(v) for k, v in vc.to_dict().items()}


def plot_top_failed_variants(failed_df: pd.DataFrame, outpath: Path, top_n: int = 20) -> Dict[str, int]:
    ensure_dir(outpath.parent)
    v = _col_or_fallback(failed_df, "variant_id", "variant")
    if failed_df.empty or v.empty:
        return {}
    vc = v.astype(str).value_counts().head(int(top_n))
    if vc.empty:
        return {}
    plt.figure(figsize=(12, 7))
    plt.barh(vc.index[::-1], vc.values[::-1])
    plt.title(f"Top {top_n} Variants nach Failed-Count")
    plt.xlabel("Failed Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    return {str(k): int(v) for k, v in vc.to_dict().items()}


def plot_state_overlay(
    variant_id: str,
    tf: str,
    symbol: str,
    merged: pd.DataFrame,
    outpath: Path,
    title_suffix: str = "",
    max_points: int = 6000,
    mark_switches: bool = True,
) -> bool:
    if merged.empty:
        return False

    ensure_dir(outpath.parent)

    df = merged.copy()
    if len(df) > int(max_points):
        df = df.iloc[-int(max_points):].copy()

    t = df["time"].to_numpy()
    close = df["close"].to_numpy(dtype=float)
    st = df["trend_state"].to_numpy(dtype=int)

    plt.figure(figsize=(14, 6))
    plt.plot(t, close, linewidth=1.0)

    def state_alpha(s: int) -> float:
        return 0.10 if s == 0 else 0.14

    def state_color(s: int) -> str:
        if s == 1:
            return "#2ca02c"
        if s == -1:
            return "#d62728"
        return "#7f7f7f"

    changes = np.flatnonzero(np.diff(st) != 0) + 1
    boundaries = np.r_[0, changes, len(st)]

    for i in range(len(boundaries) - 1):
        a = int(boundaries[i])
        b = int(boundaries[i + 1])
        s = int(st[a])
        plt.axvspan(t[a], t[b - 1], alpha=state_alpha(s), color=state_color(s), linewidth=0)

    if mark_switches and len(changes) > 0:
        plt.scatter(t[changes], close[changes], s=10)

    ttl = f"Trend-Regime Overlay: {symbol} | {tf} | {variant_id}"
    if title_suffix:
        ttl += f" | {title_suffix}"
    plt.title(ttl)
    plt.xlabel("Time")
    plt.ylabel("Close")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    return True


# =========================
# Candidate selection (FIXED)
# =========================

def pick_candidates(
    passed_df: pd.DataFrame,
    failed_df: pd.DataFrame,
    mode: str,
    per_tf: int,
    per_variant: int,
    min_tf: str,
    variants_filter: Optional[List[str]],
) -> pd.DataFrame:
    """
    Returns candidate list with columns: variant_id, tf, symbol, split_time_utc.
    Fixes:
      - TF filter via --min-tf
      - optional variants filter
      - diversification per variant
      - UNIQUE SYMBOL selection per tf (core fix for "only EURUSD")
    """
    passed_df = _norm_cols(passed_df)
    failed_df = _norm_cols(failed_df)

    # filter variants
    if variants_filter:
        keep = set(map(str, variants_filter))
        if not passed_df.empty and "variant_id" in passed_df.columns:
            passed_df = passed_df[passed_df["variant_id"].astype(str).isin(keep)].copy()
        if not failed_df.empty and "variant_id" in failed_df.columns:
            failed_df = failed_df[failed_df["variant_id"].astype(str).isin(keep)].copy()

    # filter timeframes
    def _tf_filter(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "tf" not in df.columns:
            return df
        tfs = df["tf"].astype(str).unique().tolist()
        allowed = set(filter_tfs_min(tfs, min_tf=min_tf))
        return df[df["tf"].astype(str).isin(allowed)].copy()

    passed_df = _tf_filter(passed_df)
    failed_df = _tf_filter(failed_df)

    # ---------- FAILED EXAMPLES ----------
    if mode == "failed_examples":
        if failed_df.empty:
            return pd.DataFrame()
        df = failed_df.copy()

        if "rule_fail_count" in df.columns:
            df["rule_fail_count"] = pd.to_numeric(df["rule_fail_count"], errors="coerce")
        else:
            df["rule_fail_count"] = np.nan

        for c in ["symbol", "variant_id"]:
            if c not in df.columns:
                df[c] = ""

        df = df.sort_values(
            ["tf", "rule_fail_count", "symbol", "variant_id"],
            ascending=[True, False, True, True]
        )

        # diversify per variant, then UNIQUE symbols per tf, then cap per tf
        out = (
            df.groupby(["tf", "variant_id"], dropna=False)
              .head(int(per_variant))
              .reset_index(drop=True)
        )
        out = (
            out.drop_duplicates(subset=["tf", "symbol"], keep="first")
               .groupby("tf", dropna=False)
               .head(int(per_tf))
               .reset_index(drop=True)
        )
        return out

    # ---------- SELECTED (if exists) ----------
    if mode == "selected_if_exists" and SELECTED_PATH.exists():
        sel = read_csv_safe(SELECTED_PATH)
        sel = _norm_cols(sel)
        sel = _tf_filter(sel)

        if variants_filter and not sel.empty and "variant_id" in sel.columns:
            sel = sel[sel["variant_id"].astype(str).isin(set(map(str, variants_filter)))].copy()

        if not sel.empty and {"tf", "symbol", "variant_id"}.issubset(sel.columns):
            sel["symbol"] = sel["symbol"].astype(str)
            sel["variant_id"] = sel["variant_id"].astype(str)
            sel = sel.sort_values(["tf", "variant_id", "symbol"], ascending=[True, True, True])

            out = (
                sel.groupby(["tf", "variant_id"], dropna=False)
                   .head(int(per_variant))
                   .reset_index(drop=True)
            )
            out = (
                out.drop_duplicates(subset=["tf", "symbol"], keep="first")
                   .groupby("tf", dropna=False)
                   .head(int(per_tf))
                   .reset_index(drop=True)
            )
            return out

    # ---------- PASSED (default) ----------
    if passed_df.empty:
        return pd.DataFrame()

    dfp = passed_df.copy()

    for c in ["coverage_oos", "sep_mean_pos_frac_oos", "switch_rate_oos", "avg_state_duration_oos", "n_rows_oos"]:
        if c not in dfp.columns:
            dfp[c] = np.nan
        dfp[c] = pd.to_numeric(dfp[c], errors="coerce")

    for c in ["symbol", "variant_id"]:
        if c not in dfp.columns:
            dfp[c] = ""

    dfp = dfp.sort_values(
        ["tf", "coverage_oos", "sep_mean_pos_frac_oos", "avg_state_duration_oos", "switch_rate_oos", "n_rows_oos", "variant_id", "symbol"],
        ascending=[True, False, False, False, True, False, True, True]
    )

    out = (
        dfp.groupby(["tf", "variant_id"], dropna=False)
           .head(int(per_variant))
           .reset_index(drop=True)
    )
    # CORE FIX: unique symbols per tf
    out = (
        out.drop_duplicates(subset=["tf", "symbol"], keep="first")
           .groupby("tf", dropna=False)
           .head(int(per_tf))
           .reset_index(drop=True)
    )
    return out


# =========================
# Main build
# =========================

def build(
    per_tf: int,
    per_variant: int,
    mode: str,
    top_n: int,
    max_points: int,
    mark_switches: bool,
    do_is_oos_overlays: bool,
    min_tf: str,
    variants_filter: Optional[List[str]],
) -> Dict[str, object]:
    ensure_dir(ANALYSE_DIR)
    ensure_dir(OVERLAY_DIR)

    passed_df = read_csv_safe(PASSED_PATH)
    failed_df = read_csv_safe(FAILED_PATH)

    pass_rates_tf = plot_pass_rate_by_tf(passed_df, failed_df, ANALYSE_DIR / "pass_rate_by_tf.png")
    pass_rates_variant = plot_pass_rate_by_variant(passed_df, failed_df, ANALYSE_DIR / "pass_rate_by_variant.png")
    top_failed_rules = plot_top_failed_rules(failed_df, ANALYSE_DIR / "top_failed_rules.png", top_n=int(top_n))
    top_failed_tfs = plot_top_failed_timeframes(failed_df, ANALYSE_DIR / "top_failed_timeframes.png", top_n=int(top_n))
    top_failed_vars = plot_top_failed_variants(failed_df, ANALYSE_DIR / "top_failed_variants.png", top_n=int(top_n))

    cand = pick_candidates(
        passed_df, failed_df,
        mode=str(mode),
        per_tf=int(per_tf),
        per_variant=int(per_variant),
        min_tf=str(min_tf),
        variants_filter=variants_filter,
    )

    overlays_done = []
    overlays_failed = []

    required_cols = {"tf", "symbol", "variant_id"}
    if not cand.empty and required_cols.issubset(set(cand.columns)):
        for _, r in cand.iterrows():
            tf = str(r["tf"])
            sym = str(r["symbol"])
            vid = str(r["variant_id"])
            split_time = _parse_split_time(r)

            states = read_states(vid, tf, sym)
            ohlc = read_ohlc(tf, sym)
            merged_full = merge_states_ohlc(states, ohlc)

            if merged_full.empty:
                overlays_failed.append({"variant_id": vid, "tf": tf, "symbol": sym, "reason": "missing_states_or_ohlc_or_no_overlap"})
                continue

            if do_is_oos_overlays and split_time is not None:
                is_df, oos_df = split_is_oos(merged_full, split_time)

                out_is = OVERLAY_DIR / vid / tf / f"{sym}__IS.png"
                out_oos = OVERLAY_DIR / vid / tf / f"{sym}__OOS.png"

                ok_is = plot_state_overlay(vid, tf, sym, is_df, out_is, title_suffix="IS",
                                           max_points=int(max_points), mark_switches=bool(mark_switches))
                ok_oos = plot_state_overlay(vid, tf, sym, oos_df, out_oos, title_suffix="OOS",
                                            max_points=int(max_points), mark_switches=bool(mark_switches))

                if ok_is:
                    overlays_done.append({"variant_id": vid, "tf": tf, "symbol": sym, "part": "IS", "file": str(out_is)})
                else:
                    overlays_failed.append({"variant_id": vid, "tf": tf, "symbol": sym, "part": "IS", "reason": "empty_after_split"})
                if ok_oos:
                    overlays_done.append({"variant_id": vid, "tf": tf, "symbol": sym, "part": "OOS", "file": str(out_oos)})
                else:
                    overlays_failed.append({"variant_id": vid, "tf": tf, "symbol": sym, "part": "OOS", "reason": "empty_after_split"})

            else:
                out_full = OVERLAY_DIR / vid / tf / f"{sym}__FULL.png"
                ok = plot_state_overlay(vid, tf, sym, merged_full, out_full, title_suffix="FULL",
                                        max_points=int(max_points), mark_switches=bool(mark_switches))
                if ok:
                    overlays_done.append({"variant_id": vid, "tf": tf, "symbol": sym, "part": "FULL", "file": str(out_full)})
                else:
                    overlays_failed.append({"variant_id": vid, "tf": tf, "symbol": sym, "reason": "plot_failed"})
    else:
        overlays_failed.append({
            "reason": "no_candidates_or_missing_columns",
            "need": sorted(list(required_cols)),
            "have": list(cand.columns) if not cand.empty else []
        })

    summary = {
        "meta": {
            "updated_at_utc": utc_now_str(),
            "root": str(ROOT.resolve()),
            "filter_dir": str(FILTER_DIR.resolve()),
            "analyse_dir": str(ANALYSE_DIR.resolve()),
            "states_base_dir": str(STATES_BASE_DIR.resolve()),
            "states_var_dir": str(STATES_VAR_DIR.resolve()),
            "ohlc_dir": str(OHLC_DIR.resolve()),
            "mode": mode,
            "per_tf": int(per_tf),
            "per_variant": int(per_variant),
            "min_tf": str(min_tf),
            "variants_filter": variants_filter,
            "max_points": int(max_points),
            "mark_switches": bool(mark_switches),
            "top_n": int(top_n),
            "do_is_oos_overlays": bool(do_is_oos_overlays),
        },
        "counts": {
            "passed_rows": int(len(passed_df)) if not passed_df.empty else 0,
            "failed_rows": int(len(failed_df)) if not failed_df.empty else 0,
            "candidates": int(len(cand)) if not cand.empty else 0,
            "overlays_done": int(len(overlays_done)),
            "overlays_failed": int(len(overlays_failed)),
        },
        "pass_rates_by_tf": pass_rates_tf,
        "pass_rates_by_variant": pass_rates_variant,
        "top_failed_rules": top_failed_rules,
        "top_failed_timeframes": top_failed_tfs,
        "top_failed_variants": top_failed_vars,
        "overlays_done": overlays_done,
        "overlays_failed": overlays_failed,
        "files": {
            "pass_rate_by_tf_png": str((ANALYSE_DIR / "pass_rate_by_tf.png").resolve()),
            "pass_rate_by_variant_png": str((ANALYSE_DIR / "pass_rate_by_variant.png").resolve()),
            "top_failed_rules_png": str((ANALYSE_DIR / "top_failed_rules.png").resolve()),
            "top_failed_timeframes_png": str((ANALYSE_DIR / "top_failed_timeframes.png").resolve()),
            "top_failed_variants_png": str((ANALYSE_DIR / "top_failed_variants.png").resolve()),
            "overlay_dir": str(OVERLAY_DIR.resolve()),
        }
    }

    ensure_dir(SUMMARY_PATH.parent)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tf", type=int, default=5, help="max overlays per timeframe (unique symbols)")
    ap.add_argument("--per-variant", type=int, default=3, help="pre-cap per variant before unique-symbol selection")
    ap.add_argument("--mode", type=str, default="selected_if_exists",
                    choices=["selected_if_exists", "passed_all", "failed_examples"])
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--max-points", type=int, default=6000)
    ap.add_argument("--no-switch-markers", action="store_true")
    ap.add_argument("--no-is-oos-overlays", action="store_true",
                    help="if set, write only FULL overlays (ignore split_time_utc)")
    ap.add_argument("--min-tf", type=str, default="H4", help="Analyse/Overlays only for TF >= min-tf")
    ap.add_argument("--variants", nargs="*", default=None, help="Optional: only these variant_ids")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not PASSED_PATH.exists() and not FAILED_PATH.exists():
        raise RuntimeError(f"Missing passed/failed CSV in: {FILTER_DIR.resolve()}")

    print("[INFO] ROOT       =", ROOT.resolve())
    print("[INFO] FILTER_DIR =", FILTER_DIR.resolve())
    print("[INFO] ANALYSE    =", ANALYSE_DIR.resolve())
    print("[INFO] mode       =", args.mode)
    print("[INFO] per_tf     =", int(args.per_tf))
    print("[INFO] per_variant=", int(args.per_variant))
    print("[INFO] min_tf     =", str(args.min_tf))
    print("[INFO] variants   =", args.variants)
    print("[INFO] top_n      =", int(args.top_n))
    print("[INFO] overlays   =", "IS/OOS" if not args.no_is_oos_overlays else "FULL-only")

    build(
        per_tf=int(args.per_tf),
        per_variant=int(args.per_variant),
        mode=str(args.mode),
        top_n=int(args.top_n),
        max_points=int(args.max_points),
        mark_switches=not bool(args.no_switch_markers),
        do_is_oos_overlays=not bool(args.no_is_oos_overlays),
        min_tf=str(args.min_tf),
        variants_filter=list(args.variants) if args.variants else None,
    )

    print("[DONE] summary  ->", SUMMARY_PATH.resolve())
    print("[DONE] overlays ->", OVERLAY_DIR.resolve())


if __name__ == "__main__":
    main()
