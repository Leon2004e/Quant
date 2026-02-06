# -*- coding: utf-8 -*-
"""
4_Regime_Builder/Trend_Achse/Filter_System.py  (UPDATED for folder-based KPI output)

Zweck:
  Filter-System für Trend-Regime-Achse.
  Liest KPI-Dateien aus dem neuen KPI_Messung.py Ordner-Output und wendet Regel-Checks an.

NEUER KPI-INPUT (Ordnerstruktur, keine Einzeldateien pro Symbol):
  <ROOT>/1_Data_Center/Data/Regime/KPI/Trend/Variants/<variant_id>/<TF>.csv
  -> jede Datei enthält viele Zeilen (eine pro Symbol)

Output:
  <ROOT>/1_Data_Center/Data/Regime/Filter/Trend/
      passed.csv
      failed.csv
      selected.csv (optional bei --top-k > 0)
      summary.json

Wichtig:
- Dieses Filter-System arbeitet mit den KPI-Spalten aus dem von mir gelieferten KPI_Messung-Script:
    coverage_all, coverage_is, coverage_oos
    balance_*_{all|is|oos}, switch_rate_{all|is|oos}, avg_state_duration_{all|is|oos}
    n_rows_{all|is|oos}
    n_bull_h{h}_{all|is|oos}, n_bear_h{h}_{...}, n_neutral_h{h}_{...}
    sep_mean_bull_bear_h{h}_{...}
    boot_pneg_sep_mean_h{h}_{...}  (falls bootstrapping aktiviert)

- Falls IS/OOS Split nicht aktiv ist (is_frac invalid), fehlen *_is/*_oos.
  Dann greifen OOS-Regeln nicht -> Datei fällt durch (konservativ). Du kannst mit --allow-no-oos lockern.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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

# NEW: KPI folder layout
KPI_DIR = ROOT / "1_Data_Center" / "Data" / "Regime" / "KPI" / "Trend" / "Variants"

FILTER_DIR = ROOT / "1_Data_Center" / "Data" / "Regime" / "Filter" / "Trend"
PASSED_PATH = FILTER_DIR / "passed.csv"
FAILED_PATH = FILTER_DIR / "failed.csv"
SELECTED_PATH = FILTER_DIR / "selected.csv"
SUMMARY_PATH = FILTER_DIR / "summary.json"


# =========================
# SETTINGS
# =========================

TIMEFRAMES_PREFERRED = ["M1", "M5", "M15", "H1", "H4", "H8", "H12", "D1", "W1", "MN1", "Q", "Y"]
DEFAULT_HORIZONS = [1, 5, 20, 50]


# =========================
# IO atomic
# =========================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        df.to_csv(tmp, index=False)
        tmp.replace(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def atomic_write_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =========================
# Discovery (NEW layout)
# =========================

def list_variants() -> List[str]:
    if not KPI_DIR.exists():
        return []
    return sorted([p.name for p in KPI_DIR.iterdir() if p.is_dir()])


def list_timeframes_for_variant(variant_id: str) -> List[str]:
    vdir = KPI_DIR / variant_id
    if not vdir.exists():
        return []
    tfs = [p.stem for p in vdir.glob("*.csv")]  # file names are TF.csv
    # order by preferred list
    pref = [x for x in TIMEFRAMES_PREFERRED if x in tfs]
    rest = sorted([x for x in tfs if x not in TIMEFRAMES_PREFERRED])
    return pref + rest


def kpi_file_path(variant_id: str, tf: str) -> Path:
    return KPI_DIR / variant_id / f"{tf}.csv"


# =========================
# Rule engine
# =========================

@dataclass(frozen=True)
class Rule:
    name: str
    fn: Any  # (row) -> (pass, value)
    required_cols: List[str]


def _get(row: Dict[str, Any], col: str) -> float:
    v = row.get(col, np.nan)
    try:
        return float(v)
    except Exception:
        return np.nan


def _is_finite(x: float) -> bool:
    return x is not None and np.isfinite(x)


def rule_ge(col: str, thr: float, name: Optional[str] = None) -> Rule:
    nm = name or f"{col} >= {thr}"
    def _fn(row: Dict[str, Any]) -> Tuple[bool, Any]:
        v = _get(row, col)
        return (_is_finite(v) and v >= thr), v
    return Rule(nm, _fn, [col])


def rule_le(col: str, thr: float, name: Optional[str] = None) -> Rule:
    nm = name or f"{col} <= {thr}"
    def _fn(row: Dict[str, Any]) -> Tuple[bool, Any]:
        v = _get(row, col)
        return (_is_finite(v) and v <= thr), v
    return Rule(nm, _fn, [col])


def rule_between(col: str, lo: float, hi: float, name: Optional[str] = None) -> Rule:
    nm = name or f"{lo} <= {col} <= {hi}"
    def _fn(row: Dict[str, Any]) -> Tuple[bool, Any]:
        v = _get(row, col)
        return (_is_finite(v) and (lo <= v <= hi)), v
    return Rule(nm, _fn, [col])


def rule_bootstrap_pneg_max_nanpass(horizons: List[int], pneg_max: float, suffix: str) -> List[Rule]:
    """
    KPI_Messung output column:
      boot_pneg_sep_mean_h{h}_{suffix}
    NaN => PASS (nicht beurteilbar).
    """
    rules: List[Rule] = []
    for h in horizons:
        col = f"boot_pneg_sep_mean_h{h}_{suffix}"
        nm = f"{col} <= {pneg_max} OR NaN"
        def _fn_factory(c: str):
            def _fn(row: Dict[str, Any]) -> Tuple[bool, Any]:
                v = _get(row, c)
                if not np.isfinite(v):
                    return True, np.nan
                return (v <= pneg_max), v
            return _fn
        rules.append(Rule(nm, _fn_factory(col), [col]))
    return rules


def rule_horizon_min_samples(horizons: List[int], n_bull: int, n_bear: int, n_neutral: int, suffix: str) -> List[Rule]:
    rules: List[Rule] = []
    for h in horizons:
        rules.append(rule_ge(f"n_bull_h{h}_{suffix}", float(n_bull), name=f"n_bull_h{h}_{suffix} >= {n_bull}"))
        rules.append(rule_ge(f"n_bear_h{h}_{suffix}", float(n_bear), name=f"n_bear_h{h}_{suffix} >= {n_bear}"))
        rules.append(rule_ge(f"n_neutral_h{h}_{suffix}", float(n_neutral), name=f"n_neutral_h{h}_{suffix} >= {n_neutral}"))
    return rules


def rule_sep_mean_positive(horizons: List[int], min_frac_pos: float, suffix: str) -> Rule:
    """
    Monotonicity-Ersatz: Anteil der Horizonte mit sep_mean(bull-bear) > 0.
      sep_mean_bull_bear_h{h}_{suffix}
    """
    nm = f"sep_mean_pos_frac_{suffix} >= {min_frac_pos}"
    req = [f"sep_mean_bull_bear_h{h}_{suffix}" for h in horizons]

    def _fn(row: Dict[str, Any]) -> Tuple[bool, Any]:
        vals = []
        for h in horizons:
            vals.append(_get(row, f"sep_mean_bull_bear_h{h}_{suffix}"))
        vals = np.array(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return False, np.nan
        frac = float(np.mean(vals > 0.0))
        return (frac >= float(min_frac_pos)), frac

    return Rule(nm, _fn, req)


def rule_has_oos(allow_no_oos: bool) -> Rule:
    """
    Wenn OOS fehlt, konservativ FAIL (allow_no_oos=False).
    Bei allow_no_oos=True => PASS.
    """
    nm = "has_oos_metrics"
    def _fn(row: Dict[str, Any]) -> Tuple[bool, Any]:
        # we consider OOS present if n_rows_oos exists and is finite
        v = _get(row, "n_rows_oos")
        if allow_no_oos:
            return True, v
        return (_is_finite(v) and v > 0), v
    return Rule(nm, _fn, ["n_rows_oos"])


# =========================
# Default ruleset (OOS-first, adapted to KPI_Messung columns)
# =========================

def default_rules(
    horizons: List[int],
    oos_min_rows: int,
    require_is_sanity: bool,
    allow_no_oos: bool,
) -> List[Rule]:
    rules: List[Rule] = []

    # must have OOS (unless allowed)
    rules += [rule_has_oos(allow_no_oos=allow_no_oos)]

    # ---------- Data quality (OOS) ----------
    rules += [
        rule_ge("coverage_oos", 0.98),
        rule_ge("n_rows_oos", float(oos_min_rows)),
    ]

    # ---------- Balance (OOS) ----------
    rules += [
        rule_ge("balance_neutral_oos", 0.05),
        rule_between("balance_bull_oos", 0.15, 0.80),
        rule_between("balance_bear_oos", 0.15, 0.80),
    ]

    # ---------- Stability / turnover (OOS) ----------
    rules += [
        rule_le("switch_rate_oos", 0.05),
        rule_ge("avg_state_duration_oos", 10.0),
    ]

    # ---------- Sample adequacy per horizon (OOS) ----------
    rules += rule_horizon_min_samples(horizons, n_bull=300, n_bear=300, n_neutral=100, suffix="oos")

    # ---------- Separation sanity across horizons (OOS) ----------
    rules += [
        rule_sep_mean_positive(horizons, min_frac_pos=0.50, suffix="oos"),
    ]

    # ---------- Bootstrap uncertainty (OOS, NaN -> PASS) ----------
    rules += rule_bootstrap_pneg_max_nanpass(horizons, pneg_max=0.55, suffix="oos")

    # ---------- Optional IS sanity ----------
    if require_is_sanity:
        rules += [
            rule_ge("coverage_is", 0.98),
            rule_ge("n_rows_is", float(oos_min_rows)),
            rule_ge("balance_neutral_is", 0.05),
            rule_between("balance_bull_is", 0.15, 0.80),
            rule_between("balance_bear_is", 0.15, 0.80),
            rule_le("switch_rate_is", 0.05),
            rule_ge("avg_state_duration_is", 10.0),
            rule_sep_mean_positive(horizons, min_frac_pos=0.50, suffix="is"),
        ]
        rules += rule_horizon_min_samples(horizons, n_bull=300, n_bear=300, n_neutral=100, suffix="is")
        rules += rule_bootstrap_pneg_max_nanpass(horizons, pneg_max=0.55, suffix="is")

    return rules


# =========================
# Load KPI rows (NEW: file contains many symbols)
# =========================

def read_kpi_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    # one dict per symbol
    return df.to_dict(orient="records")


# =========================
# Apply rules
# =========================

def apply_rules(row: Dict[str, Any], rules: List[Rule]) -> Dict[str, Any]:
    results = []
    pass_count = 0
    fail_count = 0

    for r in rules:
        ok, val = r.fn(row)
        results.append({"rule": r.name, "pass": bool(ok), "value": val})
        if ok:
            pass_count += 1
        else:
            fail_count += 1

    out = dict(row)
    out["rule_pass_count"] = int(pass_count)
    out["rule_fail_count"] = int(fail_count)
    out["rule_pass_all"] = bool(fail_count == 0)

    failed_rules = [x["rule"] for x in results if not x["pass"]]
    out["failed_rules"] = "|".join(failed_rules[:50])

    return out


# =========================
# Build cycle
# =========================

def build_cycle(
    horizons: List[int],
    oos_min_rows: int,
    require_is_sanity: bool,
    allow_no_oos: bool,
    top_k: int,
    variants_filter: Optional[List[str]],
    verbose: bool = True,
) -> Dict[str, Any]:
    ensure_dir(FILTER_DIR)

    variants = list_variants()
    if not variants:
        raise RuntimeError(f"Keine Variant-Ordner gefunden unter: {KPI_DIR}")

    if variants_filter:
        keep = set(variants_filter)
        variants = [v for v in variants if v in keep]

    rules = default_rules(
        horizons=horizons,
        oos_min_rows=int(oos_min_rows),
        require_is_sanity=bool(require_is_sanity),
        allow_no_oos=bool(allow_no_oos),
    )

    passed_rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []
    counters: Dict[str, int] = {}

    for variant_id in variants:
        tfs = list_timeframes_for_variant(variant_id)
        if not tfs:
            counters["no_tfs"] = counters.get("no_tfs", 0) + 1
            continue

        for tf in tfs:
            fpath = kpi_file_path(variant_id, tf)
            try:
                rows = read_kpi_rows(fpath)
                if not rows:
                    counters["empty_kpi_file"] = counters.get("empty_kpi_file", 0) + 1
                    continue

                for row in rows:
                    row["variant_id"] = row.get("variant_id", variant_id)
                    row["tf"] = row.get("tf", tf)

                    out = apply_rules(row, rules)
                    if out["rule_pass_all"]:
                        passed_rows.append(out)
                        counters["passed"] = counters.get("passed", 0) + 1
                    else:
                        failed_rows.append(out)
                        counters["failed"] = counters.get("failed", 0) + 1

            except Exception as e:
                counters["error"] = counters.get("error", 0) + 1
                failed_rows.append({
                    "variant_id": variant_id,
                    "tf": tf,
                    "symbol": "",
                    "rule_pass_all": False,
                    "rule_pass_count": 0,
                    "rule_fail_count": len(rules),
                    "failed_rules": f"error:{str(e)}",
                })

    passed_df = pd.DataFrame(passed_rows) if passed_rows else pd.DataFrame()
    failed_df = pd.DataFrame(failed_rows) if failed_rows else pd.DataFrame()

    # Optional top_k selection (deterministic; quality-first on OOS)
    selected_df = passed_df.copy()
    if top_k > 0 and not selected_df.empty:
        # compute sort helpers if missing
        sort_cols = [
            "rule_pass_count",
            "coverage_oos",
            "sep_mean_pos_frac_oos",     # produced by rule function? not stored yet
            "switch_rate_oos",
            "avg_state_duration_oos",
            "n_rows_oos",
        ]
        # Ensure sep_mean_pos_frac_oos exists by recomputing it on dataframe if needed
        if "sep_mean_pos_frac_oos" not in selected_df.columns:
            def _sep_frac_row(r):
                vals = []
                for h in horizons:
                    vals.append(r.get(f"sep_mean_bull_bear_h{h}_oos", np.nan))
                vals = np.array(vals, dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    return np.nan
                return float(np.mean(vals > 0.0))
            selected_df["sep_mean_pos_frac_oos"] = selected_df.apply(_sep_frac_row, axis=1)

        for c in sort_cols:
            if c not in selected_df.columns:
                selected_df[c] = np.nan

        ascending = [False, False, False, True, False, False]
        selected_df = selected_df.sort_values(sort_cols, ascending=ascending).head(int(top_k))

    atomic_write_csv(passed_df, PASSED_PATH)
    atomic_write_csv(failed_df, FAILED_PATH)
    if top_k > 0:
        atomic_write_csv(selected_df, SELECTED_PATH)

    summary: Dict[str, Any] = {
        "meta": {
            "updated_at_utc": utc_now_str(),
            "root": str(ROOT.resolve()),
            "kpi_dir": str(KPI_DIR.resolve()),
            "filter_dir": str(FILTER_DIR.resolve()),
            "horizons": [int(h) for h in horizons],
            "oos_min_rows": int(oos_min_rows),
            "require_is_sanity": bool(require_is_sanity),
            "allow_no_oos": bool(allow_no_oos),
            "top_k": int(top_k),
            "variants_filter": variants_filter,
        },
        "counts": counters,
        "rules": [r.name for r in rules],
        "files": {
            "passed": str(PASSED_PATH),
            "failed": str(FAILED_PATH),
            "selected": str(SELECTED_PATH) if top_k > 0 else None,
        }
    }
    atomic_write_json(summary, SUMMARY_PATH)

    if verbose:
        print("[INFO] counts:", counters)
        print("[INFO] passed  ->", PASSED_PATH.resolve())
        print("[INFO] failed  ->", FAILED_PATH.resolve())
        if top_k > 0:
            print("[INFO] selected->", SELECTED_PATH.resolve())
        print("[INFO] summary ->", SUMMARY_PATH.resolve())

    return summary


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--horizons", type=str, default="1,5,20,50")
    ap.add_argument("--oos-min-rows", type=int, default=500)
    ap.add_argument("--require-is-sanity", action="store_true")
    ap.add_argument("--allow-no-oos", action="store_true", help="Wenn gesetzt: fehlende *_oos Metriken werden nicht hart gefailed.")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--variants", nargs="*", default=None, help="Optional: nur diese variant_ids filtern")
    return ap.parse_args()


def parse_horizons(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    hs: List[int] = []
    for p in parts:
        try:
            v = int(p)
        except Exception:
            continue
        if v > 0:
            hs.append(v)
    hs = sorted(list(set(hs)))
    return hs if hs else DEFAULT_HORIZONS


def main() -> None:
    args = parse_args()
    if not args.once:
        args.once = True

    horizons = parse_horizons(args.horizons)

    if not KPI_DIR.exists():
        raise RuntimeError(f"KPI_DIR not found: {KPI_DIR.resolve()}")

    print("[INFO] ROOT      =", ROOT.resolve())
    print("[INFO] KPI_DIR   =", KPI_DIR.resolve())
    print("[INFO] FILTERDIR =", FILTER_DIR.resolve())
    print("[INFO] horizons  =", horizons)
    print("[INFO] OOS min   =", int(args.oos_min_rows))
    print("[INFO] IS sanity =", bool(args.require_is_sanity))
    print("[INFO] allow_no_oos =", bool(args.allow_no_oos))
    print("[INFO] top_k     =", int(args.top_k))
    print("[INFO] variants  =", args.variants)

    build_cycle(
        horizons=horizons,
        oos_min_rows=int(args.oos_min_rows),
        require_is_sanity=bool(args.require_is_sanity),
        allow_no_oos=bool(args.allow_no_oos),
        top_k=int(args.top_k),
        variants_filter=list(args.variants) if args.variants else None,
        verbose=True,
    )


if __name__ == "__main__":
    main()
