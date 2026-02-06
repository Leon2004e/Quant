# -*- coding: utf-8 -*-
"""
Erzeugt 4_Regime_Builder/Trend_Achse/trend_variants.json (UTF-8, ohne BOM)

Run:
  python 4_Regime_Builder/Trend_Achse/make_trend_variants_json.py
Optional:
  python 4_Regime_Builder/Trend_Achse/make_trend_variants_json.py --out "...\trend_variants.json"
  python 4_Regime_Builder/Trend_Achse/make_trend_variants_json.py --n 60
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple


def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "1_Data_Center").exists():
            return p
    return start.resolve().parents[1]


ROOT = find_project_root(Path(__file__))


@dataclass(frozen=True)
class Variant:
    name: str
    lookback_bars: int
    smooth_span: int
    w_spread: float
    w_price: float
    w_slope: float
    t_enter: float
    t_exit: float


def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def norm_weights(a: float, b: float, c: float) -> Tuple[float, float, float]:
    s = float(a) + float(b) + float(c)
    if s <= 0:
        return (0.50, 0.40, 0.10)
    return (float(a) / s, float(b) / s, float(c) / s)


def generate_variants(n: int = 40) -> List[Variant]:
    """
    Liefert eine robuste Mischung:
      - base_like
      - fast / stable
      - spread/price/slope-heavy
      - tight/wide hysteresis
      - intraday presets
      - plus grid-sweep Kombinationen (name eindeutig)
    """
    variants: List[Variant] = []

    # 1) einige hand-curated presets
    presets = [
        Variant("base_like", 600, 10, 0.50, 0.40, 0.10, 0.80, 0.40),
        Variant("fast_1", 900, 4, 0.60, 0.35, 0.05, 0.70, 0.30),
        Variant("fast_2", 900, 6, 0.55, 0.35, 0.10, 0.75, 0.35),
        Variant("stable_1", 1200, 14, 0.50, 0.40, 0.10, 0.90, 0.45),
        Variant("stable_2", 1400, 20, 0.50, 0.40, 0.10, 1.00, 0.50),
        Variant("spread_heavy_1", 600, 10, 0.75, 0.20, 0.05, 0.85, 0.45),
        Variant("spread_heavy_2", 600, 12, 0.80, 0.15, 0.05, 0.90, 0.50),
        Variant("price_heavy_1", 600, 10, 0.30, 0.65, 0.05, 0.80, 0.40),
        Variant("price_heavy_2", 600, 12, 0.25, 0.70, 0.05, 0.85, 0.45),
        Variant("slope_heavy_1", 600, 10, 0.40, 0.25, 0.35, 0.75, 0.35),
        Variant("slope_heavy_2", 600, 14, 0.35, 0.25, 0.40, 0.80, 0.40),
        Variant("tight_hyst_1", 600, 10, 0.50, 0.40, 0.10, 0.70, 0.60),
        Variant("tight_hyst_2", 600, 10, 0.50, 0.40, 0.10, 0.75, 0.65),
        Variant("wide_hyst_1", 600, 10, 0.50, 0.40, 0.10, 1.10, 0.30),
        Variant("wide_hyst_2", 600, 12, 0.50, 0.40, 0.10, 1.20, 0.25),
        Variant("intraday_fast", 1000, 4, 0.60, 0.35, 0.05, 0.65, 0.30),
        Variant("intraday_stable", 1400, 12, 0.55, 0.35, 0.10, 0.90, 0.45),
    ]
    variants.extend(presets)

    # 2) grid sweep – sorgt für "viele" Varianten
    smooth_spans = [4, 6, 10, 14]
    enters = [0.70, 0.80, 0.90, 1.00]
    exits = [0.30, 0.40, 0.50]
    looks = [600, 900, 1200]

    weight_sets = [
        (0.50, 0.40, 0.10),
        (0.60, 0.35, 0.05),
        (0.55, 0.35, 0.10),
        (0.70, 0.25, 0.05),
        (0.30, 0.65, 0.05),
        (0.40, 0.25, 0.35),
    ]

    idx = 0
    for lb in looks:
        for sp in smooth_spans:
            for te in enters:
                for tx in exits:
                    if tx >= te:  # Hysterese muss enger sein als enter
                        continue
                    for (ws, wp, wl) in weight_sets:
                        ws, wp, wl = norm_weights(ws, wp, wl)
                        name = f"grid_{idx:03d}_lb{lb}_sp{sp}_te{te:.2f}_tx{tx:.2f}_w{ws:.2f}-{wp:.2f}-{wl:.2f}"
                        variants.append(Variant(name, lb, sp, ws, wp, wl, te, tx))
                        idx += 1
                        if len(variants) >= n:
                            # dedupe by name and return
                            seen = set()
                            out = []
                            for v in variants:
                                if v.name in seen:
                                    continue
                                seen.add(v.name)
                                out.append(v)
                            return out

    # fallback dedupe
    seen = set()
    out = []
    for v in variants:
        if v.name in seen:
            continue
        seen.add(v.name)
        out.append(v)
    return out


def validate(variants: List[Variant]) -> List[Dict]:
    out = []
    for v in variants:
        ws, wp, wl = norm_weights(v.w_spread, v.w_price, v.w_slope)

        te = float(v.t_enter)
        tx = float(v.t_exit)

        if te <= 0 or tx <= 0:
            continue
        if tx >= te:
            continue

        out.append(
            {
                "name": str(v.name).strip(),
                "lookback_bars": int(v.lookback_bars),
                "smooth_span": int(v.smooth_span),
                "w_spread": float(ws),
                "w_price": float(wp),
                "w_slope": float(wl),
                "t_enter": float(te),
                "t_exit": float(tx),
            }
        )
    # unique names
    seen = set()
    unique = []
    for d in out:
        if d["name"] in seen:
            continue
        seen.add(d["name"])
        unique.append(d)
    return unique


def main() -> None:
    ap = argparse.ArgumentParser()
    default_out = Path(__file__).resolve().with_name("trend_variants.json")
    ap.add_argument("--out", type=str, default=str(default_out))
    ap.add_argument("--n", type=int, default=80, help="Anzahl Varianten (inkl. presets + grid)")
    args = ap.parse_args()

    out_path = Path(args.out).expanduser()
    variants = generate_variants(int(args.n))
    vdicts = validate(variants)

    obj = {"variants": vdicts}

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # UTF-8 ohne BOM
    out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[DONE] wrote variants:", out_path.resolve())
    print("[INFO] n_variants:", len(vdicts))
    if len(vdicts) > 0:
        print("[INFO] first:", vdicts[0]["name"])
        print("[INFO] last :", vdicts[-1]["name"])


if __name__ == "__main__":
    main()
