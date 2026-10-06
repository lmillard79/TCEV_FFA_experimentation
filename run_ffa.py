"""Single entry point: fit TCEV (plus GEV and Gumbel baselines) to every
inputs/FLIKE_Input_*.csv (see --pattern) and write flat-file outputs, plots and a provenance
record to outputs/<timestamp>/.

Usage:
    python run_ffa.py [--inputs inputs] [--out outputs] [--starts 30] [--seed 20260930]
"""
import argparse
import hashlib
import json
import logging
import platform
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tcev

AEPS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
GRID_AEP = np.logspace(np.log10(0.9), np.log10(0.005), 80)  # curve for plotting


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def file_record(p):
    p = Path(p)
    return {"path": str(p.resolve()),
            "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
            "sha256": sha256(p)}


def gumbel_x(aep):
    """Gumbel reduced variate of non-exceedance 1 - AEP, for an AEP (plot axis)."""
    return -np.log(-np.log(1 - np.asarray(aep, dtype=float)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, default=Path("inputs"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--pattern", default="FLIKE_Input_*.csv", help="input file glob within --inputs")
    ap.add_argument("--starts", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260930)
    a = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = a.out / stamp
    out.mkdir(parents=True)
    logging.basicConfig(filename=out / f"run_log_{stamp}.txt", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    log = logging.getLogger()

    files = sorted(a.inputs.glob(a.pattern))
    if not files:
        raise SystemExit(f"No {a.pattern} in {a.inputs}")
    log.info("Inputs: %s", [f.name for f in files])

    params_rows, quant_rows, written = [], [], []

    for f in files:
        gauge = f.stem.replace("FLIKE_Input_", "")
        d = pd.read_csv(f, index_col=0)
        # Duplicate-year check: one maximum per water year, else abort loudly.
        if d["Maximum Year"].duplicated().any():
            raise SystemExit(f"{gauge}: duplicate years in {f.name}")
        x = d["Annual Discharge"].to_numpy(float)
        n = len(x)
        log.info("%s: n=%d", gauge, n)

        # per-gauge rng: results do not depend on gauge order
        rng = np.random.default_rng(a.seed)
        par, nll_t, starts = tcev.fit(x, rng, a.starts)
        starts_df = pd.DataFrame(starts).sort_values("nll")
        starts_df.to_csv(out / f"starts_{gauge}.csv", index=False)
        best = starts_df["nll"].iloc[0]
        n_same = int((starts_df["nll"] < best + 1e-3).sum())
        flags = tcev.boundary_flags(par)
        log.info("%s TCEV: %s nll=%.3f  starts at optimum %d/%d  flags=%s",
                 gauge, dict(zip(tcev.PARAM_NAMES, np.round(par, 4))), nll_t,
                 n_same, a.starts, flags)

        gev_par, nll_g, gev_q = tcev.fit_gev(x)
        gum_par, nll_u, gum_q = tcev.fit_gumbel(x)

        # QA: GEV nests Gumbel, and TCEV approaches it as l2 -> 0, so neither
        # may have a worse NLL than Gumbel. If so the optimiser failed.
        nest_ok = bool(nll_g <= nll_u + 1e-3 and nll_t <= nll_u + 1e-3)
        if not nest_ok:
            log.warning("%s: NLL ordering violated (tcev %.3f, gev %.3f, gumbel %.3f) - "
                        "optimiser failure, do not use", gauge, nll_t, nll_g, nll_u)

        params_rows.append({
            "gauge": gauge, "n": n, **dict(zip(tcev.PARAM_NAMES, par)),
            "tcev_nll": nll_t, "tcev_aic": tcev.aic(nll_t, 4),
            "gev_xi": -gev_par[0], "gev_loc": gev_par[1], "gev_scale": gev_par[2],
            "gev_nll": nll_g, "gev_aic": tcev.aic(nll_g, 3),
            "gumbel_loc": gum_par[0], "gumbel_scale": gum_par[1],
            "gumbel_nll": nll_u, "gumbel_aic": tcev.aic(nll_u, 2),
            "starts_at_optimum": n_same, "starts_total": a.starts,
            "starts_success": int(starts_df["success"].sum()), **flags,
            "qa_nll_ordering_ok": nest_ok,
        })

        # ---- single source of truth for quantiles: computed once here ----
        def q_tcev(p):
            try:
                return tcev.quantile(p, par)
            except ValueError:
                return np.nan

        qt = pd.DataFrame({"AEP": AEPS})
        qt["gauge"] = gauge
        qt["TCEV"] = [q_tcev(p) for p in AEPS]
        qt["GEV"] = gev_q(np.array(AEPS))
        qt["Gumbel"] = gum_q(np.array(AEPS))
        quant_rows.append(qt)

        # curve data: exactly what is plotted, exported as CSV
        curve = pd.DataFrame({"AEP": GRID_AEP, "gumbel_reduced_variate": gumbel_x(GRID_AEP)})
        curve["TCEV"] = [q_tcev(p) for p in GRID_AEP]
        curve["GEV"] = gev_q(GRID_AEP)
        curve["Gumbel"] = gum_q(GRID_AEP)
        # component curves: x where each component alone has non-exceedance 1-AEP
        l1, t1, l2, t2 = par
        curve["TCEV_comp1_only"] = -t1 * np.log(-np.log(1 - GRID_AEP) / l1)
        curve["TCEV_comp2_only"] = -t2 * np.log(-np.log(1 - GRID_AEP) / l2)
        curve.to_csv(out / f"curve_{gauge}.csv", index=False)

        pts = d[["Maximum Year", "Annual Discharge", "AEP plot posn", "AEPpc"]].copy()
        pts["AEP"] = pts["AEPpc"] / 100
        pts["gumbel_reduced_variate"] = gumbel_x(pts["AEP"])
        pts.to_csv(out / f"plotdata_points_{gauge}.csv", index=False)

        # ---- plot: reads the exported tables, not a recomputation ----
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.plot(curve["gumbel_reduced_variate"], curve["TCEV"], "k-", lw=2, label="TCEV")
        ax.plot(curve["gumbel_reduced_variate"], curve["GEV"], "-", color="tab:blue", label="GEV")
        ax.plot(curve["gumbel_reduced_variate"], curve["Gumbel"], "--", color="tab:green", label="Gumbel")
        ax.plot(curve["gumbel_reduced_variate"], curve["TCEV_comp1_only"], ":", color="tab:orange",
                label="TCEV component 1 (frequent)")
        ax.plot(curve["gumbel_reduced_variate"], curve["TCEV_comp2_only"], ":", color="tab:red",
                label="TCEV component 2 (rare)")
        ax.plot(pts["gumbel_reduced_variate"], pts["Annual Discharge"], "o", mfc="none", color="grey",
                label="Observed (Cunnane)")
        ticks = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        ax.set_xticks(gumbel_x(np.array(ticks)))
        ax.set_xticklabels([f"{t*100:g}%" for t in ticks])
        ax.set_xlabel("AEP")
        ax.set_ylabel("Peak flow (m$^3$/s)")
        ymax = max(x.max() * 1.5, 1)
        ax.set_ylim(0, ymax)
        ax.set_title(f"{gauge}  (n={n}, {int(d['Maximum Year'].min())}-{int(d['Maximum Year'].max())})")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
        fig.savefig(out / f"fig_{gauge}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    pd.DataFrame(params_rows).to_csv(out / "fit_parameters.csv", index=False)
    pd.concat(quant_rows).to_csv(out / "quantiles.csv", index=False)

    written = sorted(p.name for p in out.iterdir())
    prov = {
        "run_stamp": stamp, "seed": a.seed, "starts": a.starts,
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "inputs": [file_record(f) for f in files],
        "gauge_register": file_record(a.inputs / "gauge_register.csv")
        if (a.inputs / "gauge_register.csv").exists() else None,
        "code": [file_record(Path(__file__)), file_record(Path(tcev.__file__))],
        "outputs": written,
    }
    (out / f"qa_provenance_{stamp}.json").write_text(json.dumps(prov, indent=2))
    log.info("Done. Outputs: %s", written)
    print(f"Outputs written to {out}")


if __name__ == "__main__":
    main()
