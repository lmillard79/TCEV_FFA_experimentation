"""Replicate Totaro et al. (2024) Table 2 from the FFA site data.

For each of the paper's 12 sites (inputs/ffa_site/paper_sites.csv):
  1. Data check   - sample t3, t4 from the site series vs the paper's t3, t4.
                    These depend only on the data, so a match means same data.
  2. Fit check    - L-moment TCEV fit (paper's method) vs the paper's parameters.
  3. Method check - maximum-likelihood TCEV fit on the same data, for contrast.

Series used = annual maxima + censored records (the full record; the paper fits
all the data without censoring low values).

Usage: python replicate_paper.py [--dir inputs/ffa_site] [--out outputs] [--seed 20260930]
"""
import argparse
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import tcev
import tcev_lmom


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("inputs") / "ffa_site")
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--seed", type=int, default=20260930)
    a = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = a.out / f"replication_{stamp}"
    out.mkdir(parents=True)
    logging.basicConfig(filename=out / f"run_log_{stamp}.txt", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    log = logging.getLogger()

    paper = pd.read_csv(a.dir / "paper_sites.csv", dtype={"site_id": str})
    ams = pd.read_csv(a.dir / "ams_all.csv", dtype={"id": str})

    rows = []
    for _, p in paper.iterrows():
        sid = p["site_id"]
        d = ams[ams["id"] == sid]
        if d["year"].duplicated().any():
            raise SystemExit(f"{sid}: duplicate years")
        x = d["flow_m3s"].to_numpy(float)
        n = len(x)

        # 1. data check
        l1, l2, t3, t4 = tcev_lmom.sample_lmoments(x)

        # 2. L-moment fit (paper's method)
        rng = np.random.default_rng(a.seed)
        par, cost, _, starts = tcev_lmom.fit_lmom(x, rng)
        pd.DataFrame(starts).sort_values("cost").to_csv(out / f"lmom_starts_{sid}.csv", index=False)
        pl1, pt1, pl2, pt2 = par
        pop = tcev_lmom.population_lmoments(par)

        # 3. MLE for contrast
        rng = np.random.default_rng(a.seed)
        mle, nll, _ = tcev.fit(x, rng, 30)

        # the paper's own L-moment fit, scored against OUR sample L-moments: how far off
        # are the paper's parameters from the data we hold?
        ppar = (p["lambda1"], p["theta1"], p["lambda2"], p["theta2"])
        ppop = tcev_lmom.population_lmoments(ppar)

        def q(par_, aep):
            return tcev.quantile(aep, par_)

        rows.append({
            "paper_id": p["paper_id"], "site_id": sid, "name": p["paper_name"],
            "n_paper": p["paper_n_years"], "n_site": n,
            "t3_paper": p["t3"], "t3_site": t3, "t4_paper": p["t4"], "t4_site": t4,
            "lambda1_paper": p["lambda1"], "lambda1_lmom": pl1, "lambda1_mle": mle[0],
            "theta1_paper": p["theta1"], "theta1_lmom": pt1, "theta1_mle": mle[1],
            "lambda2_paper": p["lambda2"], "lambda2_lmom": pl2, "lambda2_mle": mle[2],
            "theta2_paper": p["theta2"], "theta2_lmom": pt2, "theta2_mle": mle[3],
            "theta_star_paper": p["theta_star"], "theta_star_lmom": pt2 / pt1, "theta_star_mle": mle[3] / mle[1],
            "lambda_star_paper": p["lambda_star"], "lambda_star_lmom": pl2 / pl1 ** (pt1 / pt2),
            "lambda_star_mle": mle[2] / mle[0] ** (mle[1] / mle[3]),
            "lmom_fit_cost": cost,                      # ~0 = sample L-moments inside TCEV domain
            "lmom_fit_t3_check": pop[2], "lmom_fit_t4_check": pop[3],
            "paper_params_vs_site_t3": ppop[2] - t3, "paper_params_vs_site_t4": ppop[3] - t4,
            "q100_paper_params": q(ppar, 0.01), "q100_lmom": q(par, 0.01), "q100_mle": q(mle, 0.01),
            "q100_site_ffa2020_lp3": np.nan,
        })
        log.info("%s %s n=%d t3 %.3f/%.3f t4 %.3f/%.3f cost=%.2e", sid, p["paper_name"], n,
                 p["t3"], t3, p["t4"], t4, cost)

    res = pd.DataFrame(rows).drop(columns="q100_site_ffa2020_lp3")
    res.to_csv(out / "replication_table2.csv", index=False)

    prov = {"run_stamp": stamp, "seed": a.seed,
            "inputs": {f.name: sha(f) for f in (a.dir / "paper_sites.csv", a.dir / "ams_all.csv")},
            "code": {f: sha(f) for f in ("replicate_paper.py", "tcev.py", "tcev_lmom.py")},
            "outputs": sorted(p.name for p in out.iterdir())}
    (out / f"qa_provenance_{stamp}.json").write_text(json.dumps(prov, indent=2))
    print(f"Outputs written to {out}")


if __name__ == "__main__":
    main()
