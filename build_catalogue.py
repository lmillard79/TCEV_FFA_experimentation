"""Consolidate inputs/ffa_site/raw/*.json into flat catalogue tables.

Writes to inputs/ffa_site/:
    stations.csv        one row per downloaded station: metadata, record counts, QA flags,
                        FFA 2020 / 2014 quantiles at 2,5,10,20,50,100 yr, source file hash
    ams_all.csv         every annual-max and censored record, all stations (long format)
    quantiles_all.csv   every FFA quantile row, all stations (long format)
    paper_sites.csv     the 12 Totaro et al. (2024) sites matched to station ids (see PAPER_SITES)

Station metadata is also kept in the JSON, so nothing here is hand-typed except PAPER_SITES.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Totaro, Gioia, ... Iacobellis (2024), SERRA 38:2157-2174, Tables 1-2 (as published).
# lambda1, theta1, lambda2, theta2, theta*, lambda*, t3, t4 are the paper's L-moment fits.
PAPER_SITES = [
    (1, "Pascoe River at Fall Creek", -12.88, 142.98, 651, 53, "1968-2020", 2.82, 253.33, 1.79, 733.66, 2.90, 1.25, 0.270, 0.168),
    (2, "Barron River at Picnic Crossing", -17.26, 145.54, 228, 95, "1926-2020", 3.11, 29.00, 0.70, 156.11, 5.38, 0.57, 0.423, 0.248),
    (3, "Fisher Creek at Nerada", -17.57, 145.91, 16, 91, "1929-2019", 4.19, 9.97, 1.05, 74.78, 7.50, 0.86, 0.400, 0.185),
    (4, "Don River at Ida Creek", -20.29, 148.12, 604, 63, "1958-2020", 3.97, 167.25, 0.48, 1304.44, 7.80, 0.40, 0.507, 0.318),
    (5, "Waterpark Creek at Byfield", -22.84, 150.67, 212, 67, "1953-2019", 2.92, 103.61, 0.35, 459.55, 4.43, 0.27, 0.398, 0.290),
    (6, "Barker Creek at Brooklands", -26.74, 151.82, 249, 80, "1941-2020", 2.14, 24.05, 0.79, 162.29, 6.75, 0.71, 0.426, 0.219),
    (7, "Albert River at Bromfleet", -27.91, 153.11, 544, 93, "1928-2020", 2.45, 69.00, 1.25, 434.06, 6.29, 1.09, 0.349, 0.160),
    (8, "Swan Creek at Swanfels", -28.16, 152.28, 83, 101, "1920-2020", 2.58, 9.81, 0.73, 87.19, 8.89, 0.66, 0.462, 0.230),
    (9, "Logan R at Forest Home", -28.20, 152.77, 175, 66, "1954-2019", 2.26, 80.55, 0.66, 333.34, 4.14, 0.54, 0.381, 0.239),
    (10, "Moonan Brook at Moonan Brook", -31.94, 151.28, 103, 80, "1941-2020", 5.01, 5.00, 0.53, 37.12, 7.42, 0.43, 0.495, 0.304),
    (11, "Williams River at Tillegra", -32.32, 151.69, 194, 89, "1932-2020", 2.61, 75.04, 1.08, 321.58, 4.29, 0.87, 0.349, 0.193),
    (12, "Corang River at Hockeys", -35.15, 150.03, 166, 90, "1930-2019", 2.24, 30.90, 1.84, 136.23, 4.41, 1.53, 0.277, 0.146),
]
PAPER_COLS = ["paper_id", "paper_name", "paper_lat", "paper_lon", "paper_area_km2", "paper_n_years",
              "paper_period", "lambda1", "theta1", "lambda2", "theta2", "theta_star", "lambda_star", "t3", "t4"]
Q_AEPS = [2, 5, 10, 20, 50, 100]


def norm(s):
    """Normalise a station name for matching."""
    s = re.sub(r"\b(river|r|creek|ck|brook|at)\b", "", str(s).lower())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("inputs") / "ffa_site")
    ap.add_argument("--min-years", type=int, default=50, help="long-record threshold (n_total)")
    a = ap.parse_args()
    raws = sorted((a.dir / "raw").glob("*.json"))

    st, ams, qs = [], [], []
    for f in raws:
        body = f.read_bytes()
        j = json.loads(body)
        sid = f.stem
        yrs = j["newPeriod"].split("-")
        row = {
            "id": sid, "river": j["River"], "name": j["Name"], "state": j["State"],
            "rffe_region": j.get("Region"), "area_km2": float(j["Area"]) if j.get("Area") else np.nan,
            "lat": float(j["Latitude"]), "lon": float(j["Longitude"]),
            "status": j.get("Status"), "reason": j.get("Reason"),
            "ffa2020_period": j["newPeriod"], "n_annual_max": len(j["amsYear"]),
            "n_censored": len(j.get("censoredYear", [])),
            "n_total": len(j["amsYear"]) + len(j.get("censoredYear", [])),
            "first_ams_year": min(j["amsYear"]) if j["amsYear"] else np.nan,
            "last_ams_year": max(j["amsYear"]) if j["amsYear"] else np.nan,
            "max_flow": max(j["amsQuantile"]) if j["amsQuantile"] else np.nan,
            "has_ffa2014": "oldAEP" in j,
            "n_period_years": int(yrs[1]) - int(yrs[0]) + 1,
            "source_sha256": hashlib.sha256(body).hexdigest(),
        }
        row["n_ams_outside_period"] = sum(not int(yrs[0]) <= y <= int(yrs[1]) for y in j["amsYear"])
        qmap = dict(zip(j["newAEP"], j["newQuantile"]))
        for q in Q_AEPS:
            row[f"ffa2020_q{q}"] = qmap.get(q, np.nan)
        omap = dict(zip(j.get("oldAEP", []), j.get("oldQuantile", [])))
        for q in Q_AEPS:
            row[f"ffa2014_q{q}"] = omap.get(q, np.nan)
        st.append(row)

        for kind, pre in (("annual_max", "ams"), ("censored", "censored")):
            if pre + "Year" not in j:   # no censored records at this station
                continue
            ams.append(pd.DataFrame({"id": sid, "record_type": kind, "year": j[pre + "Year"],
                                     "flow_m3s": j[pre + "Quantile"], "aep_1_in_x": j[pre + "AEP"]}))
        qs.append(pd.DataFrame({"id": sid, "aep_1_in_x": j["newAEP"], "ffa2020_flow": j["newQuantile"],
                                "ffa2020_lower90": j["newLowerCF"], "ffa2020_upper90": j["newUpperCF"]}))

    stations = pd.DataFrame(st)
    # record length = annual maxima + censored (the n used for plotting positions)
    stations["long_record"] = stations["n_total"] >= a.min_years
    stations["usable"] = stations["status"].eq("Use")   # site's own Use / Not Use flag
    stations.to_csv(a.dir / "stations.csv", index=False)
    long_ids = set(stations.loc[stations["long_record"], "id"])
    stations[stations["long_record"]].sort_values("n_total", ascending=False).to_csv(
        a.dir / f"stations_long{a.min_years}.csv", index=False)
    pd.concat(ams, ignore_index=True).to_csv(a.dir / "ams_all.csv", index=False)
    pd.concat(qs, ignore_index=True).to_csv(a.dir / "quantiles_all.csv", index=False)

    # ---- match the paper's 12 sites by name + location (never by name alone) ----
    paper = pd.DataFrame(PAPER_SITES, columns=PAPER_COLS)
    stations["key"] = (stations["river"] + " " + stations["name"]).map(norm)
    out = []
    for _, p in paper.iterrows():
        d_lat = (stations["lat"] - p["paper_lat"]).abs()
        d_lon = (stations["lon"] - p["paper_lon"]).abs()
        near = stations[(d_lat < 0.05) & (d_lon < 0.05)].copy()   # ~5 km
        pk = norm(p["paper_name"])
        near["name_match"] = near["key"].map(lambda k: k == pk or set(pk.split()) <= set(k.split()))
        pick = near.sort_values(["name_match", "n_total"], ascending=False).head(1)
        r = p.to_dict()
        r["n_candidates_within_5km"] = len(near)
        if len(pick):
            for c in ("id", "river", "name", "state", "area_km2", "lat", "lon", "ffa2020_period",
                      "n_annual_max", "n_censored", "first_ams_year", "last_ams_year"):
                r["site_" + c] = pick.iloc[0][c]
            r["name_match"] = bool(pick.iloc[0]["name_match"])
        out.append(r)
    pd.DataFrame(out).to_csv(a.dir / "paper_sites.csv", index=False)
    print(f"{len(stations)} stations -> stations.csv ({len(long_ids)} with n_total >= {a.min_years}); "
          "paper matches:",
          sum("site_id" in r for r in out), "of 12")


if __name__ == "__main__":
    main()
