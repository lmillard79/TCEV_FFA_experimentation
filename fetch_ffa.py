"""Fetch station data from the WMAwater FFA Visualiser (https://ffa.wmawater.com.au)
via the JSON endpoint the page itself uses: POST /get-station-info/<station id>.

Two modes:
    python fetch_ffa.py 102101 125004          # named stations -> raw JSON + per-station CSVs
    python fetch_ffa.py --states NSW QLD       # whole state(s) -> raw JSON only (resumes)

Always writes inputs/ffa_site/station_catalogue.csv (every station id on the site, with the
state/region lists it appears in) and station_lists.json (the site's own listOfStations).
Consolidated tables for bulk runs are built by build_catalogue.py from raw/*.json.

Per named station (inputs/ffa_site/):
    ams_<id>.csv         annual maxima + censored records (year, flow, AEP)
    quantiles_<id>.csv   FFA 2020 quantiles with 90% limits, FFA 2014 quantiles
    meta_<id>.json       station metadata, posterior moments, QA checks
"""
import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SITE = "https://ffa.wmawater.com.au/"
BASE = SITE + "get-station-info/"
UA = "TCEV_experiments/1.0 (WRM Water & Environment; research use)"
DELAY_S = 1.0  # be gentle with the server


def http(url, post=False):
    req = urllib.request.Request(url, data=b"" if post else None, method="POST" if post else "GET",
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def site_station_lists():
    """The site embeds its station lists in the page script: listOfStations = {...};"""
    html = http(SITE).decode("utf-8")
    m = re.search(r"listOfStations\s*=\s*\{", html)
    if not m:
        raise SystemExit("listOfStations not found on page - site layout changed")
    start, depth = m.end() - 1, 0
    for i in range(start, len(html)):  # match braces; a regex end-marker was unreliable
        depth += html[i] == "{"
        depth -= html[i] == "}"
        if depth == 0:
            return json.loads(html[start:i + 1].replace("'", '"'))
    raise SystemExit("unterminated listOfStations")


def write_catalogue(lists, out):
    rows = {}
    for key, ids in lists.items():
        for sid in ids:
            rows.setdefault(str(sid), []).append(key)
    cat = pd.DataFrame({"id": list(rows), "listed_under": [" ".join(v) for v in rows.values()]})
    cat["state"] = cat["listed_under"].str.split().apply(
        lambda ks: next((k for k in ks if not k.isdigit()), ""))
    cat["rffe_region"] = cat["listed_under"].str.split().apply(
        lambda ks: next((k for k in ks if k.isdigit()), ""))
    cat.sort_values("id").to_csv(out / "station_catalogue.csv", index=False)
    (out / "station_lists.json").write_text(json.dumps(lists))
    return cat


def ams_table(j):
    """Annual maxima and censored records as one long table."""
    parts = []
    for kind, pre in (("annual_max", "ams"), ("censored", "censored")):
        if pre + "Year" not in j:  # no censored records at this station
            continue
        parts.append(pd.DataFrame({"record_type": kind, "year": j[pre + "Year"],
                                   "flow_m3s": j[pre + "Quantile"], "aep_1_in_x": j[pre + "AEP"]}))
    d = pd.concat(parts, ignore_index=True)
    d["aep_pc"] = 100 / d["aep_1_in_x"]
    # page's "AEP" axis is the standard-normal variate of non-exceedance
    d["z_variate"] = stats.norm.isf(1 / d["aep_1_in_x"])
    return d.sort_values("flow_m3s", ascending=False).reset_index(drop=True)


def quantile_table(j):
    new = pd.DataFrame({"aep_1_in_x": j["newAEP"], "ffa2020_flow": j["newQuantile"],
                        "ffa2020_lower90": j["newLowerCF"], "ffa2020_upper90": j["newUpperCF"]})
    if "oldAEP" not in j:  # some stations have no FFA 2014 analysis
        return new.assign(ffa2014_flow=np.nan).sort_values("aep_1_in_x")
    old = pd.DataFrame({"aep_1_in_x": j["oldAEP"], "ffa2014_flow": j["oldQuantile"]})
    return new.merge(old, on="aep_1_in_x", how="outer").sort_values("aep_1_in_x")


def qa_checks(j, d):
    """Cunnane (i-0.4)/(n+0.2) with i=1 gives the implied record length n."""
    top = d.loc[d["aep_1_in_x"].idxmax(), "aep_1_in_x"]
    yrs = j["newPeriod"].split("-")
    return {
        "n_annual_max": len(j["amsYear"]), "n_censored": len(j.get("censoredYear", [])),
        "n_listed_total": len(d), "n_implied_by_top_aep": round(top * 0.6 - 0.2, 1),
        "n_in_ffa2020_period": int(yrs[1]) - int(yrs[0]) + 1,
        "ams_years_outside_period": sorted(y for y in j["amsYear"]
                                           if not int(yrs[0]) <= y <= int(yrs[1])),
        "duplicate_years": bool(d["year"].duplicated().any()),
        "ams_sorted_desc": bool(np.all(np.diff(j["amsQuantile"]) <= 0)),
    }


def write_station_files(sid, body, j, out):
    d = ams_table(j)
    d.to_csv(out / f"ams_{sid}.csv", index=False)
    quantile_table(j).to_csv(out / f"quantiles_{sid}.csv", index=False)
    p = j["params"]
    meta = {k: j.get(k) for k in ("ID", "River", "Name", "State", "Region", "StateRegion", "Area",
                                  "Latitude", "Longitude", "Status", "Reason",
                                  "newProject", "newPeriod", "oldProject", "oldPeriod")}
    names = ("mean_lnQ", "ln_sd_lnQ", "skew_lnQ")
    meta["posterior_moments"] = {
        "mean": dict(zip(names, [p[0], p[2], p[4]])),
        "std_dev": dict(zip(names, [p[1], p[3], p[5]])),
        "correlation": {"sd_vs_mean": p[6], "skew_vs_mean": p[7], "skew_vs_sd": p[8]},
    }
    meta["qa"] = qa_checks(j, d)
    meta["fetch"] = {"url": BASE + sid, "time": datetime.now().isoformat(timespec="seconds"),
                     "response_sha256": hashlib.sha256(body).hexdigest()}
    (out / f"meta_{sid}.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="station ids, e.g. 102101 125004")
    ap.add_argument("--states", nargs="*", help="site list keys, e.g. NSW QLD VIC SA NT WA TAS")
    ap.add_argument("--refresh", action="store_true", help="re-download even if raw JSON exists")
    ap.add_argument("--out", type=Path, default=Path("inputs") / "ffa_site")
    a = ap.parse_args()
    if not a.ids and not a.states:
        ap.error("give station ids and/or --states")

    (a.out / "raw").mkdir(parents=True, exist_ok=True)
    lists = site_station_lists()
    cat = write_catalogue(lists, a.out)
    print(f"Catalogue: {len(cat)} stations on site -> station_catalogue.csv")

    named = [str(i) for i in a.ids]
    bulk = []
    for s in a.states or []:
        if s not in lists:
            raise SystemExit(f"unknown list {s!r}; options: {list(lists)}")
        bulk += [str(i) for i in lists[s]]
    todo = list(dict.fromkeys(named + bulk))

    log, fetched, missing, errors = [], 0, [], []
    for k, sid in enumerate(todo):
        raw = a.out / "raw" / f"{sid}.json"
        if raw.exists() and not a.refresh:
            body = raw.read_bytes()
        else:
            if fetched:
                time.sleep(DELAY_S)
            try:
                body = http(BASE + sid, post=True)
            except urllib.error.HTTPError as e:
                missing.append((sid, e.code))
                print(f"{sid}: HTTP {e.code}")
                continue
            except Exception as e:  # network blip: record and carry on
                errors.append((sid, repr(e)))
                print(f"{sid}: {e!r}")
                continue
            raw.write_bytes(body)
            fetched += 1
        if sid in named:
            j = json.loads(body)
            meta = write_station_files(sid, body, j, a.out)
            qa = meta["qa"]
            print(f"{sid}: {j['River']} at {j['Name']}  AMS={qa['n_annual_max']} "
                  f"censored={qa['n_censored']} implied n={qa['n_implied_by_top_aep']}")
        elif fetched and fetched % 50 == 0:
            print(f"  {fetched} downloaded ({k + 1}/{len(todo)})")

    print(f"Done: {fetched} downloaded, {len(todo) - fetched - len(missing) - len(errors)} from cache, "
          f"{len(missing)} missing, {len(errors)} errors")
    (a.out / "fetch_problems.json").write_text(json.dumps({"missing": missing, "errors": errors}))


if __name__ == "__main__":
    main()
