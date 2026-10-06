# TCEV flood frequency experiments

A small Python sandbox for understanding the **Two-Component Extreme Value (TCEV)** distribution
in flood frequency analysis (FFA), and for testing whether published TCEV results for eastern
Australia can be reproduced from open data.

This is a learning project, not a design tool. Nothing here replaces ARR guidance.

## What this repo does

1. **Fits TCEV** to annual maximum flood series, with Gumbel and GEV baselines (`tcev.py`, `run_ffa.py`).
2. **Fits TCEV by L-moments**, the method used by Totaro et al. (2024) (`tcev_lmom.py`).
3. **Downloads station data** from the WMAwater FFA Visualiser and catalogues it (`fetch_ffa.py`, `build_catalogue.py`).
4. **Replicates the paper's Table 2** for its 12 eastern Australian sites (`replicate_paper.py`).

## Background in plain English

*Written as a personal summary of the papers and webinar listed under References. Critique welcome.*

**1. Small and large floods often come from different processes.**
In eastern Australia the Interdecadal Pacific Oscillation (IPO) changes how often La Niña occurs:
about 50% of years in a negative IPO phase against about 12% in a positive phase.
The main effect is wetter catchments and more frequent floods, not more intense storms.
An annual maximum series can therefore mix "ordinary" years with wet-epoch, cyclone and East Coast Low floods.

**2. LP3 and GEV cannot separate them.**
Their shape parameter bends one curve through small and large floods alike.
Think of quadratic regression: a good fit inside the data, potentially explosive uncertainty beyond it.
Censoring low flows improves the tail fit but does not remove that coupling.

**3. TCEV models two flood populations.**
It has an "ordinary" and an "outlying" component, each with its own frequency and magnitude.
At four parameters it is the simplest mixture that decouples small from large floods.
It reproduces the "dog-leg" in a flood frequency plot using all the data.

TCEV does not split the data. It fits every annual maximum with a model that has two flood processes built into its structure.
The separation shows up in the fitted parameters, not in labelling individual floods.
The Multiple Grubbs-Beck Test (MGBT) does split the data: it sets a threshold and stops using the magnitudes below it.

**4. The evidence reported so far.**
Across 158 eastern Australian sites with 50+ years of record, Kuczera et al. (2025) report that the 95% confidence interval
on the 1% AEP flow is about half as wide for TCEV as for LP3.
Regional TCEV uncertainty was one third to one half of ARR regional LP3.
They attribute this to the TCEV right tail converging to a Gumbel distribution.
LP3 (with censoring) and TCEV (all data) fit about equally well. They extrapolate differently, so a good fit is necessary but not sufficient.

**Caveats.**
ARR 2019 still recommends LP3/GEV; TCEV is emerging research.
Most of the evidence so far comes from one research group.
The 178-site results (Kuczera webinar) await peer-reviewed publication.

## The TCEV in one equation

```
F(x) = exp( -λ1·exp(-x/θ1) - λ2·exp(-x/θ2) ),   θ2 > θ1
```

It is the product of two Gumbel-type distributions, one per flood process.
Each process produces a Poisson number of events per year (mean λ), and each event peak is exponential (mean exceedance θ).

| Parameter | Meaning |
|---|---|
| λ1, θ1 | Ordinary component: how often, and how big, in a typical year |
| λ2, θ2 | Outlying component: rarer, larger floods (cyclones, East Coast Lows, wet epochs) |

With one component dominant it collapses to Gumbel. That degeneracy is why the four parameters are hard to estimate.

**Three ways to estimate the parameters**, all of which give different answers on the same data:

| Method | Used by | In this repo |
|---|---|---|
| Sample L-moments (match mean, L-scale, L-skew, L-kurtosis) | Totaro et al. 2024 (SERRA) | `tcev_lmom.py` |
| Maximum likelihood | Rossi et al. 1984 | `tcev.py` |
| Robust Bayesian inference | Totaro, Kuczera & Iacobellis 2024 (J. Hydrology); Kuczera et al. 2025 | not implemented |

Confidence limits, and the "TCEV interval is half of LP3" finding, rely on the Bayesian method.
This repo reproduces point estimates only. It does not reproduce those confidence limits.

## Quick start

Requires Python 3 with numpy, scipy, pandas, matplotlib and openpyxl.

```bash
# 1. Download NSW and QLD station data (about 10 minutes, one request per second)
python fetch_ffa.py --states NSW QLD

# 2. Build the flat tables and match the paper's 12 sites to station ids
python build_catalogue.py

# 3. Reproduce the paper's Table 2
python replicate_paper.py
```

Outputs go to `outputs/<run>/` with a run log and a `qa_provenance_*.json` recording input hashes, code hashes and the random seed.

To fit your own annual maxima series, put FLIKE-format CSVs in `inputs/` and run `python run_ffa.py --pattern "MyGauge_*.csv"`.
The expected columns are `Gauged Obs, Annual Discharge, Maximum Year, AEP plot posn, AEP 1 in Y, AEPpc`.

## Data

Station data come from the [FFA Visualiser](https://ffa.wmawater.com.au), the WMAwater database described in Babister et al. (2023).
`fetch_ffa.py` calls the same JSON endpoint the web page uses.

- **The raw data are not stored in this repo.** The licence terms of the underlying data are not confirmed, so run the fetcher yourself.
- The site lists 1,712 stations. `station_catalogue.csv` lists their ids; the fetcher downloads NSW and QLD (562 stations).
- 178 of those 562 have 50+ years of record (annual maxima plus censored records). Of the 178, 132 carry the site's "Use" flag.
- The site's own "FFA 2020" curve is a Bayesian fit to log flows (the page reports the mean, standard deviation and skew of ln flow), so it appears to be LP3, not TCEV.
- The site gives annual maxima plus a separate list of censored (low) records. The paper fits the full series, so this repo combines both.

## Does it replicate? Totaro et al. (2024), Table 2

The paper fits TCEV to 12 long-record sites by L-moments. For each site I took the matching series from the FFA Visualiser
(matched by name and location) and ran the same method.

**Step 1: does the data match?** The paper reports sample L-skew (t3) and L-kurtosis (t4).
From the site data, t3 differs by up to 0.019 and t4 by up to 0.028 (medians 0.008 and 0.005).
The series are very close but not identical.

**Step 2: does the maths match?** Feeding the paper's published parameters into `tcev_lmom.py` reproduces the paper's t3 and t4 to within 0.001 at all 12 sites.
This confirms the L-moment calculation is correct, and that the paper's parameters are an exact L-moment match.

**Step 3: how close are the fitted parameters?** Each cell below reads *paper → this repo (L-moments)*.

| # | Site (station) | n paper / site | λ1 | θ1 | λ2 | θ2 | 1% AEP (m³/s), paper params / L-mom / MLE |
|---|---|---|---|---|---|---|---|
| 1 | Pascoe River (102101) | 53 / 54 | 2.82 → 2.42 | 253.3 → 279.8 | 1.79 → 1.85 | 733.7 → 719.6 | 3,802 / 3,753 / 3,538 |
| 2 | Barron River (110003) | 95 / 96 | 3.11 → 3.09 | 29.0 → 28.6 | 0.70 → 0.70 | 156.1 → 155.2 | 662 / 659 / 568 |
| 3 | Fisher Creek (112002) | 91 / 93 | 4.19 → 4.57 | 10.0 → 9.5 | 1.05 → 1.05 | 74.8 → 74.5 | 348 / 346 / 330 |
| 4 | Don River (121001) | 63 / 62 | 3.97 → 4.56 | 167.2 → 146.1 | 0.48 → 0.48 | 1304.4 → 1310.5 | 5,043 / 5,077 / 4,460 |
| 5 | Waterpark Creek (129001) | 67 / 67 | 2.92 → 3.55 | 103.6 → 118.2 | 0.35 → 0.16 | 459.6 → 640.7 | 1,632 / 1,768 / 1,688 |
| 6 | Barker Creek (136203) | 80 / 72 | 2.14 → 1.88 | 24.1 → 22.3 | 0.79 → 0.75 | 162.3 → 156.7 | 708 / 675 / 529 |
| 7 | Albert River (145102) | 93 / 94 | 2.45 → 2.24 | 69.0 → 67.4 | 1.25 → 1.29 | 434.1 → 427.4 | 2,094 / 2,076 / 1,964 |
| 8 | Swan Creek (422306) | 101 / 101 | 2.58 → 2.29 | 9.8 → 10.7 | 0.73 → 0.71 | 87.2 → 86.9 | 374 / 370 / 322 |
| 9 | Logan R (145003) | 66 / 68 | 2.26 → 2.62 | 80.5 → 82.6 | 0.66 → 0.58 | 333.3 → 346.9 | 1,395 / 1,406 / 1,106 |
| 10 | Moonan Brook (210017) | 80 / 73 | 5.01 → 5.46 | 5.0 → 4.8 | 0.53 → 0.59 | 37.1 → 33.3 | 147 / 136 / 142 |
| 11 | Williams River (210011) | 89 / 88 | 2.61 → 2.70 | 75.0 → 84.0 | 1.08 → 1.05 | 321.6 → 325.9 | 1,504 / 1,516 / 1,300 |
| 12 | Corang River (215004) | 90 / 94 | 2.24 → 2.70 | 30.9 → 26.1 | 1.84 → 1.80 | 136.2 → 138.2 | 710 / 717 / 677 |

| Parameter | Median absolute difference |
|---|---|
| λ2 (rare frequency) | 3% |
| θ2 (rare magnitude) | 1% |
| λ1 (ordinary frequency) | 12% |
| θ1 (ordinary magnitude) | 8% |

**What this shows**

- The outlying component (λ2, θ2) replicates well. The ordinary component (λ1, θ1) is less stable, because the four parameters trade off against each other.
- The 1% AEP flow is much more stable than the parameters behind it. It agrees within about 1% at 9 of 12 sites and within about 5% at 10 of 12.
- **Waterpark Creek (site 5) is the clear outlier.** The site's record for that gauge ends in 2012; the paper's runs to 2019.
- Barker Creek (6) and Moonan Brook (10) have 72 and 73 years on the site against 80 in the paper.
- **The method matters.** Maximum likelihood on the same data gives a 1% AEP that is at least 12% below the paper's at 6 of 12 sites,
  with very different ordinary-component parameters. Choosing how to fit TCEV matters as much as choosing TCEV.

Full results: `outputs/replication_20261006_162529/replication_table2.csv`.

## Limitations

- At-site TCEV with four parameters is weakly identified on short records. Treat any fit under about 30 years as illustration only.
- The site data are the FFA 2020 dataset and stop around 2018–2020. They differ slightly from the series in the paper.
- No confidence limits are computed. Parametric bootstrap or Bayesian inference would be the next step.
- The "Use" / "Not Use" flag and its reason come from the site. `stations.csv` keeps both, and filtering on them is your call.

## Files

| File | Purpose |
|---|---|
| `tcev.py` | TCEV CDF, PDF, maximum likelihood fit, quantiles; GEV and Gumbel baselines |
| `tcev_lmom.py` | Sample and population L-moments, L-moment TCEV fit |
| `run_ffa.py` | Fit TCEV, GEV, Gumbel to FLIKE-format CSVs; figures and provenance |
| `fetch_ffa.py` | Download station JSON from the FFA Visualiser; write the station catalogue |
| `build_catalogue.py` | Consolidate raw JSON into flat tables; flag 50+ year stations; match the paper's sites |
| `replicate_paper.py` | Reproduce Table 2 of Totaro et al. (2024) |

## References

1. Totaro, V., Gioia, A., Kuczera, G. & Iacobellis, V. (2024). Modelling multidecadal variability in flood frequency using the two-component extreme value distribution. *Stochastic Environmental Research and Risk Assessment*, 38, 2157–2174. https://doi.org/10.1007/s00477-024-02673-8 (open access)
2. Kuczera, G., Totaro, V., Iacobellis, V., Babister, M. & Retallick, M. (2025). Are LP3 and GEV fit-for-purpose in flood frequency analysis? *Hydrology and Water Resources Symposium 2025*.
3. Kuczera, G. *Rethinking Flood Frequency Analysis* [webinar]. Australian Water School. https://awschool.com.au/training/rethinking-ffa/
4. Totaro, V., Kuczera, G. & Iacobellis, V. (2024). Goodness-of-fit, identifiability and extrapolation: can the two-component extreme value distribution be used in at-site flood frequency analysis? *Journal of Hydrology*, 640, 131590.
5. Babister, M., Retallick, M., Vibhani, T., Breda, A., Dunning, N. & Weeks, W. (2023). Revised regional flood frequency estimation for Australia. *Hydrology and Water Resources Symposium*, Sydney.
6. Rossi, F., Fiorentino, M. & Versace, P. (1984). Two-component extreme value distribution for flood frequency analysis. *Water Resources Research*, 20(7), 847–856.
