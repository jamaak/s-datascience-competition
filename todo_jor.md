# Data Processing & Anomaly Audit (`todo_jor.md`)

## 1. Root Causes of Extreme Leaderboard Sensitivity to Single `tma_mdpl` Outliers

Assisted by Qwen

- **Fix `df_final` vs `df_train` pipeline disconnect**: Apply `clean_tma_spikes` directly to `df_train` before passing it into `run_ensemble`, because cleaning `df_final` alone leaves `df_train` completely uncleaned in Section 3.3 experiments.
- **Scrub the 1-year seasonal reference window (`2024-09-19` to `2025-05-18`)**: Sanitize every spike and zero-dropout in this exact training window first, because `predict_anchored_seasonal` and `tma_lag_1y` project those exact historical timestamps directly onto the test horizon (`2025-09-19` to `2026-05-18`).
- **Replace the single-point origin anchor (`2025-09-18 18:00:00`) with a trailing median**: Compute each station's origin offset using a 3-to-7-day trailing median instead of `.tail(1)`, because single-step jumps at the cutoff timestamp (`Kedungupit +1.27 m`, `Karangnongko -0.85 m`, `Kajangan +0.70 m`, `Jurug +0.65 m`, `Peren +0.60 m`, `Floodway Bridge C -0.50 m`) permanently shift the entire 242-day test forecast.
- **Stop using lag-1 (`t-1`) replacement on multi-step anomalies**: Replace multi-step flatlines and consecutive spike bursts (2–5 consecutive timestamps at `Brangkal`, `Jarum`, and `Kali Pepe - PTPN`) via time-bounded PCHIP or local median interpolation across `t-k` to `t+k`, rather than copying `t-1` which propagates step-1 errors across the whole burst.
- **Use Huber or Tukey biweight mean in `fit_bias_calibrator`**: Replace the arithmetic sample mean of residuals in bias calibration and station-level RMSE weighting, because a single uncleaned decimal-shift outlier (`Napel 325.83 m`, `Kali Anyar 323.21 m`, `Jarum 250.14 m`, `Bojonegoro 225.13 m`, `Kedungupit 207.74 m`) exerts up to $6,400\times$ squared-error leverage and drags the entire station's mean residual.

---

## 2. Target (`tma_mdpl`) Processing Gaps Beyond Simple Lag Normalization

- **Eliminate zero-dropout sensor failures (`tma_mdpl == 0.0`)**: Mask exact `0.0` readings at elevated gauges (`Jurug` dropping from `78.5 m` to `0.0 m` on `2025-02-28`, `Kali Pepe - PTPN` dropping from `88.7 m` to `0.0 m`, and `ONLIMOPBG02` dropping from `21.9 m` to `0.0 m`) as `NaN` prior to any lag or rolling computation.
- **Correct decimal-point and digit-entry corruption before Z-score clipping**: Explicitly filter 10x/100x telemetry entry errors (`Napel 325.83 m`, `Kali Anyar 323.21 m`, `Jarum 250.14 m`, `Bojonegoro 225.13 m`, `Kedungupit 207.74 m`, `Peren 170.10 m`, `Floodway Bridge C 47.00 m`, `Sekayu 0.72 m`) using physical elevation bounds per station rather than global standard deviations.
- **Protect low-IQR gauges from over-smoothing**: Scale outlier thresholds using hydrograph asymmetry (allow rapid upward flood rises when concurrent or upstream 24h–72h rainfall is high, while penalizing isolated single-step drops or spikes during dry spells) so narrow-range gauges (`Ngrembang` IQR `0.12 m`, `Lorog` IQR `0.20 m`, `Kali Pepe - PTPN` IQR `0.26 m`) do not lose real flood peaks.
- **Bridge the February 2025 training blackout (`89%` missing target rows)**: Increase seasonal-reference interpolation beyond `limit=6` (2 days) or fall back to multi-year station-month-day climatology (`2023` + `2024`) when constructing `tma_lag_1y` for `February 2026`, because `February 2025` contains only `280` rows across all 30 stations and leaves `February 2026` seasonal anchors mostly empty.

---

## 3. Hidden Exogenous (`data_lingkungan.csv`) Bugs & Anomalies Others Miss

- **Fix the 139-day `solar_radiation_mj_m2 == -999` blackout in Test (`2025-12-31` to `2026-05-18`)**: Never apply linear `.interpolate(limit_direction="both")` across the `100,080` sentinel `-999` rows in Test (57% of the test horizon), because the last valid reading occurs at `2025-12-30 23:00` (`0.0` at night), which forward-fills `0.0` across all daytime hours (`06:00`, `12:00`, `18:00`) for 139 consecutive days and simulates a 5-month solar eclipse. Impute missing solar radiation using each station's historical **month × hour** median instead.
- **Drop zero-variance `*_missing` indicator flags**: Remove `solar_radiation_missing`, `nino_34_missing`, and `soil_moisture_*_missing` from model features, because they equal `0` across 100% of `df_train` and only flip to `1` during late-test sensor outages (`2025-12-31` to `2026-05-18`), introducing pure covariate shift with zero learned split value.
- **Impute the 18-day `nino_34` outage (`2026-05-01` to `2026-05-18`) with damped persistence**: Replace flat forward-filling across the `12,960` missing late-test rows of `nino_34` and the `720` missing final-day rows (`2026-05-18`) of `soil_moisture_*`, `surface_pressure_hpa`, `pressure_msl_hpa`, `rmm1`, `rmm2`, and `mjo_*` with station-specific trailing trends or climatological decay.
- **Never fall back to cross-station global medians in `clean_environment`**: Group all fallback `.fillna()` steps strictly by `nama_pos`, because global median imputation blends highland stations (`Ngadipiro` `993.8 hPa`) with coastal plains (`Karanggeneng` `1010.9 hPa`) and corrupts barometric altitude proxies.
- **Clip physically impossible supersaturation (`dew_point_c > temperature_c`)**: Enforce `dew_point_c <= temperature_c` (or clip `dewpoint_depression >= 0.0`) across the 24 rows where dew point exceeds air temperature due to reanalysis rounding.
- **Aggregate hourly weather before downsampling to the 6h/12h target timestamps**: Compute cumulative rainfall, peak hourly intensity, daytime solar integral, and soil moisture drawdown on the full 1-hour `data_lingkungan` grid before merging with `df_train`/`df_test`, because sampling only at `06:00`, `12:00`, and `18:00` misses 21 out of 24 hourly observations per day.
- **Drop redundant `rainfall_openmeteo_mm` and correct solar unit assumptions**: Remove `rainfall_openmeteo_mm` (100% identical to `rainfall_mm` across all `888,480` rows), and treat `solar_radiation_mj_m2` as instantaneous irradiance in $\text{W/m}^2$ (peaks at `1,043.8` at noon) rather than $\text{MJ/m}^2$ when engineering evapotranspiration proxies.
- **Break identical weather-grid ties across co-located gauges**: Add static catchment and hydraulic interaction features (`built_surface_m2`, elevation proxy, upstream river distance) to distinguish station pairs that share 100% identical Open-Meteo reanalysis series (`Jarum == Serenan`, `Ketonggo == Napel`, and `Kali Anyar - Kreteg Abang == Kali Pepe - PTPN == Kali Pepe - Tugu Boto`).
