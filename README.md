# Toxic Gas Detection – Domain Shift Robust Pipeline

Multi-output regression pipeline (23 targets) built for the ENS x Bertin Technologies Challenge 2025.  
The main challenge is a strong **domain shift induced by humidity**, heavily impacting sensor signals.

---

## Problem

Sensor data (IMS channels + auxiliary sensors) show a major train/test mismatch due to:
- strong dependency of signals on humidity
- different humidity regimes between datasets

This leads to a clear **covariate shift** that breaks standard models.

---

## Approach

### 1. External dehumidification

Each IMS sensor is corrected independently using a polynomial Ridge regression:

$$\[
X_j^{resid} = X_j - \hat{f}_j(H)
\]$$

- fitted on **train + test (transductive setting)**
- removes humidity-induced bias
- both raw and corrected features are kept

---

### 2. Feature engineering (sensor geometry)

Features focus on **signal shape and spatial structure**, not raw amplitude:

- normalized views (L2)
- local gradients and curvature
- inter-block contrasts
- statistical descriptors (mean, std, energy, IQR)
- directional similarity (cosine)

---

### 3. Model

`RandomForestRegressor` (multi-output)

Chosen for:
- robustness to noise and non-linearities
- stability under distribution shift
- minimal sensitivity to feature scaling

---

## Results

- strong reduction of train/test mismatch
- stable predictions under extreme humidity
- public leaderboard score: **0.14318**
- ranking: **Top 10 / 160**

---

## Author

Erwan Ouabdesselam  
Master MS2A – Sorbonne Université
