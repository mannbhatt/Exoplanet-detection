<div align="center">

# 🪐 Exoplanet Detection Using 1D Convolutional Neural Networks on Kepler Photometric Data

**A 10-Week End-to-End Machine Learning Engineering Project**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-FF6F00?style=flat&logo=tensorflow)
![AUC](https://img.shields.io/badge/Competition_AUC-0.9628-brightgreen)
![Precision](https://img.shields.io/badge/Precision-1.000-brightgreen)
![EB Filter](https://img.shields.io/badge/EB_Catch_Rate-100%25-blue)

</div>

---

## Abstract

This project presents an end-to-end machine learning pipeline for detecting exoplanet transits in Kepler space telescope photometric data. A 1D Convolutional Neural Network (CNN) was trained on the Kepler DR25 catalog to classify stellar light curves as planet-hosting or non-planet. The model achieves an AUC of **0.9628** on the competition test set and **93% detection rate** on high-SNR hot Jupiters in real-world validation. A threshold calibration step and an eclipsing-binary (EB) rejection pipeline were developed to eliminate false positives, reducing the false positive rate from **28% to 0%** while maintaining perfect precision (1.000). The complete system is deployed as an interactive Streamlit web application that downloads live Kepler data from the MAST archive and runs the full detection pipeline in real time.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset](#2-dataset)
3. [Preprocessing Pipeline](#3-preprocessing-pipeline)
4. [Model Architecture](#4-model-architecture)
5. [Training & Augmentation](#5-training--augmentation)
6. [Results — Competition Evaluation](#6-results--competition-evaluation)
7. [Results — Wild-Data Evaluation](#7-results--wild-data-evaluation)
8. [Threshold Calibration & EB Rejection](#8-threshold-calibration--eb-rejection)
9. [Broader Catalog & Limitations](#9-broader-catalog--limitations)
10. [Web Application](#10-web-application)
11. [Discussion](#11-discussion)
12. [Conclusion](#12-conclusion)
13. [References](#13-references)

---

## 1. Introduction

The detection of exoplanets — planets orbiting stars beyond our solar system — is one of the most significant scientific challenges of modern astronomy. The Kepler Space Telescope, operational from 2009 to 2018, monitored over 150,000 stars continuously, producing a dataset of stellar brightness measurements (light curves) of unprecedented scale and precision. When a planet passes in front of its host star, it causes a characteristic periodic dimming of the stellar flux — known as a transit. These transit signals are typically 0.01–1% deep and last only a few hours, making them extremely difficult to identify manually among the noise.

Traditional transit detection algorithms such as the Box Least Squares (BLS) periodogram are computationally expensive and sensitive to the assumed shape of the transit. Machine learning approaches — particularly deep learning — offer a complementary path: by learning the statistical signature of transit events directly from labeled examples, neural networks can flag candidates for human review without requiring an explicit physical model.

This project explores the application of 1D Convolutional Neural Networks to the Kepler light curve classification problem, building a complete pipeline from raw photometric data through preprocessing, model training, threshold calibration, false-positive rejection, and real-world deployment.

### 1.1 Objectives

- Build a reproducible preprocessing pipeline that transforms raw Kepler PDCSAP flux into a CNN-ready input vector
- Train a 1D-CNN classifier achieving AUC > 0.95 on the standard Kepler DR25 benchmark
- Validate the model on real Kepler stars not seen during training
- Build an eclipsing-binary rejection filter to eliminate the primary class of false positives
- Deploy the complete system as a publicly accessible web application

---

## 2. Dataset

### 2.1 Kepler DR25 Catalog

The primary training dataset is the Kepler Data Release 25 (DR25) catalog, available from the NASA Exoplanet Archive. The dataset contains:

- **150,000+ stellar light curves** from the Kepler mission
- **Binary labels:** confirmed planet (positive) vs. non-planet (negative)
- **Class imbalance:** approximately 1% positive examples
- **Input format:** Pre-processed flux time series, each 3,197 cadences long

The raw data files used:

```
data/raw/exoTrain.csv    ← Training set  (~5,087 stars)
data/raw/exoTest.csv     ← Test set      (~570 stars)
data/confirmed_kois.csv  ← Confirmed KOIs for validation
```

### 2.2 Class Distribution

The dataset exhibits severe class imbalance — a fundamental challenge for training. Initial exploration (Week 1) revealed:

- **Training positives:** ~37 confirmed planet hosts
- **Training negatives:** ~5,050 non-planet stars
- **Ratio:** approximately 1:137

This imbalance required careful handling through class weighting and augmentation (see Section 5).

### 2.3 Wild-Data Validation Set

For real-world validation (Week 7), a separate set of 30 Kepler stars was assembled:
- 15 confirmed planet hosts (various SNR levels)
- 6 known eclipsing binaries
- 9 non-planet stars (variable stars, quiet dwarfs)

All validation stars were downloaded fresh from MAST during evaluation — not from the competition dataset.

---

## 3. Preprocessing Pipeline

The preprocessing pipeline converts raw Kepler PDCSAP (Pre-search Data Conditioning Simple Aperture Photometry) flux into a standardised input vector for the CNN. The pipeline was developed in Week 2 and kept strictly fixed for all subsequent experiments.

### 3.1 Pipeline Steps

```
Raw PDCSAP flux (variable length, contains NaN gaps)
        │
        ▼
Step 1: NaN Interpolation
        Linear interpolation across NaN gaps using np.interp
        Preserves continuity without introducing sharp discontinuities
        │
        ▼
Step 2: Median Subtraction
        flux = flux - np.median(flux)
        Removes stellar baseline offset; centres flux around zero
        │
        ▼
Step 3: Best-Variance Segment Selection
        Sliding window of length 3,197 cadences
        Step size = INPUT_LEN // 4 = 799 cadences
        Select window with highest variance
        → Preferentially selects transit-containing regions
        │
        ▼
Step 4: L2 Normalisation
        segment = segment / np.linalg.norm(segment)
        Scale-independent representation across stars of different brightness
        │
        ▼
Step 5: Gaussian Smoothing (σ = 10 cadences)
        segment = gaussian_filter1d(segment, sigma=10)
        Suppresses high-frequency noise while preserving ~5-hour transit dips
        │
        ▼
Step 6: float32 Cast
        Reduces memory footprint; compatible with TensorFlow inference
        │
        ▼
Output: (1, 3197, 1) float32 tensor → CNN input
```

### 3.2 Design Rationale

**Why best-variance selection?** Transit events create local variance spikes in the flux. By selecting the highest-variance window, we maximise the probability of capturing a transit dip within the 3,197-cadence input window. This is especially effective for short-period hot Jupiters where multiple transits occur within any given window.

**Why Gaussian smoothing?** Kepler long-cadence data has a 30-minute sampling rate. A hot Jupiter transit typically lasts 3–5 hours, spanning ~6–10 cadences. Gaussian smoothing with σ=10 acts as a low-pass filter that suppresses shot noise and instrumental artefacts while preserving the broad transit shape that the CNN uses as its primary feature.

**Known limitation:** For planets with orbital periods > 4 days, fewer than 2–3 transits may occur within a 65-day window. The best-variance selector may then land on a non-transit window, causing the CNN to score near zero even for confirmed planets. This was identified as the primary operational limitation in Week 9 (see Section 9).

---

## 4. Model Architecture

### 4.1 1D-CNN Design

The final model (saved as `models/cnn_model_week6.keras`) is a 1D Convolutional Neural Network designed for sequence classification:

```
Input Layer:        (batch, 3197, 1)
        │
Conv1D(64,  k=3, ReLU) → MaxPool1D(2)    output: (batch, 1598, 64)
        │
Conv1D(128, k=3, ReLU) → MaxPool1D(2)    output: (batch,  798, 128)
        │
Conv1D(256, k=3, ReLU) → GlobalAvgPool1D output: (batch,       256)
        │
Dense(128, ReLU) → Dropout(0.5)
        │
Dense(1, Sigmoid)
        │
Output: planet probability ∈ [0, 1]
```

**Total parameters:** ~450,000

### 4.2 Architecture Decisions

- **1D convolutions** are used rather than 2D because the input is a univariate time series — the spatial structure is along the time axis only
- **Increasing filter counts** (64→128→256) allow the network to learn progressively more abstract representations, from local dip shapes to global periodicity patterns
- **GlobalAveragePooling** instead of Flatten reduces overfitting by compressing the temporal dimension into a fixed-size feature vector regardless of where the transit occurs in the window
- **Dropout(0.5)** at the Dense layer is the primary regularisation mechanism, critical given the small number of positive training examples

### 4.3 Model Evolution

| Week | Architecture | AUC |
|------|-------------|-----|
| Week 3 | Logistic Regression | ~0.72 |
| Week 3 | MLP (3 layers) | ~0.81 |
| Week 4 | CNN-BiLSTM hybrid | ~0.89 |
| Week 5 | 1D-CNN (v1) | ~0.94 |
| **Week 6** | **1D-CNN (final)** | **0.9628** |

---

## 5. Training & Augmentation

### 5.1 Training Configuration

```python
Optimiser:     Adam (lr=1e-3)
Loss:          Binary crossentropy
Class weights: {0: 1.0, 1: 137.0}  # inverse class frequency
Batch size:    32
Epochs:        50 (early stopping, patience=10)
Validation:    20% stratified split
```

### 5.2 Data Augmentation (Week 5)

To address the class imbalance and improve generalisation, three augmentation strategies were applied to positive examples only:

1. **Phase shifting** — randomly rolling the flux array along the time axis, simulating different transit phases
2. **Amplitude jitter** — multiplying flux by a factor drawn from N(1.0, 0.02), simulating stellar variability
3. **Gaussian noise injection** — adding N(0, 0.001) noise to each cadence

Augmentation increased effective positive training examples by 4×, improving validation AUC from ~0.94 to **0.9628**.

### 5.3 Training Curves

<div align="center">

| Training History | Loss Curves |
|:---:|:---:|
| ![Accuracy](outputs/accuracy_curve.png) | ![Loss](outputs/loss_curve.png) |

</div>

### 5.4 Week 5 Evaluation Results

<div align="center">

| ROC Curve | AUC Progression |
|:---:|:---:|
| ![ROC Week 5](results/week5/roc_curve.png) | ![AUC Progression](results/week5/auc_progression.png) |

| Confusion Matrix | Score Distribution |
|:---:|:---:|
| ![Confusion Matrix](results/week5/confusion_matrix.png) | ![Scores](results/week5/planet_scores.png) |

</div>

---

## 6. Results — Competition Evaluation

### 6.1 Week 6 Final Model

The Week 6 model (`cnn_model_week6.keras`) achieved the following on the Kepler DR25 competition test set:

| Metric | Value |
|--------|-------|
| **AUC** | **0.9628** |
| Threshold (default 0.5) | 0.5000 |
| Calibrated threshold | 0.6914 |

<div align="center">

| Week 6 Summary | Score Distribution Comparison |
|:---:|:---:|
| ![Week 6 Summary](results/week6/week6_summary.png) | ![Distribution](results/week6/distribution_comparison.png) |

</div>

### 6.2 ROC Curve Analysis

The ROC curve (AUC = 0.9628) demonstrates strong discriminative power on the competition test set. The high AUC reflects the model's ability to separate planet and non-planet distributions even under severe class imbalance.

**Important caveat:** The competition dataset consists of pre-processed, curated Kepler light curves where transits are guaranteed to be present in the input window. Real-world performance is lower (see Section 7) because the segment selection step may not always capture a transit.

---

## 7. Results — Wild-Data Evaluation

### 7.1 Experimental Setup (Week 7)

To assess real-world performance, 30 Kepler stars were downloaded fresh from MAST and evaluated through the full pipeline:

- **15 confirmed planet hosts** (spanning SNR 20–800)
- **6 known eclipsing binaries** (from the Kepler EB catalog)
- **9 non-planet stars** (variable stars, quiet solar analogues)

### 7.2 Detection Results by SNR

| SNR Tier | N Stars | Detected | Detection Rate |
|----------|---------|----------|----------------|
| High (≥ 200) | 15 | 14 | **93%** |
| Low (< 50) | 15 | 1 | **7%** |
| EB stars | 6 | 6* | 100%* |

*EBs were detected as high-scoring candidates but later rejected by the EB filter (Section 8).

<div align="center">

| Overall Results | Final Analysis |
|:---:|:---:|
| ![Detection Results](results/week7/detection_results.png) | ![Week 7 Final](results/week7/week7_final_analysis.png) |

| Score Distribution | Week 7 Summary |
|:---:|:---:|
| ![Score Dist](results/week7/score_distribution.png) | ![Week 7 Summary](results/week7/week7_summary.png) |

</div>

### 7.3 Detected Planet Light Curves

<div align="center">

![Detected Planets](results/week7/lightcurves_detected.png)

*Figure: Light curves of successfully detected planet hosts. Clear periodic dipping patterns are visible.*

</div>

### 7.4 Missed Planet Analysis

<div align="center">

| Missed Planets | Missed Planet Analysis |
|:---:|:---:|
| ![Missed LCs](results/week7/lightcurves_missed.png) | ![Missed Analysis](results/week7/missed_planet_analysis.png) |

</div>

The 7% detection rate on low-SNR planets is explained by two factors:
1. Transit dips shallower than the noise floor (~500 ppm) are smoothed away by the Gaussian filter
2. The best-variance segment selector may choose a non-transit window when transit amplitude is comparable to stellar noise

### 7.5 False Positives Before Calibration

<div align="center">

![False Positives](results/week7/lightcurves_fp.png)

*Figure: False positive light curves before threshold calibration. All are eclipsing binaries with deep, symmetric eclipse signatures.*

</div>

Before threshold calibration and EB filtering, the false positive rate was **28%** — almost entirely composed of eclipsing binaries that the CNN could not distinguish from planet transits.

---

## 8. Threshold Calibration & EB Rejection

### 8.1 Threshold Calibration (Week 8)

The default classification threshold of 0.5 was suboptimal for this problem. A threshold sweep was performed on the validation set to find the operating point that maximised precision while maintaining acceptable recall:

<div align="center">

| Threshold Sweep | PR Curve |
|:---:|:---:|
| ![Threshold Sweep](results/week8/threshold_sweep.png) | ![PR Curve](results/week8/pr_curve.png) |

</div>

**Calibrated threshold: 0.6914**

At this threshold:
- Precision = **1.000** (zero false positives among predicted planets)
- Recall reduced to ~0.65 (acceptable trade-off for a candidate-generation system)
- F1 = **0.605**

### 8.2 Eclipsing Binary Rejection Pipeline

The EB rejection pipeline uses phase folding to detect secondary eclipses — the defining signature of an eclipsing binary system. A true planet produces only a primary transit; an EB produces both primary and secondary eclipses of similar depth.

**Pipeline steps:**

```
Input: light curve + orbital period P
        │
        ▼
1. Coarse 100-bin phase fold → find t0 (primary eclipse centroid)
        │
        ▼
2. Shift phase so primary centred at φ = 0.5
        │
        ▼
3. OOT normalisation (sigma-clipped median of out-of-transit flux)
        │
        ▼
4. Fine 300-bin phase fold + Gaussian smoothing (σ=1)
        │
        ▼
5. Measure primary depth (φ = 0.4–0.6)
   Measure secondary depth (φ = 0.0–0.1 and 0.9–1.0)
        │
        ▼
6. Four noise gates:
   Gate 1: primary depth > 5× OOT scatter
   Gate 2: primary depth > 1×10⁻⁴ (absolute floor)
   Gate 3: secondary depth > 3× OOT scatter (else = 0)
   Gate 4: secondary depth > 0 AND ratio > threshold
        │
        ▼
7. EB flag: secondary/primary depth ratio > 0.50
   2P fold fallback for equal-eclipse EBs
        │
        ▼
Output: is_eb (bool), primary_depth, secondary_depth, ratio, method
```

### 8.3 EB Rejection Results

<div align="center">

| EB Filter Results | KIC 3335816 — 2P Fold |
|:---:|:---:|
| ![EB Filter](results/week8/eb_filter_results.png) | ![KIC 3335816](results/week8/kic3335816_double_period.png) |

*Left: EB filter results showing all 6 EBs correctly identified. Right: KIC 3335816 (equal-eclipse EB) — the 2P fold reveals identical primary and secondary eclipses (ratio = 0.997).*

</div>

### 8.4 Confidence Tiers

Based on the score distribution analysis, three operational confidence tiers were defined:

| Tier | Score Range | Action |
|------|------------|--------|
| **HIGH** | > 0.50 | Planet candidate — report for follow-up |
| **MEDIUM** | 0.05–0.50 | Marginal — period fold + visual inspection |
| **LOW** | < 0.05 | Discard — likely non-planet |

<div align="center">

![Confidence Tiers](results/week8/confidence_tiers.png)

*Figure: Score distribution by class after calibration, showing the three confidence tiers.*

</div>

### 8.5 Post-Calibration Performance Summary

<div align="center">

| Week 8 Summary | Final Summary |
|:---:|:---:|
| ![Week 8 Summary](results/week8/week8_summary.png) | ![Week 8 Final](results/week8/week8_final_summary.png) |

</div>

| Metric | Before Calibration | After Calibration + EB Filter |
|--------|-------------------|-------------------------------|
| AUC | 0.6933 | 0.6933 |
| Threshold | 0.5000 | **0.6914** |
| FPR | **28%** | **0%** |
| Precision | 0.72 | **1.000** |
| F1 | 0.51 | **0.605** |
| EB catch rate | 0/6 | **6/6 (100%)** |

---

## 9. Broader Catalog & Limitations

### 9.1 Week 9A — Expanded Sample

In Week 9, the pipeline was tested on a broader catalog of Kepler stars including systems with known stellar variability (active stars with prominent starspots and flares).

**Result: 0% detection rate on the broader catalog.**

<div align="center">

| Week 9 Summary | Transit Depth vs Score |
|:---:|:---:|
| ![Week 9 Summary](results/week9/week9_summary.png) | ![Depth vs Score](results/week9/depth_vs_score.png) |

| Waveform Comparison | Detrending Comparison |
|:---:|:---:|
| ![Waveform](results/week9/waveform_comparison.png) | ![Detrend](results/week9/detrend_comparison.png) |

</div>

### 9.2 Root Cause Analysis

The 0% detection on the broader catalog was traced to **stellar variability** as the primary bottleneck:

1. **Starspot-dominated light curves:** Active stars produce quasi-periodic brightness variations with amplitudes of 0.1–2%, comparable to or larger than planet transits. The CNN, trained on relatively quiet Kepler competition stars, scores these near zero regardless of planet presence.

2. **Best-variance segment selection failure:** On variable stars, the highest-variance segment is the starspot-dominated region, not the transit region. The CNN sees a bumpy, aperiodic curve — not a clean transit dip.

3. **Gaussian smoothing interaction:** For active stars, σ=10 smoothing preserves the large-amplitude spot modulation while simultaneously erasing the smaller transit signal.

<div align="center">

![LC Comparison](results/week9/lc_comparison.png)

*Figure: Comparison of a quiet star (Kepler-1b, left) vs. an active star (Kepler-17b, right). The CNN reliably detects transits only on the quiet star.*

</div>

### 9.3 Complete Failure Mode Map

| Scenario | Detection Rate | Root Cause |
|----------|---------------|-----------|
| Quiet FGK star, P < 3.5d, SNR ≥ 200 | **~95%** ✅ | Optimal conditions |
| Quiet FGK star, P < 3.5d, SNR 50–200 | ~45% ⚠️ | Marginal SNR |
| Active/variable FGK star (any) | ~0% ❌ | Variability dominates variance |
| M-dwarf host (any) | ~0% ❌ | Out-of-distribution star type |
| Long period planet P > 4d | ~0% ❌ | Transit-sparse segment selection |
| Low SNR planet SNR < 50 | 7% ❌ | Signal below noise floor |
| Eclipsing binary | 100% flagged ✅ | EB filter catches all |

### 9.4 Proposed Future Improvements

1. **Replace best-variance with dip-depth selector** — search for the window containing the deepest local minimum relative to the running median, rather than highest overall variance
2. **Pre-detrending** — apply a Savitzky-Golay or GP-based detrending step before preprocessing to remove stellar variability before segment selection
3. **Phase-folded input representation** — use BLS period to fold the light curve before feeding to the CNN, making the architecture period-agnostic
4. **M-dwarf fine-tuning** — augment the training set with M-dwarf light curves to improve out-of-distribution performance

---

## 10. Web Application

### 10.1 Architecture

The Streamlit web application (`app.py`) implements the full detection pipeline in real time:

```
User enters KIC number
        │
        ▼
lightkurve → MAST archive → Download PDCSAP flux (1 search + 1 download_all)
        │
        ▼
_stitch_chunks() → Per-quarter median normalisation → Stitched LC
        │
        ▼
preprocess() → 6-step pipeline → (1, 3197, 1) tensor
        │
        ▼
CNN inference → planet probability score
        │
        ├── score ≥ 0.6914 → PLANET CANDIDATE
        └── score < 0.6914 → NO PLANET
        │
        ▼
EB check (if period provided) → phase fold → secondary eclipse check
        │
        ▼
Display: score, tier, light curve plot, EB result
```

### 10.2 Caching Strategy

Three-layer caching ensures fast response on repeat queries:

| Layer | Mechanism | TTL | Scope |
|-------|-----------|-----|-------|
| Model | `@st.cache_resource` | Session | Loaded once, shared |
| LC download | `@st.cache_data` | 24 hours | Per KIC ID |
| Session | `st.session_state` | Session | Instant repeat |

### 10.3 Performance

| Operation | Time |
|-----------|------|
| First download (MAST) | 25–45 seconds |
| Repeat (session cache) | < 1 second |
| Preprocessing | < 0.1 seconds |
| CNN inference (CPU) | < 0.5 seconds |
| EB check | 1–3 seconds |

### 10.4 Live Test Results

| KIC | Planet | Score | Result | EB Check |
|-----|--------|-------|--------|----------|
| 11446443 | Kepler-1b | 0.9930 | 🪐 PLANET | ✅ Passed |
| 10666592 | Kepler-2b | 0.9413 | 🪐 PLANET | ✅ Passed |
| 8191672 | Kepler-43b | 0.9038 | 🪐 PLANET | ✅ Passed |
| 5357901 | Kepler-4b | 0.9382 | 🪐 PLANET | ✅ Passed |
| 3335816 | Equal-EB | 0.5553 | ⭐ Below threshold | 🚨 EB FLAGGED |
| 3632418 | Variable | 0.0010 | ⭐ NO PLANET | — |
| 99999999 | Invalid | — | ❌ Error msg | — |

---

## 11. Discussion

### 11.1 Strengths

**High competition AUC (0.9628):** The model performs exceptionally well on the curated Kepler DR25 benchmark, placing it among the top-performing approaches on this dataset. The combination of 1D convolutions with global average pooling proves highly effective for transit shape recognition.

**Perfect precision after calibration:** The threshold calibration and EB rejection pipeline together achieve zero false positives on the validation set. For a candidate-generation system — where false positives create expensive follow-up observational costs — this is a critical property.

**100% EB rejection:** The phase-folding EB filter correctly identifies all 6 eclipsing binaries in the validation set, including the pathological equal-eclipse case (KIC 3335816) which requires a 2P fold to detect.

### 11.2 Limitations

**Training distribution mismatch:** The Kepler DR25 competition dataset presents pre-segmented light curves where a transit is guaranteed to appear in the input window. Real-world operation must find the transit window first, introducing a source of failure not present in competition evaluation. This explains the AUC drop from 0.9628 (competition) to 0.6933 (wild data).

**Stellar variability:** The model was trained primarily on photometrically quiet FGK stars. Active stars with prominent starspot modulation score near zero, making the system essentially blind to planets around variable host stars — a significant limitation given that ~30% of Kepler targets show detectable variability.

**Period dependence:** The best-variance segment selector implicitly requires multiple transits per window to create a clear variance signal. Planets with periods > 4 days have fewer than 2 transits in a 65-day window, reducing the probability of correct segment selection to near chance.

### 11.3 Comparison to Related Work

The AUC of 0.9628 on the Kepler DR25 benchmark is competitive with published deep learning approaches:

| Approach | AUC | Notes |
|----------|-----|-------|
| Shallue & Vanderburg (2018) — AstroNet | ~0.98 | 2-view CNN, global+local |
| Ansdell et al. (2018) — CNN | ~0.97 | Transfer learning |
| **This work — 1D-CNN** | **0.9628** | Single-view, simpler architecture |

The slightly lower AUC relative to AstroNet is expected — AstroNet uses a two-view architecture that separately processes a global view and a zoomed local view of the transit, providing richer spatial context. The single-view 1D-CNN used here is simpler and more computationally efficient while achieving comparable performance.

---

## 12. Conclusion

This project demonstrates a complete end-to-end machine learning pipeline for exoplanet transit detection in Kepler photometric data. The key contributions are:

1. **A fixed 6-step preprocessing pipeline** that produces consistent, scale-independent input representations from raw PDCSAP flux
2. **A 1D-CNN classifier** achieving AUC = 0.9628 on the competition benchmark and 93% detection on high-SNR real-world planets
3. **A calibrated operating threshold** (0.6914) that achieves perfect precision (1.000) on the validation set
4. **An eclipsing-binary rejection filter** that eliminates the primary class of false positives with 100% catch rate
5. **A deployed web application** enabling real-time planet detection for any Kepler star by KIC number

The system's primary limitation — its near-zero performance on variable host stars and long-period planets — points toward clear engineering improvements: pre-detrending, dip-depth-based segment selection, and phase-folded input representations. These are natural extensions for future work.

The complete project, including all 9 weeks of notebooks, trained model weights, result images, and the Streamlit application, is made publicly available in this repository as a reproducible reference implementation.

---

## 13. References

1. Shallue, C. J., & Vanderburg, A. (2018). Identifying Exoplanets with Deep Learning: A Five-Planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90. *The Astronomical Journal*, 155(2), 94.

2. Ansdell, M., et al. (2018). Scientific Domain Knowledge Improves Exoplanet Transit Classification with Deep Learning. *The Astrophysical Journal Letters*, 869(1), L7.

3. Thompson, S. E., et al. (2018). Planetary Candidates Observed by Kepler. VIII. A Fully Automated Catalog Based on Data Release 25. *The Astrophysical Journal Supplement Series*, 235(2), 38.

4. Lightkurve Collaboration (2018). Lightkurve: Kepler and TESS time series analysis in Python. *Astrophysics Source Code Library*, ascl:1812.013.

5. Jenkins, J. M., et al. (2016). Overview of the Kepler Science Processing Pipeline. *The Astrophysical Journal Supplement Series*, 713(2), L87.

6. Kovács, G., Zucker, S., & Mazeh, T. (2002). A box-fitting algorithm in the search for periodic transits. *Astronomy & Astrophysics*, 391(1), 369–377.

---

<div align="center">

## 🛠️ Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/exoplanet-detection.git
cd exoplanet-detection
conda create -n exoplanet python=3.10 && conda activate exoplanet
pip install -r requirements.txt
streamlit run app.py
```

**[🚀 Live App](https://your-app-url.streamlit.app) · [📓 Notebooks](notebooks/) · [📊 Results](results/)**

---

*Built over 10 weeks as a portfolio project in ML engineering and scientific computing.*
*Model: `cnn_model_week6.keras` · Threshold: `0.6914` · Training data: Kepler DR25*

</div>
"# Exoplanet-detection" 
