# E-QRGMM: Efficient Generative Metamodeling for Covariate-Dependent Uncertainty Quantification

This repository contains the official implementation of the paper:
**[Efficient Generative Metamodeling for Covariate-Dependent Uncertainty Quantification](https://arxiv.org/abs/2601.19256)**.

The focus is on conditional distribution modeling, together with estimation for estimands of interest (mean/quantiles/tail probabilities) and bootstrap-based uncertainty quantification.

## Benchmarks

- **Synthetic data**: conditional Normal / HalfNormal / Student-t distributions.
- **Inventory problem**: learning the distribution of inventory costs.

Evaluation metrics: KS statistic, Wasserstein distance, as well as bootstrap CI length and coverage.

## Requirements

We recommend using Python >= 3.6, <= 3.8 (mainly to ensure compatibility with the [pyqreg](https://github.com/mozjay0619/pyqreg) package).

To install requirements:

```bash
pip install -r requirements.txt
```

## Training and evaluation

### Synthetic Data

**Figure 1 & 2 (Ablation study)**

```bash
cd "synthetic data/Figure 1 and 2"
python ablation_study.py
```

**Table 1 (E-QRGMM)**
```bash
cd "synthetic data/Table 1/E-QRGMM"
python E-QRGMM_normal.py
python E-QRGMM_halfnormal.py
python E-QRGMM_t.py
```

**Table 1 (Deep generative models: GAN / DDIM / RectFlow)**

```bash
cd "synthetic data/Table 1/other generative models"
python try_normal.py
python try_halfnormal.py
python try_t.py
```

**Table 2 (Bootstrap CI: E-QRGMM)**
```bash
cd "synthetic data/Table 2/E-QRGMM"
python ci_normal.py
python ci_halfnormal.py
python ci_t.py
```

**Table 2 (Bootstrap CI: RectFlow)**
```bash
cd "synthetic data/Table 2/RectFlow"
python flow_normal.py
python flow_halfnormal.py
python flow_t.py
```

### Inventory Problem

**Table 3 (E-QRGMM)**

```bash
cd "inventory problem/Table 3/E-QRGMM"
python inventory_distribution.py # Generate a reference set from the ground-truth cost distribution
python data_generation.py # Generate training datasets
python EQRGMM_KSWD.py
```

**Table 3 (Deep generative models: GAN / DDIM / RectFlow)**

```bash
cd "inventory problem/Table 3/other generative models"
python inventory_distribution.py # Generate a reference set from the ground-truth cost distribution
python data_generation.py # Generate training datasets
python try_inventory.py
```

**Table 4 (Bootstrap CI: E-QRGMM)**

```bash
cd "inventory problem/Table 4/E-QRGMM"
python data_generation.py # Generate training datasets
python ci_im.py
```

**Table 4 (Bootstrap CI: RectFlow)**

```bash
cd "inventory problem/Table 4/RectFlow"
python data_generation.py # Generate training datasets
python flow_inventory.py
```

## Results

Our model achieves the following performance on:

**Synthetic Datasets**

| Models   | Normal KS  | Normal WD  | Normal Time(s) | Halfnormal KS | Halfnormal WD | Halfnormal Time(s) | Student's t KS | Student's t WD | Student's t Time(s) |
| -------- | ---------- | ---------- | -------------- | ------------- | ------------- | ------------------ | -------------- | -------------- | ------------------- |
| GAN      | 0.5064     | 1.5522     | 12.957         | 0.7067        | 1.8117        | 12.990             | 0.4098         | 1.1811         | 12.843              |
| DDIM     | 0.0534     | 0.1672     | 18.646         | 0.0953        | 0.1623        | 18.660             | 0.0465         | 0.1911         | 18.545              |
| RectFlow | 0.0436     | 0.1270     | 18.427         | 0.0856        | 0.1313        | 18.428             | 0.0370         | 0.1675         | 18.394              |
| E-QRGMM  | **0.0110** | **0.0281** | **0.0960**     | **0.0113**    | **0.0148**    | **0.0879**         | **0.0110**     | **0.0528**     | **0.0990**          |

| Estimand       | Model    | Normal Coverage | Normal Width | Halfnormal Coverage | Halfnormal Width | Student's t Coverage | Student's t Width |
| -------------- | -------- | --------------- | ------------ | ------------------- | ---------------- | -------------------- | ----------------- |
| Mean           | E-QRGMM  | 0.89            | **0.0481**   | 0.90                | **0.0296**       | 0.89                 | **0.0612**        |
| Mean           | RectFlow | 1.00            | 0.4187       | 1.00                | 0.3396           | 1.00                 | 0.4495            |
| Quantile       | E-QRGMM  | 0.90            | **0.0695**   | 0.90                | **0.0551**       | 0.91                 | **0.0830**        |
| Quantile       | RectFlow | 1.00            | 0.4598       | 0.95                | 0.3779           | 1.00                 | 0.5015            |
| Survival Func. | E-QRGMM  | 0.90            | **0.0144**   | 0.90                | **0.0143**       | 0.91                 | **0.0146**        |
| Survival Func. | RectFlow | 1.00            | 0.0985       | 0.95                | 0.1062           | 1.00                 | 0.0941            |

**Inventory Dataset**

| Model    | KS         | WD         | Time(s)    |
| -------- | ---------- | ---------- | ---------- |
| GAN      | 0.2018     | 4.0173     | 13.006     |
| DDIM     | 0.0428     | 1.0598     | 18.635     |
| RectFlow | 0.0293     | 0.6676     | 18.444     |
| E-QRGMM  | **0.0132** | **0.2359** | **0.1219** |

| Estimand       | Model    | Coverage | Width      |
| -------------- | -------- | -------- | ---------- |
| Mean           | E-QRGMM  | 0.90     | **0.5505** |
| Mean           | RectFlow | 1.00     | 2.2306     |
| Quantile       | E-QRGMM  | 0.89     | **0.7992** |
| Quantile       | RectFlow | 1.00     | 2.4798     |
| Survival Func. | E-QRGMM  | 0.90     | **0.0214** |
| Survival Func. | RectFlow | 1.00     | 0.0658     |
