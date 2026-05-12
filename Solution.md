# Solution Report

---

## 1. Reproducibility instructions

### Environment

- Python: 3.10 or newer
- Hardware: a GPU is basically required if you want reasonable runtime. CPU works but was very slow in my run (feature extraction over the training set took on the order of hours).

Install dependencies and run:

```bash
pip install -r requirements.txt
python solution.py
```

---

## 2. Final solution description

### Final approach

#### Feature extraction (`aggregation.py`)

Each sample gives a hidden-state tensor of shape `(n_layers, seq_len, hidden_dim)`. I take six layer indices spread between layer 1 and the last layer using fixed depth fractions.

For each of those layers I concatenate:

- a masked mean over all non-padding positions (a summary of the whole visible sequence), and  
- the last real token vector (end of the unpadded part, within `max_length=512`).

On top of that I append:

- depth drift: last-token difference between the middle selected layer and the deepest one (`h_final − h_mid`), and  
- per-dimension masked standard deviation over tokens on the deepest selected layer (how much the activations spread along the sequence).

There is also `extract_geometric_features` for optional extras; it only activates if you set `USE_GEOMETRIC = True` in `solution.py`. I left the default off so the pipeline stays simpler and faster.

#### Classifier (`probe.py`)

I scale inputs with StandardScaler inside each `fit` (so every fold’s probe fits its own scaler on its training data; the final probe fits on the union of train and validation indices that `solution.py` passes, as designed).

The model is a small MLP: two ReLU hidden blocks with dropout, then one logit. I train with AdamW and BCEWithLogitsLoss, using `pos_weight` to handle class imbalance, gradient clipping, mini-batches, and a simple early stop when the training loss gets very small.

`fit_hyperparameters` walks thresholds on validation predicted probabilities to maximize F1; `evaluate.py` calls that when validation indices exist.

#### Data splitting (`splitting.py`)

I use stratified 5-fold: each fold holds out about 20% of the 689 labelled rows as test, keeping class balance. On the remaining rows I do a stratified train/validation split with `val_frac` around 0.12 so I can tune the threshold without touching that fold’s test labels.

### 2.3 Rationale for these choices


Stratified k-fold on a small dataset gives less noisy metrics.

Weighted BCE, dropout, and AdamW are there to deal with imbalance and with having many features relative to sample size.

### 2.4 What contributed most to improving the metric

Almost all of the lift came from richer aggregation (several depths, two pooling styles, drift, std). The MLP mostly learns a decision boundary on top of that; threshold tuning on validation helps F1 and the discrete metrics you see on the internal test split.

---

## 3. Experiments and failed attempts

- `USE_GEOMETRIC = True` in `solution.py`: the code path exists in `extract_geometric_features`, but I kept it off by default so feature size and runtime stay predictable without another tuning pass.

- Much larger networks or training forever: clear overfitting + and long runs hurt on CPU

- Explicit PCA or other linear dimensionality reduction in `probe.py`: I did not need it in the end; StandardScaler plus the MLP bottleneck was enough for me.
