"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch


def _masked_mean(h: torch.Tensor, mask_1d: torch.Tensor) -> torch.Tensor:
    m = mask_1d.float().unsqueeze(-1)
    denom = m.sum(dim=0).clamp(min=1.0)
    return (h * m).sum(dim=0) / denom


def _last_real_index(mask_1d: torch.Tensor) -> int:
    nz = mask_1d.nonzero(as_tuple=False)
    return int(nz[-1].item())


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the aggregation below.
    # ------------------------------------------------------------------
    n_layers, _, hidden_dim = hidden_states.shape

    lo, hi = 1, n_layers - 1
    if hi < lo:
        lo, hi = 0, n_layers - 1

    fracs = (0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
    layer_ids: list[int] = []
    for f in fracs:
        li = int(lo + f * (hi - lo))
        li = max(lo, min(li, hi))
        if li not in layer_ids:
            layer_ids.append(li)

    last_pos = _last_real_index(attention_mask)
    parts: list[torch.Tensor] = []

    for li in layer_ids:
        h = hidden_states[li]
        parts.append(_masked_mean(h, attention_mask))
        parts.append(h[last_pos])
    mid = layer_ids[len(layer_ids) // 2]
    h_mid = hidden_states[mid][last_pos]
    h_fin = hidden_states[layer_ids[-1]][last_pos]
    parts.append(h_fin - h_mid)

    h_last = hidden_states[layer_ids[-1]]
    m = attention_mask.float().unsqueeze(-1)
    mu = _masked_mean(h_last, attention_mask)
    centered = (h_last - mu) * m
    var = (centered.pow(2) * m).sum(dim=0) / m.sum(dim=0).clamp(min=1.0)
    parts.append(torch.sqrt(var + 1e-6))

    return torch.cat(parts, dim=0)
    # ------------------------------------------------------------------


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------
    n_layers, _, _ = hidden_states.shape
    last_pos = _last_real_index(attention_mask)
    feats: list[torch.Tensor] = []

    n_real = float(attention_mask.sum().item())
    feats.append(torch.tensor([n_real / 512.0], device=hidden_states.device))

    lo, hi = 1, n_layers - 1
    if hi > lo:
        a = hidden_states[lo][last_pos]
        b = hidden_states[hi][last_pos]
        cos = torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0), dim=1, eps=1e-8
        )
        feats.append(cos)

    m = attention_mask.bool()
    h = hidden_states[-1][m]
    feats.append(torch.tensor([h.norm(p="fro") / (h.numel() ** 0.5 + 1e-8)], device=hidden_states.device))

    return torch.cat(feats, dim=0)
    # ------------------------------------------------------------------


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
