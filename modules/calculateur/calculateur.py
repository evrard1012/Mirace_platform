# ------------------------------------------------------------------
# 0) Imports nécessaires (si pas déjà faits tout en haut du script)
# ------------------------------------------------------------------
import numpy as np
import pandas as pd
from tslearn.metrics import soft_dtw

# ------------------------------------------------------------------
# 1) Distance Soft-DTW multivariée
#    Somme des Soft-DTW univariés (une par variable)
# ------------------------------------------------------------------
GAMMA = 1.0                     # identique à Profilage 2

def sd_distance(seq_a: np.ndarray, seq_b: np.ndarray, gamma: float = GAMMA) -> float:
    """
    seq_a, seq_b : shape (T, n_vars)
    Renvoie la somme des Soft-DTW univariés.
    """
    d = 0.0
    for j in range(seq_a.shape[1]):
        d += soft_dtw(seq_a[:, j], seq_b[:, j], gamma)
    return d


# ------------------------------------------------------------------
# 2) Interpolation « deux étapes » :
#    • α rapproche 2020 du centroïde-2020
#    • β interpole de ce 2020 ajusté vers le centroïde-2021
# ------------------------------------------------------------------
def two_step_interpolation(val_2020: pd.Series,
                           c_2020: pd.Series,
                           c_2021: pd.Series,
                           bounds: pd.DataFrame,
                           eps: float = 0.2,
                           max_iter: int = 30):
    """
    Calcule (val_2020_adj, val_2021_pred, α, β) tels que
    Soft-DTW([2020_adj, 2021_pred], [c2020, c2021]) ≤ eps,
    en minimisant α puis β.

    val_2020 : Series valeurs pays 2020
    c_2020   : Series centroïde 2020
    c_2021   : Series centroïde 2021
    bounds   : DataFrame index=['min','max'], colonnes = mêmes variables
    eps      : tolérance Soft-DTW
    """
    v0 = val_2020.values.astype(float)
    c0 = c_2020.values.astype(float)
    c1 = c_2021.values.astype(float)

    # ----------------  Étape α (ajustement 2020) ----------------
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        alpha = (lo + hi) / 2
        v0p = (1 - alpha) * v0 + alpha * c0
        if sd_distance(v0p.reshape(1, -1), c0.reshape(1, -1)) <= eps:
            hi = alpha
        else:
            lo = alpha
    v0p = (1 - hi) * v0 + hi * c0
    v0p = np.minimum(np.maximum(v0p, bounds.loc["min"].values),
                     bounds.loc["max"].values)

    # ----------------  Étape β (projection 2021) ----------------
    lo, hi2 = 0.0, 1.0
    for _ in range(max_iter):
        beta = (lo + hi2) / 2
        v1 = (1 - beta) * v0p + beta * c1
        v1 = np.minimum(np.maximum(v1, bounds.loc["min"].values),
                        bounds.loc["max"].values)

        if sd_distance(np.vstack([v0p, v1]), np.vstack([c0, c1])) <= eps:
            hi2 = beta
        else:
            lo  = beta
    v1 = (1 - hi2) * v0p + hi2 * c1
    v1 = np.minimum(np.maximum(v1, bounds.loc["min"].values),
                    bounds.loc["max"].values)

    # ----------------  Retour au format Series ------------------
    v0_series = pd.Series(v0p, index=val_2020.index).round(3)
    v1_series = pd.Series(v1 , index=val_2020.index).round(3)

    return v0_series, v1_series, hi, hi2   # val_2020_adj, val_2021_pred, α, β
