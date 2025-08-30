import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

def baseline_als(y, lam=1e4, p=0.01, niter=10):
    """
    Asymmetric Least Squares baseline correction.
    y: 1D array
    """
    y = np.asarray(y, dtype=float)
    L = len(y)
    D = np.diff(np.eye(L), 2)
    w = np.ones(L)
    for _ in range(niter):
        W = np.diag(w)
        Z = W + lam * D.T @ D
        # Solve Z z = W y
        z = np.linalg.solve(Z, W @ y)
        w = p * (y > z) + (1-p) * (y <= z)
    return z

def ttp_and_slope(time, corrected, frac=0.2):
    """
    Time-to-positive: first time index where corrected exceeds baseline + frac*(max-min).
    Returns index and approximate local slope via finite diff.
    """
    corrected = np.asarray(corrected, dtype=float)
    time = np.asarray(time, dtype=float)
    lo, hi = corrected.min(), corrected.max()
    thr = lo + frac*(hi-lo)
    idx = None
    for i,v in enumerate(corrected):
        if v >= thr:
            idx = i; break
    # slope near idx
    slope = np.nan
    if idx is not None and 1 <= idx < len(corrected)-1:
        slope = (corrected[idx+1] - corrected[idx-1]) / (time[idx+1] - time[idx-1] + 1e-9)
    return idx, slope, thr

class BiosensorQuant:
    """
    Standard-curve-like regressor: features = [TTP, slope, max, AUC]
    If no training given, falls back to heuristic mapping.
    """
    def __init__(self):
        self.model = GradientBoostingRegressor(random_state=42)
        self.trained = False

    def featurize(self, time, corrected):
        ttpi, slope, thr = ttp_and_slope(time, corrected, frac=0.2)
        mx = float(np.max(corrected))
        auc = float(np.trapz(np.maximum(corrected,0), time))
        ttpv = float(time[ttpi]) if ttpi is not None else float(time[-1]*1.25)
        return np.array([ttpv, slope if np.isfinite(slope) else 0.0, mx, auc], dtype=float)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.trained = True

    def predict(self, feats):
        feats = np.atleast_2d(feats)
        if self.trained:
            y = self.model.predict(feats)
            return float(y[0])
        # Heuristic fallback: higher conc → earlier TTP, larger slope/mx/auc
        ttp, slope, mx, auc = feats[0]
        score = ( (1.0 / max(ttp,1e-3)) * 0.5 +
                  max(slope,0) * 0.2 +
                  mx * 0.15 +
                  (auc / max(ttp,1.0)) * 0.15 )
        return float(score)
