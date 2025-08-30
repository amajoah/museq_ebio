import numpy as np
from sklearn.linear_model import LogisticRegression

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def calibrate_logistic(X, y):
    """
    Optional calibration model mapping raw evidence -> event probability.
    X: [[seq_logit, bio_logit]]  y: [0/1]
    """
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)
    return lr

def naive_bayes_fusion(p_seq, p_bio, prior=0.1, corr=0.0):
    """
    Naive Bayes odds product with optional correlation fudge (corr in [-0.5, 0.5]).
    """
    eps = 1e-6
    odds = (p_seq/(1-p_seq+eps)) * (p_bio/(1-p_bio+eps)) * (prior/(1-prior+eps))
    odds *= (1.0 - corr)  # down-weight if correlated
    p = odds / (1.0 + odds)
    return float(np.clip(p, 1e-4, 1-1e-4))

def grade(p, base=0.6, warn=0.75, danger=0.9):
    if p < base: return "정상"
    if p < warn: return "주의"
    if p < danger: return "경계"
    return "위험"
