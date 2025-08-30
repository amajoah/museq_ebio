# fusion.py
# 기존 3지표 융합(레거시) 유지 + 품질인지형(quality-aware) 2모달 융합 추가
from __future__ import annotations
from typing import Iterable, Dict, Tuple
import numpy as np

def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def minmax_scale(x: Iterable[float]) -> np.ndarray:
    """
    간단 Min-Max 스케일러. 입력이 상수이거나 NaN이면 0으로 반환.
    """
    arr = np.asarray(list(x), dtype=float)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def risk_fusion(
    cnn_score: float,
    resid_score: float,
    if_score: float,
    w: Tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> Tuple[int, Dict[str, float]]:
    """
    [레거시] 3지표(영상, 잔차, IF) 고정가중 융합.
    입력 점수는 0~1 범위를 권장(안전하게 0~1로 클립).
    반환: (0~100 정수 위험도, 모듈별 정규화 점수 딕셔너리)
    """
    s1 = _clip01(cnn_score)
    s2 = _clip01(resid_score)
    s3 = _clip01(if_score)

    w1, w2, w3 = w
    s = w1 + w2 + w3
    if s <= 0:
        w1 = w2 = w3 = 1 / 3
    else:
        w1, w2, w3 = w1 / s, w2 / s, w3 / s

    risk01 = w1 * s1 + w2 * s2 + w3 * s3
    risk = int(round(risk01 * 100))
    parts = {"cnn": s1, "resid": s2, "iforest": s3}
    return risk, parts

def dynamic_threshold(
    risk_history: Iterable[float],
    base: int = 60,
    k: float = 1.2,
    clip: Tuple[int, int] = (40, 90),
) -> int:
    """
    IQR 기반 동적 임계치.
    - risk_history: 과거 위험도 시퀀스(0~100 권장)
    - base: 최저 기준 임계
    - k: 민감도(클수록 임계↑)
    - clip: [하한, 상한] 결과 범위 클립
    """
    arr = np.asarray(list(risk_history), dtype=float)
    if arr.size < 10 or not np.isfinite(arr).all():
        thr = base
    else:
        q1, q3 = np.quantile(arr, [0.25, 0.75])
        iqr = q3 - q1
        thr = max(base, q3 + k * iqr)
    lo, hi = clip
    return int(np.clip(thr, lo, hi))

def risk_fusion_qaware(
    camera_risk01: float,
    sensor_risk01: float,
    weights: Dict[str, float],
) -> int:
    """
    [신규] 품질인지형 2모달 융합.
    - camera_risk01: 카메라 측 위험/확률 (0~1)
    - sensor_risk01: 센서 측 위험/확률 (0~1)
    - weights: {"camera": w_cam, "sensor": w_sen} (합 1 아니어도 자동 정규화)
               quality_weighting.compute_quality_weights()의 출력 사용 권장.
    반환: 0~100 정수 위험도
    """
    wc = float(weights.get("camera", 0.5))
    ws = float(weights.get("sensor", 0.5))
    s = wc + ws
    if s <= 0:
        wc = ws = 0.5
        s = 1.0
    wc /= s
    ws /= s

    c = _clip01(camera_risk01)
    r = _clip01(sensor_risk01)
    fused01 = wc * c + ws * r
    return int(round(fused01 * 100))
