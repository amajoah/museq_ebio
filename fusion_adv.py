# fusion_adv.py
# 마커×복제 행렬 -> 어텐션 가중 융합 + 베이지안식 신뢰도 보정
import numpy as np
from typing import Dict, List

def replicate_reliability(rep_counts: List[int]) -> float:
    # 복제 일치도/깊이를 0~1로 정규화 (간단 지표)
    reps = np.array(rep_counts, dtype=float)
    if reps.size == 0: return 0.5
    depth = np.clip(reps.mean()/max(reps.max(),1.0), 0, 1)  # 평균/최대
    presence = (reps > 0).mean()
    return float(0.6*presence + 0.4*depth)  # 0~1

def softmax(x):
    x = np.array(x, dtype=float)
    m = x.max() if x.size else 0.0
    ex = np.exp(x - m)
    s = ex.sum() + 1e-9
    return ex / s

def attention_fusion(marker_rep_matrix: Dict[str, List[int]], tau: float = 1.0) -> float:
    """
    marker_rep_matrix: {marker_name: [replicate_counts...]}
    반환: 0~1 위험 신호 스코어 (마커별 존재/강도 + 복제 신뢰도 기반)
    """
    if not marker_rep_matrix: return 0.0
    markers = list(marker_rep_matrix.keys())
    rels = []
    vals = []
    for m in markers:
        reps = marker_rep_matrix[m]
        rel = replicate_reliability(reps)               # 0~1
        val = np.tanh(np.mean(reps)/10.0)               # 신호 강도(0~1 근사)
        rels.append(rel); vals.append(val)
    # 어텐션 점수 = rel/tau + val
    logits = (np.array(rels) / max(tau,1e-6)) + np.array(vals)
    attn = softmax(logits)
    fused = float((attn * np.array(vals)).sum())        # 가중 합
    return fused  # 0~1

def bayes_reliability_blend(p_raw: float, rel: float, prior=0.1) -> float:
    """
    p_raw: 검출 신호 기반 확률(0~1), rel: 샘플 신뢰도(0~1)
    신뢰도가 높을수록 p_raw에 가깝게, 낮을수록 prior로 수축(shrinkage)
    """
    w = 0.2 + 0.8*rel    # 최소 0.2 가중
    return float(np.clip(w*p_raw + (1-w)*prior, 1e-4, 1-1e-4))
