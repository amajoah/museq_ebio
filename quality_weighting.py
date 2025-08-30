import math, numpy as np

def variance_of_laplacian(img3chw: np.ndarray) -> float:
    g = img3chw.mean(axis=0)
    k = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    pad = np.pad(g, ((1,1),(1,1)), mode="reflect")
    h,w = g.shape; out = np.zeros_like(g)
    for i in range(h):
        for j in range(w):
            out[i,j] = np.sum(pad[i:i+3, j:j+3] * k)
    return float(out.var())

def camera_quality_from_frame(frame_3chw: np.ndarray, cls_probs: np.ndarray):
    blur = variance_of_laplacian(frame_3chw)
    blur_norm = min(blur/50.0, 1.0)
    bright = float(np.clip(frame_3chw.mean(), 0.0, 1.0))
    dark_ratio = float((frame_3chw.mean(axis=0) < 0.08).mean())
    occ = 1.0 - dark_ratio
    det_conf = float(cls_probs.max())
    q = max(0.0, min(1.0, 0.40*det_conf + 0.25*blur_norm + 0.20*bright + 0.15*occ))
    return {"q_score": q, "blur_metric": blur, "brightness": bright, "occlusion_ratio": 1.0-occ, "det_conf_mean": det_conf}

def _softmax(xs):
    m = max(xs); ex = [math.exp(x-m) for x in xs]; s = sum(ex); return [e/s for e in ex]

def compute_quality_weights(q_cam: float, q_sen: float, softness: float=6.0, floor: float=0.1):
    logits = [softness*(q_cam-0.5), softness*(q_sen-0.5)]
    raw = _softmax(logits)
    clipped = [max(floor, r) for r in raw]
    z = sum(clipped)
    return {"camera": clipped[0]/z, "sensor": clipped[1]/z}
