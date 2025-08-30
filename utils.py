import os, json, numpy as np, matplotlib.pyplot as plt

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def moving_average(x, w=5):
    x = np.asarray(x, dtype=float)
    if len(x) < w: return x.copy()
    c = np.convolve(x, np.ones(w)/w, mode="same")
    return c

def plot_biosensor(time, raw, baseline, corrected, d1, ttp_idx, savepath):
    plt.figure(figsize=(8,5))
    plt.plot(time, raw, label="raw")
    plt.plot(time, baseline, label="baseline", linestyle="--")
    plt.plot(time, corrected, label="corrected")
    plt.plot(time, d1, label="d/dt", alpha=0.6)
    if ttp_idx is not None:
        plt.axvline(time[ttp_idx], color="r", linestyle=":", label=f"TTP ~ {time[ttp_idx]:.1f}")
    plt.xlabel("time"); plt.ylabel("signal")
    plt.legend(); plt.tight_layout()
    plt.savefig(savepath, dpi=160); plt.close()

def barplot_seq(names, rpms, savepath):
    plt.figure(figsize=(7,4))
    idx = np.arange(len(names))
    plt.bar(idx, rpms)
    plt.xticks(idx, names, rotation=20, ha="right")
    plt.ylabel("RPM (reads per million)")
    plt.tight_layout()
    plt.savefig(savepath, dpi=160); plt.close()
