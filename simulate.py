import numpy as np, pandas as pd

DNA = "ACGT"
def rand_dna(n, rng): return "".join(rng.choice(list(DNA), size=n))

def inject_motifs(seq, motifs, rng, max_inserts=3):
    s = list(seq)
    k = rng.integers(1, max_inserts+1)
    for _ in range(k):
        m = rng.choice(motifs)
        pos = rng.integers(0, max(1, len(s)-len(m)))
        s[pos:pos+len(m)] = list(m)
    return "".join(s)

def simulate_fastq(path, n_reads=2000, read_len=400, inject_rate=0.05, targets=None, seed=42):
    rng = np.random.default_rng(seed)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n_reads):
            seq = rand_dna(read_len, rng)
            # with probability inject_rate inject motifs from random target
            if rng.random() < inject_rate and targets:
                tt = rng.choice(targets)
                seq = inject_motifs(seq, tt["motifs"], rng)
            qual = "I"*read_len
            f.write(f"@synthetic_{i}\n{seq}\n+\n{qual}\n")

def simulate_biosensor_csv(path, length=300, dt=1.0, positive=True, seed=42, level=1.0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, length*dt, dt)
    # baseline noise
    y = rng.normal(0.0, 0.05, size=len(t))
    if positive:
        # logistic growth-like signal
        mid = int(0.35*len(t) / max(level,1e-3))
        k = 0.05 * (1.5*level)
        sig = 1.0 / (1.0 + np.exp(-k*(np.arange(len(t))-mid)))
        y += sig
    # trend drift
    y += 0.001*np.arange(len(t))
    df = pd.DataFrame({"time": t, "signal": y})
    df.to_csv(path, index=False)
