import numpy as np

def qc_checks(df):
    recent = df.tail(48)
    rng_mean = float(np.mean([
        (recent["temp"].between(-2, 40)).mean(),
        (recent["flow"].between(0, 500)).mean(),
        (recent["chl"].between(0, 500)).mean(),
        (recent["do"].between(0, 20)).mean(),
    ]))

    def spike_rate(x, thr=6.0):
        d = np.diff(x.values); mad = np.median(np.abs(d - np.median(d))) + 1e-6
        return float(np.mean(np.abs(d) > thr*mad))
    sr = max(spike_rate(recent["temp"]), spike_rate(recent["flow"]),
             spike_rate(recent["chl"]),  spike_rate(recent["do"]))
    spike_ok = (sr < 0.07)

    def plateau_rate(x, eps=1e-3):
        d = np.abs(np.diff(x.values)); return float(np.mean(d < eps))
    pr = max(plateau_rate(recent["temp"]), plateau_rate(recent["flow"]),
             plateau_rate(recent["chl"]),  plateau_rate(recent["do"]))
    plateau_ok = (pr < 0.9)

    gap_ratio = float(recent.isna().mean().mean())
    gap_ok = (gap_ratio < 0.05)

    passed = sum([rng_mean > 0.95, spike_ok, plateau_ok, gap_ok])
    qc_pass_rate = passed/4.0
    info = dict(range_mean=rng_mean, spike_rate_max=sr, plateau_rate_max=pr, data_gap_ratio=gap_ratio)
    return qc_pass_rate, info
