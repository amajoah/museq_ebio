import argparse, json, yaml, numpy as np, pandas as pd, os
from utils import ensure_dir, save_json, plot_biosensor, barplot_seq
from seq_scan import parse_fastq, scan_targets
from biosensor import baseline_als, ttp_and_slope, BiosensorQuant
from risk import naive_bayes_fusion, grade, sigmoid
from simulate import simulate_fastq, simulate_biosensor_csv

def load_targets(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["targets"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastq", type=str, default=None, help="FASTQ path")
    ap.add_argument("--biosensor", type=str, default=None, help="biosensor CSV path (time,signal)")
    ap.add_argument("--targets", type=str, default="targets.yaml", help="YAML with targets/motifs")
    ap.add_argument("--simulate", action="store_true", help="generate synthetic data")
    ap.add_argument("--out", type=str, default="outputs", help="output directory")
    ap.add_argument("--reads", type=int, default=2000, help="synthetic reads")
    ap.add_argument("--inject", type=float, default=0.05, help="injection rate (0-1)")
    args = ap.parse_args()

    ensure_dir(args.out)

    # 1) Prepare inputs (simulate if requested)
    targets = load_targets(args.targets)

    if args.simulate:
        ensure_dir("sample_data")
        fq = "sample_data/sim.fastq"
        bs = "sample_data/signal.csv"
        simulate_fastq(fq, n_reads=args.reads, inject_rate=args.inject, targets=targets, seed=42)
        simulate_biosensor_csv(bs, length=300, dt=1.0, positive=True, seed=123, level=1.0)
        fastq_path, biosensor_path = fq, bs
    else:
        if not args.fastq or not args.biosensor:
            raise SystemExit("실데이터 사용 시 --fastq 와 --biosensor 둘 다 필요합니다.")
        fastq_path, biosensor_path = args.fastq, args.biosensor

    # 2) Sequencing branch
    seqs = parse_fastq(fastq_path, qmin=10, min_len=200, max_reads=None)
    counts, rpm = scan_targets(seqs, targets)
    names = list(rpm.keys()); rpms = [rpm[k] for k in names]
    barplot_seq(names, rpms, os.path.join(args.out, "seq_barplot.png"))

    # Convert RPM -> probability proxy (logit-like mapping)
    # heuristic: P_seq = sigmoid( a*log1p(rpm_max) + b )
    rpm_max = max(rpms) if rpms else 0.0
    p_seq = float(1.0/(1.0+np.exp(-(0.8*np.log1p(rpm_max) - 2.0))))

    # 3) Biosensor branch
    df = pd.read_csv(biosensor_path)
    t = df["time"].values; y = df["signal"].values
    baseline = baseline_als(y, lam=1e4, p=0.01, niter=10)
    corrected = y - baseline
    d1 = np.gradient(corrected, t)
    ttpi, slope, thr = ttp_and_slope(t, corrected, frac=0.2)
    quant = BiosensorQuant()
    feats = quant.featurize(t, corrected)
    # default: heuristic quant score
    qscore = quant.predict(feats)
    # probability proxy from biosensor: sigmoid(c0 * qscore + c1)
    p_bio = float(1.0/(1.0+np.exp(-(1.5*qscore - 1.0))))

    plot_biosensor(t, y, baseline, corrected, d1, ttpi, os.path.join(args.out, "biosensor_curves.png"))

    # 4) Risk fusion
    p = naive_bayes_fusion(p_seq, p_bio, prior=0.1, corr=0.1)
    level = grade(p, base=0.6, warn=0.75, danger=0.9)

    # 5) Save outputs
    summary = {
        "inputs": {
            "fastq": fastq_path, "biosensor": biosensor_path, "targets": args.targets,
            "n_sequences": len(seqs)
        },
        "seq": {"counts": counts, "rpm": rpm, "p_seq": round(p_seq,4)},
        "biosensor": {
            "ttp_index": int(ttpi) if ttpi is not None else None,
            "features": {"ttp": feats[0], "slope": feats[1], "max": feats[2], "auc": feats[3]},
            "qscore": round(qscore,4),
            "p_bio": round(p_bio,4)
        },
        "fusion": {"p_final": round(p,4), "level": level}
    }
    save_json(summary, os.path.join(args.out, "summary.json"))
    with open(os.path.join(args.out, "fusion.txt"), "w", encoding="utf-8") as f:
        f.write(f"p_final={p:.4f}, level={level}\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[OK] 결과 저장: {args.out}/summary.json, biosensor_curves.png, seq_barplot.png, fusion.txt")

if __name__ == "__main__":
    main()
