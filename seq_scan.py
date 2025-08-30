from typing import List, Dict, Tuple
import numpy as np

def parse_fastq(path, qmin=10, min_len=200, max_reads=None):
    """Parse FASTQ (4-line blocks). Returns list of sequences (A/C/G/T)."""
    seqs = []
    qmap = {c:i for i,c in enumerate("!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~")}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        i = 0
        while True:
            h = f.readline()
            if not h: break
            s = f.readline().strip().upper()
            p = f.readline()
            q = f.readline().strip()
            if not q: break
            if len(s) < min_len: 
                i+=1; continue
            # quality filter (mean Q)
            if len(q) == len(s):
                qq = np.array([qmap.get(c,0) for c in q])
                if qq.mean() < qmin:
                    i+=1; continue
            seqs.append(s.replace("N",""))
            i+=1
            if max_reads and len(seqs)>=max_reads: break
    return seqs

def hamming_leq(a: str, b: str, k: int) -> bool:
    """Return True if Hamming distance between substrings equals <= k (lengths must match)."""
    d = 0
    for x,y in zip(a,b):
        if x!=y:
            d += 1
            if d>k: return False
    return True

def count_motif_hits(seq: str, motif: str, max_mismatch: int) -> int:
    """Sliding window Hamming-distance search."""
    m = len(motif); n = len(seq)
    if n < m: return 0
    c = 0
    for i in range(n-m+1):
        if hamming_leq(seq[i:i+m], motif, max_mismatch):
            c += 1
    return c

def scan_targets(seqs: List[str], targets: List[Dict]) -> Tuple[Dict[str,int], Dict[str,float]]:
    """Return raw counts and RPM by target."""
    total_reads = len(seqs)
    counts = {}
    for t in targets:
        name = t["name"]; motifs = t["motifs"]; mm = int(t.get("max_mismatch",0))
        cnt = 0
        for s in seqs:
            # count if any motif occurs at least once
            hits = 0
            for motif in motifs:
                if count_motif_hits(s, motif, mm) > 0:
                    hits += 1
            if hits>0: cnt += 1
        counts[name] = cnt
    rpm = {k: (1e6* v / max(total_reads,1)) for k,v in counts.items()}
    return counts, rpm
