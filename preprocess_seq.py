# preprocess_seq.py
# EE(예상 오류) 필터, Levenshtein(1) 병합, 간이 chimera 제거
import numpy as np
from typing import List, Tuple, Dict

PHRED = {c:i for i,c in enumerate(
    "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
)}

def expected_errors(qual: str) -> float:
    # EE = sum(10^(-Q/10)); 여기선 ASCII->Q 변환
    return float(sum(10 ** (-(PHRED.get(c, 0))/10.0) for c in qual))

def parse_fastq_with_qual(path: str, min_len=100) -> List[Tuple[str,str]]:
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            h = f.readline()
            if not h: break
            s = f.readline().strip().upper()
            p = f.readline()
            q = f.readline().strip()
            if not q: break
            if len(s) >= min_len:
                out.append((s, q))
    return out

def levenshtein1(a: str, b: str) -> bool:
    # True if edit distance <=1 (fast path)
    if a == b: return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1: return False
    # one substitution or one indel
    i = j = diff = 0
    while i<la and j<lb:
        if a[i]==b[j]:
            i+=1; j+=1
        else:
            diff += 1
            if diff>1: return False
            if la==lb:  # substitution
                i+=1; j+=1
            elif la>lb: # deletion in b
                i+=1
            else:       # insertion in b
                j+=1
    diff += (la-i) + (lb-j)
    return diff <= 1

def merge_close_reads(reads: List[str], min_count=2) -> Dict[str,int]:
    # Collapse near-duplicates (<=1 edit)
    counts: Dict[str,int] = {}
    for r in reads: counts[r] = counts.get(r,0)+1
    seqs = sorted(counts.items(), key=lambda x: -x[1])
    kept: Dict[str,int] = {}
    for s,c in seqs:
        merged = False
        for k in list(kept.keys()):
            if levenshtein1(s, k):
                kept[k] += c
                merged = True
                break
        if not merged:
            if c >= min_count:
                kept[s] = c
    return kept

def is_chimera(seq: str, parents: List[str], k=20) -> bool:
    # 간이 검사: 임의 분할점에서 앞은 P1, 뒤는 P2와 일치하면 chimera로 간주
    for i in range(k, len(seq)-k, k):
        left, right = seq[:i], seq[i:]
        match_left = any(left in p for p in parents)
        match_right= any(right in p for p in parents)
        if match_left and match_right:
            return True
    return False

def preprocess_fastq(path: str, ee_max: float = 1.0, min_len: int = 150) -> Dict[str,int]:
    pairs = parse_fastq_with_qual(path, min_len=min_len)
    # EE 필터
    filt = [s for (s,q) in pairs if expected_errors(q) <= ee_max]
    if not filt: return {}
    # 근접 병합
    merged = merge_close_reads(filt, min_count=2)
    if not merged: return {}
    # chimera 제거
    parents = list(merged.keys())
    clean: Dict[str,int] = {}
    for s,c in merged.items():
        if not is_chimera(s, parents):
            clean[s] = c
    return clean  # {sequence: count}
