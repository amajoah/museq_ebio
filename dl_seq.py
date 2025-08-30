# dl_seq.py
# 1D-CNN: k-mer one-hot -> Conv -> GlobalPool -> Sigmoid (target prob)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from typing import List

DNA = "ACGT"
IDX = {c:i for i,c in enumerate(DNA)}

def onehot(seq: str, k: int = 1):
    # simple one-hot over nucleotides (k=1). For k>1, expand vocab accordingly.
    x = np.zeros((len(seq), 4), dtype=np.float32)
    for i,ch in enumerate(seq):
        j = IDX.get(ch, None)
        if j is not None: x[i, j] = 1.0
    return x.T  # [4, L]

class SeqCNN(nn.Module):
    def __init__(self, in_ch=4, c1=64, c2=128, k=9):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, c1, k, padding=k//2)
        self.bn1   = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, k, padding=k//2)
        self.bn2   = nn.BatchNorm1d(c2)
        self.head  = nn.Linear(c2, 1)

    def forward(self, x):  # x: [B,4,L]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.mean(x, dim=-1)  # global average pool
        x = self.head(x)
        return torch.sigmoid(x).squeeze(-1)  # [B]

class EDNAClassifier:
    """Loadable classifier wrapper. Train elsewhere; here we support inference."""
    def __init__(self, weight_path: str | None = None, device: str = "cpu"):
        self.device = device
        self.model = SeqCNN().to(device).eval()
        if weight_path:
            self.model.load_state_dict(torch.load(weight_path, map_location=device))

    @torch.no_grad()
    def predict_probs(self, seqs: List[str], max_len: int = 500):
        # center-crop or pad to max_len
        batch = []
        for s in seqs:
            s = s[:max_len]
            X = onehot(s)  # [4,L]
            if X.shape[1] < max_len:
                pad = np.zeros((4, max_len - X.shape[1]), dtype=np.float32)
                X = np.concatenate([X, pad], axis=1)
            elif X.shape[1] > max_len:
                X = X[:, :max_len]
            batch.append(X)
        x = torch.tensor(np.stack(batch), dtype=torch.float32).to(self.device)
        return self.model(x).detach().cpu().numpy().tolist()
