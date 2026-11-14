# -*- coding: utf-8 -*-
r"""LoopGuard M2 — streaming training (E1) on paper-1 data.

Protocol: causal models, per-step CE from t>=WARMUP with window label,
class-weighted; 12-channel edge subset (SUBSET12) or full 131.
Runs: {gru,ssm} x seeds{0,1,2} x 12ch (+ optional full via --chan full).
GPU single job, serial. ckpt -> E:\NewPaper20261110\ckpt\
"""
from __future__ import annotations
import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
P1 = r"E:\NewPaper20261021"
ROOT = r"E:\NewPaper20261110"
sys.path.insert(0, P1)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models"))
from stream_nets import build, SUBSET12                      # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WARMUP = 16          # 不计损失的起始段
LOSS_LAST = 32        # 只对最后32步计损失(窗口目标,见首轮复盘)
LR, BS, EPOCHS = 1e-3, 256, 12


def load(split, chans):
    meta = json.load(open(os.path.join(P1, "results", "sim_meta.json")))
    mu = np.array(meta["chan_mean"], dtype=np.float32)
    sd = np.array(meta["chan_std"], dtype=np.float32)
    d = np.load(os.path.join(P1, "results", f"sim_{split}_L128.npz"))
    X = (d["X"].astype(np.float32) - mu) / sd
    y = d["y"].astype(np.int64)
    if chans == "12":
        X = X[:, :, SUBSET12]
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gru")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chan", default="12")
    ap.add_argument("--d", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    Xtr, ytr = load("train", args.chan)
    Xva, yva = load("val", args.chan) if os.path.exists(
        os.path.join(P1, "results", "sim_val_L128.npz")) else (None, None)
    m = Xtr.shape[2]
    model = build(args.model, m, d=args.d).to(DEV)
    n_par = sum(p.numel() for p in model.parameters())

    cnt = np.bincount(ytr, minlength=5)
    w = cnt.sum() / (5 * np.maximum(cnt, 1))
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=DEV)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    t0 = time.time()
    hist = []
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(Xtr))
        tot, nb, loss_sum = 0.0, 0, 0.0
        for s in range(0, len(perm), BS):
            idx = perm[s:s + BS]
            xb = torch.tensor(Xtr[idx], device=DEV)
            yb = torch.tensor(ytr[idx], device=DEV)
            logits = model(xb)                       # (B,T,5)
            T = logits.shape[1]
            lo = max(WARMUP, T - LOSS_LAST)
            lg = logits[:, lo:, :].reshape(-1, 5)
            yv = yb[:, None].expand(-1, T - lo).reshape(-1)
            loss = F.cross_entropy(lg, yv, weight=w)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            xs = torch.tensor(Xtr[:2048], device=DEV)
            lg = model(xs)[:, -1].argmax(1).cpu().numpy()
            from sklearn.metrics import f1_score
            f1 = f1_score(ytr[:2048], lg, average="macro", labels=list(range(5)))
        hist.append({"ep": ep, "loss": round(loss_sum / nb, 4), "f1_train2048": round(f1, 4)})
        print(f"[{args.model} s{args.seed} {args.chan}ch] ep{ep} loss {loss_sum/nb:.4f} f1 {f1:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # 诊断:预测分布(定位"学没学")
    with torch.no_grad():
        xs = torch.tensor(Xtr[:4096], device=DEV)
        pr = model(xs)[:, -1].argmax(1).cpu().numpy()
    import collections
    hist = dict(collections.Counter(pr.tolist()))
    print(f"[{args.model} s{args.seed} {args.chan}ch] pred hist {hist}", flush=True)
    name = f"stream_{args.model}_{args.chan}ch_d{args.d}_s{args.seed}.pt"
    torch.save(model.state_dict(), os.path.join(ROOT, "ckpt", name))
    with open(os.path.join(ROOT, "results", f"train_{name}.json"), "w", encoding="utf-8") as fh:
        json.dump({"model": args.model, "seed": args.seed, "chan": args.chan,
                   "d": args.d, "params": int(n_par), "epochs": args.epochs,
                   "hist": hist, "m": m}, fh, ensure_ascii=False, indent=1)
    print(f"[done] {name} params={n_par}")


if __name__ == "__main__":
    main()
