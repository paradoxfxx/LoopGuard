# -*- coding: utf-8 -*-
r"""LoopGuard E1/E2 streaming eval: test F1 + episode latency + int8 weight-only
quantization sensitivity, for the M2 causal models (12ch subset).
Reuses the frozen episode set results/stream_episodes.npz (protocol M1a)."""
from __future__ import annotations
import argparse, glob, json, os, sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
P1 = r"E:\NewPaper20261021"
ROOT = r"E:\NewPaper20261110"
sys.path.insert(0, P1)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models"))
from stream_nets import build, SUBSET12                      # noqa: E402
ATTACK = (2, 3, 4)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_test():
    meta = json.load(open(os.path.join(P1, "results", "sim_meta.json")))
    mu = np.array(meta["chan_mean"], dtype=np.float32)
    sd = np.array(meta["chan_std"], dtype=np.float32)
    d = np.load(os.path.join(P1, "results", "sim_test_L128.npz"))
    X = (d["X"].astype(np.float32) - mu) / sd
    return X[:, :, SUBSET12], d["y"].astype(np.int64)


def quant_int8_state_dict(sd):
    out = {}
    for k, v in sd.items():
        if v.ndim == 2 and v.dtype.is_floating_point and "weight" in k:
            scale = v.abs().amax(dim=1, keepdim=True) / 127.0
            scale = torch.clamp(scale, min=1e-12)
            q = torch.clamp(torch.round(v / scale), -127, 127) * scale
            out[k] = q
        else:
            out[k] = v
    return out


@torch.no_grad()
def episode_latency(model, seg, t0, dur):
    """Causal forward over the segment; first alarm at t>=t0."""
    x = torch.tensor(seg[None], device=DEV)          # (1,T,12)
    logits = model(x)[0]                             # (T,5)
    pred = logits.argmax(1).cpu().numpy()
    hi = min(t0 + dur, len(pred))
    hits = np.where(np.isin(pred[t0:hi], ATTACK))[0]
    delay = int(hits[0]) if len(hits) else None      # 切片索引已相对 t0
    # 攻击前误报:预热(>=16)到 onset 之间出现攻击类预测的步数占比
    pre = pred[max(16, t0 - 128):t0]
    fa = float(np.isin(pre, ATTACK).mean()) if len(pre) else 0.0
    return delay, fa


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)          # e.g. stream_gru_12ch_d48_s0.pt
    args = ap.parse_args()
    stem = os.path.basename(args.ckpt)[:-3]           # stream_gru_12ch_d48_s0
    parts = stem.split("_")
    model_name, chan, d, seed = parts[1], parts[2], int(parts[3][1:]), int(parts[4][1:])
    model = build(model_name, m=len(SUBSET12), d=d).to(DEV)
    sd = torch.load(os.path.join(ROOT, "ckpt", args.ckpt), weights_only=True)
    model.load_state_dict(sd); model.eval()

    out = {"ckpt": stem, "params": sum(p.numel() for p in model.parameters())}

    # test final-step macro-F1
    Xte, yte = load_test()
    preds = []
    for s in range(0, len(Xte), 512):
        xb = torch.tensor(Xte[s:s + 512], device=DEV)
        preds.append(model(xb)[:, -1].argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    from sklearn.metrics import f1_score
    out["test_macro_f1_finalstep"] = float(f1_score(yte, preds, average="macro",
                                                    labels=list(range(5))))
    print(f"[{stem}] test macro-F1(final step) {out['test_macro_f1_finalstep']:.3f}")

    # frozen episode latency
    eps = np.load(os.path.join(ROOT, "results", "stream_episodes.npz"))
    seg, t0s, durs, kinds = eps["seg"], eps["t0"], eps["dur"], eps["kind"]
    lat = {}
    for kind in ("fdia", "replay", "rfdi"):
        idx = np.where(kinds == kind)[0]
        res = [episode_latency(model, seg[i][:, SUBSET12].astype(np.float32),
                               int(t0s[i]), int(durs[i])) for i in idx]
        delays = [d for d, f in res if d is not None]
        fas = [f for d, f in res]
        dd = np.array(delays, float)
        lat[kind] = {"det": int(dd.size), "miss": len(res) - dd.size,
                     "median": float(np.median(dd)) if dd.size else None,
                     "p95": round(float(np.percentile(dd, 95)), 1) if dd.size else None,
                     "max": float(dd.max()) if dd.size else None,
                     "pre_onset_fa_rate": round(float(np.mean(fas)), 4)}
        print(f"[{stem}] {kind}: det {lat[kind]['det']}/30 median {lat[kind]['median']} "
              f"p95 {lat[kind]['p95']} preFA {lat[kind]['pre_onset_fa_rate']}")
    out["latency_steps"] = lat

    # int8 weight-only quantization sensitivity (test F1 + one latency class)
    model.load_state_dict(quant_int8_state_dict(sd)); model.eval()
    preds = []
    for s in range(0, len(Xte), 512):
        xb = torch.tensor(Xte[s:s + 512], device=DEV)
        preds.append(model(xb)[:, -1].argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    out["test_macro_f1_int8weights"] = float(f1_score(yte, preds, average="macro",
                                                      labels=list(range(5))))
    latq = {}
    for kind in ("fdia", "rfdi"):
        idx = np.where(kinds == kind)[0]
        res = [episode_latency(model, seg[i][:, SUBSET12].astype(np.float32),
                               int(t0s[i]), int(durs[i])) for i in idx]
        dd = np.array([d for d, f in res if d is not None], float)
        latq[kind] = {"det": int(dd.size),
                      "median": float(np.median(dd)) if dd.size else None}
    out["latency_int8weights"] = latq
    print(f"[{stem}] int8-weight F1 {out['test_macro_f1_int8weights']:.3f} "
          f"fdia det {latq['fdia']['det']}/30")

    dst = os.path.join(ROOT, "results", f"eval_{stem}.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[done] ->", dst)


if __name__ == "__main__":
    main()
