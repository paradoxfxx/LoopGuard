# -*- coding: utf-8 -*-
"""已发表基线: LSTM 自编码器检测器(he2017real方法族).
同一台架数据、同一7通道、同一评估协议——公平对比.
训练: hil_stream_clean.npz 干净段(无监督重构)
评估: v1/v2攻击 + 负荷阶跃
"""
import glob, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"E:\NewPaper20261110"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIG7 = ["v701", "v706", "v712", "v718", "v725", "v733", "i_head"]
L = 128  # 窗口(与部署内核一致)
EPOCHS = 50
BS = 64
torch.manual_seed(42); np.random.seed(42)

# ============ 模型 ============
class LSTMAE(nn.Module):
    def __init__(self, m=7, h=48):
        super().__init__()
        self.enc = nn.LSTM(m, h, num_layers=1, batch_first=True)
        self.dec = nn.LSTM(h, m, num_layers=1, batch_first=True)
    def forward(self, x):
        z, _ = self.enc(x)
        y, _ = self.dec(z)
        return y

model = LSTMAE().to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# ============ 数据准备 ============
cl = np.load(os.path.join(ROOT, "results", "hil_stream_clean.npz"))
ca, cn = cl["data"], [str(x) for x in cl["names"]]
ri = [cn.index(s + "_rep") for s in SIG7]
pu = ca[:, ri].astype(np.float32)
mu, sd = pu.mean(0), pu.std(0) + 1e-9

# 标准化
zs = (pu - mu) / sd
T = len(zs)
# 滑窗
X_train = np.stack([zs[i:i+L] for i in range(0, T-L+1, 1)]).astype(np.float32)
print(f"训练窗: {X_train.shape}")

# ============ 训练 ============
ds = TensorDataset(torch.tensor(X_train))
dl = DataLoader(ds, batch_size=BS, shuffle=True)
model.train()
for ep in range(EPOCHS):
    tot = 0
    for (xb,) in dl:
        xb = xb.to(DEV)
        rec = model(xb)
        loss = F.mse_loss(rec, xb)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    if ep % 10 == 9:
        print(f"  ep{ep+1} loss={tot/len(dl):.6f}")

torch.save(model.state_dict(), os.path.join(ROOT, "ckpt", "lstm_ae_baseline.pt"))

# ============ 评估函数 ============
@torch.no_grad()
def lstm_ae_eval(pu_stream, onset, offset, fs, thr_percentile=99):
    """重构误差检测: 窗口末步误差>阈值=报警."""
    z = (pu_stream - mu) / sd
    T2 = len(z)
    errors = np.zeros(T2)
    for s_ in range(0, T2 - L + 1):
        w = torch.tensor(z[s_:s_+L][None], device=DEV)
        rec = model(w)
        err = F.mse_loss(rec[0, -1], w[0, -1]).item()
        errors[s_+L-1] = err
    # 阈值: 干净段重构误差的99百分位
    clean_err = errors[:onset] if onset > L else errors[:100]
    thr = np.percentile(clean_err[clean_err > 0], thr_percentile)
    alarm = errors > thr
    # 去抖(与z-score一致: 20样本)
    delay = None; run = 0
    for t in range(onset, min(offset, T2)):
        run = run + 1 if alarm[t] else 0
        if run >= 20:
            delay = (t - 19 - onset) / fs
            break
    pre_fa = alarm[30:onset].mean() if onset > 30 else 0
    peak = errors[onset:offset].max() / thr if offset > onset else 0
    return {"delay_s": round(delay, 2) if delay else None,
            "preFA": round(pre_fa, 4), "peak_ratio": round(peak, 1),
            "fired": delay is not None}

# ============ 评估所有episode ============
print("\n" + "=" * 70)
print("LSTM-AE 基线 vs 部署GRU vs z-score (同台架,同协议)")
print("=" * 70)
print(f"{'Episode':<22s} {'LSTM-AE':>10s} {'delay':>7s} {'preFA':>7s} {'peak':>6s}")
print("-" * 70)

results = {}
for f in sorted(glob.glob(os.path.join(ROOT, "results", "v2_*.npz"))):
    d = np.load(f)
    arr, names = d["data"], [str(x) for x in d["names"]]
    meta = json.loads(str(d["meta"]))
    fs = meta["fs"]
    tag = os.path.basename(f)[3:-4]
    ri2 = [names.index(s + "_rep") for s in SIG7]
    pu2 = arr[:, ri2].astype(np.float32)
    on = int(meta.get("t_onset", 5) * fs)
    off = int(meta.get("t_offset", len(pu2)) * fs) if "t_offset" in meta else len(pu2)
    r = lstm_ae_eval(pu2, on, off, fs)
    results[tag] = r
    print(f"{tag:<22s} {'FIRE' if r['fired'] else 'quiet':>10s} "
          f"{str(r['delay_s'] or '-'):>7s} {r['preFA']:>7.3f} {r['peak_ratio']:>5.1f}x")

# 也在离线episode集上跑(与Table e1同协议)
print("\n--- 离线episode集(IEEE 39-bus, N_d=1) ---")
sys.path.insert(0, ROOT); sys.path.insert(0, r"E:\NewPaper20261021")
sys.path.insert(0, os.path.join(ROOT, "models"))
from stream_nets import SUBSET12  # noqa
P1M = json.load(open(os.path.join(r"E:\NewPaper20261021", "results", "sim_meta.json")))
mu_p1 = np.array(P1M["chan_mean"], np.float32)[SUBSET12][:7]
sd_p1 = np.array(P1M["chan_std"], np.float32)[SUBSET12][:7]

# 用39-bus训练集训练另一个LSTM-AE
tr = np.load(os.path.join(r"E:\NewPaper20261021", "results", "sim_train_L128.npz"))
X_tr = (tr["X"].astype(np.float32)[:, :, SUBSET12[:7]] - mu_p1) / sd_p1
model2 = LSTMAE().to(DEV)
opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
ds2 = TensorDataset(torch.tensor(X_tr))
dl2 = DataLoader(ds2, batch_size=BS, shuffle=True)
for ep in range(EPOCHS):
    for (xb,) in dl2:
        xb = xb.to(DEV)
        loss = F.mse_loss(model2(xb), xb)
        opt2.zero_grad(); loss.backward(); opt2.step()

@torch.no_grad()
def eval_offline(episode_data):
    z = (episode_data - mu_p1) / sd_p1
    T3 = len(z)
    errs = np.zeros(T3)
    for s_ in range(0, T3 - L + 1):
        w = torch.tensor(z[s_:s_+L][None], device=DEV)
        errs[s_+L-1] = F.mse_loss(model2(w)[0,-1], w[0,-1]).item()
    return errs

eps = np.load(os.path.join(ROOT, "results", "stream_episodes.npz"))
seg, t0s, durs, kinds = eps["seg"], eps["t0"], eps["dur"], eps["kind"]
delays = {"fdia": [], "replay": [], "rfdi": []}
for i in range(len(kinds)):
    ep_data = seg[i][:, SUBSET12[:7]].astype(np.float32)
    on = int(t0s[i])
    errs = eval_offline(ep_data)
    clean_part = errs[:on] if on > 30 else errs[:100]
    thr = np.percentile(clean_part[clean_part > 0], 99)
    alarm = errs > thr
    alarm[:30] = False
    run = 0
    for t in range(on, min(on + int(durs[i]), len(alarm))):
        run = run + 1 if alarm[t] else 0
        if run >= 1:  # N_d=1(与Table e1同口径)
            delays[kinds[i]].append((t - on) * 5)
            break
for k in ("fdia", "replay", "rfdi"):
    dd = delays[k]
    if dd:
        print(f"  {k}: {len(dd)}/30 det, median={np.median(dd):.0f}s")
    else:
        print(f"  {k}: 0/30 det")

json.dump(results, open(os.path.join(ROOT, "results", "lstm_ae_results.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n[done] -> results/lstm_ae_results.json")
