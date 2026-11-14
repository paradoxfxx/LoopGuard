# -*- coding: utf-8 -*-
r"""LoopGuard E7: sim-to-HIL 域校准三级阶梯(离线,用已采 hil_stream_*.npz).

A zero-shot   : 论文1 P语义权重 + 论文1 mu/sd(=上轮E6部署形态,已知OOD永久告警)
B input-recal : 同权重,HIL-clean 实测 mu/sd 重标准化(部署=改AO scaling/offset,零固件改动)
C few-shot FT : 在 B 基础上用部分 HIL episode 微调(LR 1e-4, 少 epoch),留出episode检验
指标: 每episode 攻击前误报步率(preFA) / 检出延迟(5连拍去抖,步=0.1s) / 清除后恢复
输出: results/e7_calibration.json + 微调ckpt(若C胜出)
"""
from __future__ import annotations
import glob, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
P1 = r"E:\NewPaper20261021"
ROOT = r"E:\NewPaper20261110"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models"))
from stream_nets import build, SUBSET12                      # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L, DEC = 128, 200            # 窗长; 2kS/s -> 10Hz 抽取因子
SIG7 = ["v701", "v706", "v712", "v718", "v725", "v733", "i_head"]
ATTACK = (2, 3, 4)
CLS = {"clean": 0, "loadstep": 1, "fdia": 2, "rfdi": 4}
TRAIN_EPS = ("fdia706", "fdia718", "rfdi0", "rfdi1", "loadstep0")
TEST_EPS = ("fdia712", "rfdi2", "loadstep1")


def load_all():
    eps = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "hil_stream_*.npz"))):
        d = np.load(f)
        tag = os.path.basename(f)[11:-4]
        meta = json.loads(str(d["meta"]))
        names = [str(x) for x in d["names"]]
        idx = [names.index(s + "_rep") for s in SIG7]
        x = d["data"][:, idx].astype(np.float32)          # (N,7) @2kS/s
        n = (len(x) // DEC) * DEC
        x = x[:n].reshape(n // DEC, DEC, 7).mean(axis=1)  # 10 Hz 均值抽取
        on = None
        if meta.get("bias"):
            on = (int(round(meta["onset_wall"] * meta["fs_actual"])) // DEC,
                  int(round(meta["offset_wall"] * meta["fs_actual"])) // DEC)
        x = x[50:]
        if on:
            on = (max(0, on[0] - 50), max(0, on[1] - 50))
        eps[tag] = {"x": x, "onset": on}
    return eps


def windows(ep_x, label, onset, stride=32):
    """(N,7) 流 -> 窗口 (K,L,7), 标签; 事件窗=任意重叠>=16步。"""
    X, y = [], []
    T = len(ep_x)
    for st in range(0, T - L + 1, stride):
        e = st + L
        lab, keep = 0, True
        if onset and label != 0:
            ov = max(0, min(e, onset[1]) - max(st, onset[0]))
            if ov >= 24 and st >= onset[0] - 16 and e <= onset[1] + 64:
                lab = label          # 窗体基本在攻击段内
            elif 1 <= ov < 24:
                keep = False         # 模糊过渡窗: 丢弃
        if keep:
            X.append(ep_x[st:e])
            y.append(lab)
    return (np.stack(X), np.array(y)) if X else (np.zeros((0, L, ep_x.shape[1])), np.zeros(0, int))


@torch.no_grad()
def stream_eval(model, x, onset, mu, sd, warmup=16):
    """滑动128窗因果前向(与训练分布一致):pred[t]=model(x[t-127:t])末步。
    步=0.1s;前128步无窗不计。"""
    z = (x - mu) / sd
    T = len(z)
    if T < L:
        return {"preFA": None, "note": "too short"}
    W = np.stack([z[t - L:t] for t in range(L, T)])       # (T-L, L, 7)
    preds = []
    for sI in range(0, len(W), 256):
        xb = torch.tensor(W[sI:sI + 256], device=DEV)
        preds.append(model(xb)[:, -1].argmax(1).cpu().numpy())
    pred_full = np.concatenate(preds)
    pred = np.zeros(T, dtype=pred_full.dtype)
    pred[L:] = pred_full
    atk = np.isin(pred, ATTACK)
    atk[:L + 5] = False             # 首窗预热抑制
    # 5连拍去抖(同固件FSM)
    run, alarm_t, clear_t = 0, None, None
    for t, a in enumerate(atk):
        run = run + 1 if a else 0
        if alarm_t is None and run >= 5:
            alarm_t = t
        if alarm_t is not None and run == 0 and clear_t is None:
            clear_t = t
    if onset is None:
        return {"preFA": float(atk[30:].mean()), "alarm_t": alarm_t,
                "clear_t": clear_t, "atk_rate_all": float(atk[30:].mean())}
    pre = atk[30:onset[0]]
    delay = (alarm_t - onset[0]) if (alarm_t is not None and alarm_t >= onset[0]) else None
    rec = (clear_t - onset[1]) if (clear_t is not None and clear_t >= onset[1]) else None
    return {"preFA": float(pre.mean()) if len(pre) else 0.0, "delay_steps": delay,
            "recovery_steps": rec, "alarm_t": alarm_t}


def main():
    eps = load_all()
    print("[load]", {k: v["x"].shape for k, v in eps.items()})

    # 标准化两套
    meta = json.load(open(os.path.join(P1, "results", "sim_meta.json")))
    mu_p1 = np.array(meta["chan_mean"], np.float32)[SUBSET12][:7]
    sd_p1 = np.array(meta["chan_std"], np.float32)[SUBSET12][:7]
    clean = eps["clean"]["x"]
    mu_h = clean.mean(0)
    sd_h = clean.std(0) + 1e-6
    # 注意: 固件链路里 AO 已按 mu_p1/sd_p1 标准化过一次; 离线 B 的口径 =
    # 把 rep 原值按 HIL 统计重标(部署等价 = 改AO绑定参数), 模型输入分布一致。
    muB, sdB = mu_h, sd_h          # 输入 = rep 原值(未过AO标准化) 的 HIL z-score

    model = build("gru", m=7, d=48)
    # 12ch ckpt -> 取前7通道切片加载(in_proj 输入行切片, 其余结构同)
    sd12 = torch.load(os.path.join(ROOT, "ckpt", "stream_gru_12ch_d48_s0.pt"),
                      weights_only=True, map_location="cpu")
    cols = list(range(7)) + [12 + i for i in range(7)] + [24 + i for i in range(7)]  # [x|dx|b]各取前7
    sd7 = {}
    for k, v in sd12.items():
        sd7[k] = v[:, cols].clone() if (k == "in_proj.weight") else v.clone()
    model.load_state_dict(sd7)
    model.to(DEV).eval()
    print("[model] 12ch ckpt 前7通道切片加载:", sum(p.numel() for p in model.parameters()), "params")

    # ---- D: 部署匹配口径(sigma=未池化raw sigma; 固件池化后噪声~0.1sigma) ----
    res = {"A_zeroshot": {}, "B_inputrecal": {}}
    sd_raw = (np.sqrt(200.0) * sd_h).astype(np.float32)  # 2kS/s 原始流 sigma
    res["D_deploy"] = {}
    for tag in TEST_EPS + ("clean",):
        res["D_deploy"][tag] = stream_eval(model, eps[tag]["x"], eps[tag]["onset"], muB, sd_raw)
        print(f"[D] {tag}: {res['D_deploy'][tag]}")
    res["deploy_D"] = {"ao_scaling": [round(float(1.0 / s_), 6) for s_ in sd_raw],
                       "ao_offset": [round(float(-m / s_), 6) for m, s_ in zip(mu_h, sd_raw)],
                       "sigma_raw": [round(float(s_), 6) for s_ in sd_raw]}

    # ---- A/B: 零样本 vs 输入重标定 --------------------------------------
    for tag in TEST_EPS + ("clean",):
        mu, sd = (mu_p1, sd_p1) if tag != "clean" else (mu_p1, sd_p1)
        res["A_zeroshot"][tag] = stream_eval(model, eps[tag]["x"], eps[tag]["onset"], mu_p1, sd_p1)
        res["B_inputrecal"][tag] = stream_eval(model, eps[tag]["x"], eps[tag]["onset"], muB, sdB)
        print(f"[A] {tag}: {res['A_zeroshot'][tag]}")
        print(f"[B] {tag}: {res['B_inputrecal'][tag]}")

    # ---- C: 少样本微调(在 B 口径上) --------------------------------------
    Xs, ys = [], []
    for tag in TRAIN_EPS:
        lab = CLS["fdia" if tag.startswith("fdia") else "rfdi" if tag.startswith("rfdi") else "loadstep"]
        w, yv = windows((eps[tag]["x"] - muB) / sdB, lab, eps[tag]["onset"])
        Xs.append(w); ys.append(yv)
    # clean 前半做 normal 类
    wc, yc = windows((clean[:len(clean) // 2] - muB) / sdB, 0, None)
    Xs.append(wc); ys.append(yc)
    X = np.concatenate(Xs).astype(np.float32)
    Y = np.concatenate(ys).astype(np.int64)
    print("[C] train windows:", X.shape, "class hist:", np.bincount(Y, minlength=5).tolist())

    cnt = np.bincount(Y, minlength=5)
    wts = torch.tensor(cnt.sum() / (5 * np.maximum(cnt, 1)), dtype=torch.float32, device=DEV)
    wts = (wts / wts.mean())
    wts[cnt == 0] = 0.0              # 缺席类(replay)权重清零
    head_params = [p for n_, p in model.named_parameters() if n_.startswith("head")]
    opt = torch.optim.Adam(head_params, lr=1e-3)
    for p in model.parameters():
        p.requires_grad = False
    for p in head_params:
        p.requires_grad = True
    model.train()
    for ep in range(6):
        perm = np.random.permutation(len(X))
        tot, nb = 0.0, 0
        for s in range(0, len(perm), 32):
            idx = perm[s:s + 32]
            xb = torch.tensor(X[idx], device=DEV)
            yb = torch.tensor(Y[idx], device=DEV)
            logits = model(xb)
            T = logits.shape[1]
            loss = F.cross_entropy(logits[:, max(16, T - 32):, :].reshape(-1, 5),
                                   yb[:, None].expand(-1, T - max(16, T - 32)).reshape(-1),
                                   weight=wts)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        print(f"[C] ep{ep} loss {tot/nb:.4f}")
    model.eval()
    res["C_fewshot"] = {}
    for tag in TEST_EPS + ("clean",):
        res["C_fewshot"][tag] = stream_eval(model, eps[tag]["x"], eps[tag]["onset"], muB, sdB)
        print(f"[C] {tag}: {res['C_fewshot'][tag]}")

    # 部署参数(B 口径的 AO 重绑参数; C 胜出则另出权重)
    res["deploy_B"] = {"ao_scaling": [round(float(1.0 / s), 6) for s in sd_h],
                       "ao_offset": [round(float(-m / s), 6) for m, s in zip(mu_h, sd_h)]}
    torch.save(model.state_dict(), os.path.join(ROOT, "ckpt", "stream_gru_7ch_hil_headft.pt"))

    # ---- C2: HIL 原生从头小模型(d=24) ----
    m2 = build("gru", m=7, d=24)
    m2.to(DEV)
    cnt = np.bincount(Y, minlength=5)
    w2 = torch.tensor(cnt.sum() / (5 * np.maximum(cnt, 1)), dtype=torch.float32, device=DEV)
    w2 = w2 / w2.mean(); w2[cnt == 0] = 0.0
    opt2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
    m2.train()
    for ep in range(15):
        perm = np.random.permutation(len(X))
        tot, nb = 0.0, 0
        for sI in range(0, len(perm), 32):
            idx = perm[sI:sI + 32]
            xb = torch.tensor(X[idx], device=DEV)
            yb = torch.tensor(Y[idx], device=DEV)
            lg = m2(xb)
            T = lg.shape[1]
            lo = F.cross_entropy(lg[:, max(16, T - 48):, :].reshape(-1, 5),
                                  yb[:, None].expand(-1, T - max(16, T - 48)).reshape(-1), weight=w2)
            opt2.zero_grad(); lo.backward()
            torch.nn.utils.clip_grad_norm_(m2.parameters(), 1.0)
            opt2.step()
            tot += lo.item(); nb += 1
        if ep % 5 == 4:
            print(f"[C2] ep{ep} loss {tot/nb:.4f}")
    m2.eval()
    res["C2_hilnative"] = {}
    for tag in TEST_EPS + ("clean",):
        res["C2_hilnative"][tag] = stream_eval(m2, eps[tag]["x"], eps[tag]["onset"], muB, sdB)
        print(f"[C2] {tag}: {res['C2_hilnative'][tag]}")
    torch.save(m2.state_dict(), os.path.join(ROOT, "ckpt", "stream_gru_7ch_hil_native_d24.pt"))
    json.dump(res, open(os.path.join(ROOT, "results", "e7_calibration.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[done] -> results/e7_calibration.json")


if __name__ == "__main__":
    main()
