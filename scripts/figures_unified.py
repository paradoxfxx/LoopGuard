# -*- coding: utf-8 -*-
"""统一样式重生全部 matplotlib 图(7张): 3.45in 栏宽 + 固定字号栈.
字号栈: 轴标签8 / 刻度7 / 图例7 / 注记6.5 / 标题不用."""
import json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"E:\NewPaper20261110"
FIG = os.path.join(ROOT, "figures")
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CRED = "#e34948"
INK, MUT = "#1a1a19", "#6f6e66"

STYLE = {
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUT,
    "axes.labelcolor": INK,
    "xtick.color": MUT,
    "ytick.color": MUT,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "lines.linewidth": 1.3,
}
plt.rcParams.update(STYLE)
CW = 3.45  # IEEE column width (in)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight",
                pad_inches=0.02)
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    print("[fig]", name)


def J(p):
    return json.load(open(p, encoding="utf-8"))


# ---------- fig_latency ----------
def fig_latency():
    inc = J(os.path.join(ROOT, "results", "stream_gru_baseline.json"))
    st = J(os.path.join(ROOT, "results", "eval_stream_gru_12ch_d48_s0.json"))
    import glob as g
    sts = [J(f) for f in sorted(g.glob(os.path.join(
        ROOT, "results", "eval_stream_gru_12ch_d48_s*.json")))]
    cls = ["fdia", "replay", "rfdi"]
    labels = ["FDIA", "Replay", "Random\nfalse data"]
    inc_m = [inc["latency"][c]["median"] * 5 for c in cls]
    inc_p = [inc["latency"][c]["p95"] * 5 for c in cls]
    st_m = [np.mean([st["latency_steps"][c]["median"] * 5 for st in sts
                     if st["latency_steps"][c]["median"] is not None]) for c in cls]
    st_p = [np.mean([st["latency_steps"][c]["p95"] * 5 for st in sts
                     if st["latency_steps"][c]["p95"] is not None]) for c in cls]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(CW, 1.62))
    for i in range(3):
        ax.plot([i - .14] * 2, [inc_m[i], inc_p[i]], color=C1, lw=1.4)
        ax.plot(i - .14, inc_m[i], "o", color=C1, ms=4)
        ax.plot(i - .14, inc_p[i], "_", color=C1, ms=6)
        ax.plot([i + .14] * 2, [st_m[i], st_p[i]], color=C2, lw=1.4)
        ax.plot(i + .14, st_m[i], "o", color=C2, ms=4)
        ax.plot(i + .14, st_p[i], "_", color=C2, ms=6)
        ax.annotate(f"{inc_m[i]:.0f}", (i - .14, inc_m[i]),
                    textcoords="offset points", xytext=(3, 3), fontsize=7,
                    color=C1, ha="left", va="bottom")
        ax.annotate(f"{st_m[i]:.0f}", (i + .14, st_m[i]),
                    textcoords="offset points", xytext=(3, 3), fontsize=7,
                    color=C2, ha="left", va="bottom")
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("detection delay (s, log)")
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(5, 1000)
    ax.plot([], [], "o-", color=C1, label="windowed incumbent")
    ax.plot([], [], "o-", color=C2, label="causal streaming (ours)")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig_latency")


# ---------- fig_hil_attack ----------
def fig_hil_attack():
    d = np.load(os.path.join(ROOT, "results", "hil_stream_fdia712.npz"))
    arr, names = d["data"], [str(x) for x in d["names"]]
    meta = json.loads(str(d["meta"]))
    fs = meta["fs_actual"]
    on = int(meta["onset_wall"] * fs)
    ri = [i for i, n in enumerate(names) if n.endswith("_rep")]
    rn = [names[i].replace("_rep", "") for i in ri]
    t = (np.arange(len(arr)) - on) / fs
    cl = np.load(os.path.join(ROOT, "results", "hil_stream_clean.npz"))
    ca, cn = cl["data"], [str(x) for x in cl["names"]]
    ci = [cn.index(n + "_rep") for n in rn]
    mu = ca[:, ci].mean(0)
    sd = ca[:, ci].std(0) + 1e-9
    zmax = np.abs((arr[:, ri] - mu) / sd).max(axis=1)
    fig, ax = plt.subplots(figsize=(CW, 1.46))
    ax.plot(t, zmax, color=C1, lw=1.1)
    ax.axhline(4.0, color=MUT, lw=0.7, ls="--")
    ax.text(t[-1] * .97, 4.45, "threshold $4\sigma_n$", fontsize=6.5, color=MUT,
            ha="right", va="bottom")
    ax.axvline(0, color=C2, lw=0.9)
    ax.text(1.5, zmax.max() * 1.02, "attack onset", fontsize=6.5, color=C2,
            va="bottom")
    ax.set_xlabel("time from onset (s)")
    ax.set_ylabel("max $|z|$ over 11 channels")
    ax.set_ylim(0, zmax.max() * 1.22)
    save(fig, "fig_hil_attack")


# ---------- fig_calib ----------
def fig_calib():
    e7 = J(os.path.join(ROOT, "results", "e7_calibration.json"))
    vars_ = [("A_zeroshot", "zero-shot"), ("B_inputrecal", "recalibration"),
             ("C_fewshot", "head fine-tune"), ("C2_hilnative", "bench-native")]
    prefa, delay = [], []
    for k, _ in vars_:
        # pre-FA is the step-level rate on CLEAN traffic (Table VII definition)
        prefa.append(100.0 * e7[k]["clean"]["preFA"])
        d = e7[k]["fdia712"].get("delay_steps")
        delay.append(d * 0.1 if d is not None else None)
    fig, ax = plt.subplots(figsize=(CW, 1.46))
    xp = np.arange(4)
    colors = [CRED, C1, C4, C3]
    ax.bar(xp, prefa, width=0.55, color=colors, edgecolor="none")
    for i in range(4):
        # percentage: inside tall bars, above short ones
        if prefa[i] > 12:
            ax.annotate("%.1f%%" % prefa[i], (xp[i], prefa[i] - 4), ha="center",
                        va="top", fontsize=6.5, color="white")
        else:
            ax.annotate("%.1f%%" % prefa[i], (xp[i], prefa[i] + 2), ha="center",
                        va="bottom", fontsize=6.5, color=INK)
        # detection row on a common baseline
        if delay[i] is None:
            txt, col = "no detection", MUT
        elif vars_[i][0] == "C_fewshot":
            txt, col = "FDIA 10.2 s\n(identical to B)", INK
        else:
            txt, col = "FDIA %.1f s" % delay[i], INK
        ax.annotate(txt, (xp[i], 106 if i % 2 == 0 else 122), ha="center",
                    va="bottom", fontsize=6.5, color=col)
    ax.set_xticks(xp, [v[1] for v in vars_])
    ax.tick_params(axis="x", labelsize=6.2)
    ax.set_ylabel("pre-attack FA rate on clean (%)")
    ax.set_ylim(0, 168)
    ax.set_yticks([0, 25, 50, 75, 100])
    save(fig, "fig_calib")

# ---------- fig_closedloop ----------
def fig_closedloop():
    # e6v3 is the headline protocol cited in the text (4.59 s, pre-attack DI4 = 0.000);
    # e6v4/e6v5 carry a pre-onset alarm baseline and must not be the source of Fig. closedloop.
    for pref in ("e6v3_fdia712", "e6v5_fdia712"):
        p = os.path.join(ROOT, "results", pref + ".npz")
        if os.path.exists(p):
            src = p
            break
    d = np.load(src)
    arr, names = d["data"], [str(x) for x in d["names"]]
    meta = json.loads(str(d["meta"]))
    fs = meta["fs"]
    t = np.arange(len(arr)) / fs - meta["t_onset"]
    v712 = arr[:, names.index("v712_rep")]
    di4 = arr[:, names.index("HIL0 digital input 4")]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(CW, 1.85), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(t, v712, color=C1, lw=1.0)
    a1.set_ylim(v712.min() - .010, v712.max() + .016)
    a1.axvline(0, color=C2, lw=0.9)
    rise = np.where((di4[1:] > .5) & (di4[:-1] <= .5))[0]
    tr = rise[0] / fs - meta["t_onset"] if len(rise) else None
    if tr is not None:
        a1.axvline(tr, color=CRED, lw=0.9, ls="--")
        a1.annotate(f"alarm + isolation\nt = {tr:.2f} s", (tr, v712.max() + .004),
                    xytext=(-6, 14), textcoords="offset points", fontsize=6.5,
                    color=CRED, va="bottom", ha="right",
                    arrowprops=dict(arrowstyle="-", color=CRED, lw=0.6,
                                    shrinkA=0, shrinkB=2))
    a1.text(0.4, v712.min() - .006, "attack onset", fontsize=6.5, color=C2)
    a1.set_ylabel("reported $v_{712}$ (pu)")
    a2.plot(t, di4, color=CRED, lw=1.1, drawstyle="steps-post")
    a2.set_ylabel("DI4 (alarm)")
    a2.set_xlabel("time from attack onset (s)")
    a2.set_ylim(-.1, 1.2)
    save(fig, "fig_closedloop")


# ---------- fig_profile ----------
def fig_profile():
    fl = J(os.path.join(ROOT, "hil", "firstlight.json"))
    prof = fl["v_profile_pu"]
    buses = [701, 703, 706, 709, 712, 715, 718, 725, 730, 733]
    v = [prof[str(b)] for b in buses]
    main = [i for i, b in enumerate(buses) if b <= 718]
    lat = [i for i, b in enumerate(buses) if b > 718]
    fig, ax = plt.subplots(figsize=(CW, 1.42))
    ax.plot([buses[i] for i in main], [v[i] for i in main], "o-", color=C1, ms=3.5, label="main feeder")
    ax.plot([buses[i] for i in lat], [v[i] for i in lat], "s", color=C3, ms=3.5, label="laterals")
    for i, b in enumerate(buses):
        dy = 5 if i % 2 == 0 else -9
        ax.annotate(f"{v[i]:.3f}", (b, v[i]), textcoords="offset points",
                    xytext=(0, dy), fontsize=6, color=MUT, ha="center")
    ax.set_xlabel("bus")
    ax.set_ylabel("voltage (pu)")
    ax.set_ylim(0.90, 1.02)
    ax.legend(frameon=False, loc="lower left")
    save(fig, "fig_profile")


# ---------- fig_delay ----------
def fig_delay():
    import torch
    sys.path.insert(0, ROOT)
    sys.path.insert(0, r"E:\NewPaper20261021")
    sys.path.insert(0, os.path.join(ROOT, "models"))
    from stream_nets import build, SUBSET12
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps = np.load(os.path.join(ROOT, "results", "stream_episodes.npz"))
    seg, t0s, durs, kinds = eps["seg"], eps["t0"], eps["dur"], eps["kind"]
    # Same protocol as scripts/eval_stream2.py and Table I: first attack-class
    # decision, N_d = 1, 5 s per step.
    delays = {"fdia": [], "replay": [], "rfdi": []}
    for seed in (0, 1, 2):
        m = build("gru", m=12, d=48)
        sd = torch.load(os.path.join(ROOT, "ckpt",
                        "stream_gru_12ch_d48_s%d.pt" % seed),
                        weights_only=True, map_location="cpu")
        m.load_state_dict(sd)
        m.to(DEV).eval()
        with torch.no_grad():
            for i in range(len(kinds)):
                x = torch.tensor(seg[i][:, SUBSET12].astype(np.float32)[None],
                                 device=DEV)
                pred = m(x)[0].argmax(1).cpu().numpy()
                on = int(t0s[i])
                hi = min(on + int(durs[i]), len(pred))
                hits = np.where(np.isin(pred[on:hi], (2, 3, 4)))[0]
                if hits.size:
                    delays[kinds[i]].append(int(hits[0]) * 5)
    fig, ax = plt.subplots(figsize=(CW, 1.46))
    labels = {"fdia": "FDIA", "replay": "Replay", "rfdi": "Random false"}
    cols = {"fdia": C1, "replay": C3, "rfdi": C2}
    for k in ("fdia", "replay", "rfdi"):
        dd = np.array(delays[k], float)
        xs = np.sort(dd)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post", color=cols[k],
                label="%s (%d/90)" % (labels[k], len(dd)))
        med = float(np.median(dd))
        ax.axvline(med, color=cols[k], lw=0.4, ls=":")
        dy = {"fdia": 1.005, "replay": 1.10, "rfdi": 1.005}[k]
        ax.annotate("%.1f s" % med, (med, dy),
                    ha={"fdia": "left", "replay": "right",
                        "rfdi": "left"}[k], va="bottom", fontsize=6.5,
                    color=cols[k])
    ax.set_xlabel("detection delay (s)")
    ax.set_ylabel("empirical CDF")
    ax.set_ylim(0, 1.24)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 2000)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1000"])
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig_delay")
    print("[check] fig_delay medians (s):",
          dict((k, float(np.median(v))) for k, v in delays.items()),
          "counts:", dict((k, len(v)) for k, v in delays.items()))

# ---------- fig_wcet ----------
def fig_wcet():
    # Left: code size of both builds from the archived linker maps
    # (stream_det_ram.map .text 0x1943 = 6467; _o2.map 0x1782 = 6018).
    # Right: the only on-chip cycle count on record, -O0 (hil/e6.log g_cyc).
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(CW, 1.10))
    opts = ["-O0", "-O2"]
    tw = [6467, 6018]
    a1.bar(opts, tw, width=0.5, color=[C1, C3])
    a1.set_ylim(0, max(tw) * 1.30)
    for i, c in enumerate(tw):
        a1.annotate("%d" % c, (i, c), ha="center", va="bottom", fontsize=6.5)
    a1.set_ylabel(".text (16-bit words)")
    a1.tick_params(labelsize=6.5)
    a2.bar(["-O0"], [857.684], width=0.45, color=C1)
    a2.set_ylim(0, 857.684 * 1.62)
    a2.annotate("858\n4.29 ms\n4.2% duty", (0, 857.684), ha="center",
                va="bottom", fontsize=6.5, color=INK)
    a2.set_ylabel("cycles / inference (thousands)")
    a2.tick_params(labelsize=6.5)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_wcet")

if __name__ == "__main__":
    fig_latency()
    fig_hil_attack()
    fig_calib()
    fig_closedloop()
    fig_profile()
    fig_delay()
    fig_wcet()
