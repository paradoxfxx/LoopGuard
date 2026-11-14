# -*- coding: utf-8 -*-
"""v2 隐蔽攻击向量——修正量纲与方向(与v1同向:负荷减少→电压升→电流减)."""
import json, os, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"E:\NewPaper20261110"
sys.path.insert(0, os.path.join(ROOT, "hil"))
from ieee33_params import LOAD, BRANCH, V_LN, V_LL  # noqa

BUSES = list(range(701, 734))
V_BUSES = [701, 706, 712, 718, 725, 733]
SIG7 = ["v701", "v706", "v712", "v718", "v725", "v733", "i_head"]
I_BASE = 500.0  # A (pu base)

FROM_TO = {}
for b in range(701, 733):
    if b <= 717:      f, t = b, b+1
    elif b == 718:    f, t = 702, 719
    elif b in (719,720,721): f, t = b, b+1
    elif b == 722:    f, t = 703, 723
    elif b in (723,724): f, t = b, b+1
    elif b == 725:    f, t = 706, 726
    else:             f, t = b, b+1
    FROM_TO[b] = (f, t)

def path_to_source(bus):
    path, cur = [], bus
    while cur != 701:
        for bid, (f, t) in FROM_TO.items():
            if t == cur:
                path.append(bid); cur = f; break
    return list(reversed(path))

PATHS = {b: path_to_source(b) for b in BUSES}

def make_v2(target_bus, target_shift_pu):
    """物理一致攻击: 虚构负荷减少→电压升→电流减(与v1同向)."""
    path_k = PATHS[target_bus]
    R_path = [BRANCH[str(b)][0] for b in path_k]
    cum_R = np.cumsum(R_path)
    bus_R = {FROM_TO[bid][1]: cum_R[i] for i, bid in enumerate(path_k)}

    # ΔP (虚构负荷减少量, W, 单相) → 电压升高
    R_tgt = bus_R.get(target_bus, max(bus_R.values()))
    if R_tgt < 0.01: R_tgt = 0.01
    delta_P = target_shift_pu * V_LN * V_LN / R_tgt

    a = np.zeros(7)
    for ch, bus in enumerate(V_BUSES):
        if bus in bus_R and bus_R[bus] > 0:
            # 负荷减少 → 电压升高 → 正偏置 (与v1同向)
            a[ch] = +bus_R[bus] / R_tgt * target_shift_pu
    # i_head: 负荷减少 → 电流减少 → 负偏置 (pu)
    a[6] = -(delta_P / V_LN) / I_BASE  # A→pu
    return a, delta_P, bus_R

def make_v1(target_bus, delta):
    chain = {701: 0, 703: 1, 706: 2, 709: 3, 712: 4, 715: 5, 718: 6}
    kpos = chain.get(target_bus, 4)
    a = np.zeros(7)
    for i, b in enumerate(V_BUSES):
        if b in chain and chain[b] <= kpos:
            a[i] = delta * (chain[b] + 1) / (kpos + 1)
    a[6] = -0.6 * delta
    return a

print("=" * 65)
print("v1(线性渐变) vs v2(灵敏度, 物理一致) — 同向: 负荷减少")
print("=" * 65)
scenarios = []
for target_bus in [706, 712, 718]:
    for shift in [0.012, 0.024]:
        a_v2, dP, bus_R = make_v2(target_bus, shift)
        a_v1 = make_v1(target_bus, shift)

        bias_v2 = {f"atk_{ch}": round(float(a_v2[i]), 5)
                   for i, ch in enumerate(SIG7) if abs(a_v2[i]) > 1e-6}
        scenarios.append({
            "target": target_bus, "shift_pu": shift,
            "delta_P_kW_3ph": round(dP * 3 / 1000, 1),
            "bias_v2": bias_v2,
        })

        if target_bus == 712 and shift == 0.012:
            print(f"\n目标 {target_bus}, 偏移 {shift:.3f} pu ({shift/0.003:.0f}σn):")
            print(f"  ΔP = {dP*3/1000:.1f} kW (三相, 负荷减少)")
            print(f"  {'通道':10s} {'v1(渐变)':>10s} {'v2(灵敏度)':>10s}  差异")
            for i, ch in enumerate(SIG7):
                d = abs(a_v1[i] - a_v2[i])
                mark = " ←" if d > 0.003 else ""
                print(f"  {ch:10s} {a_v1[i]:+.4f} {a_v2[i]:+.4f}  {mark}")

print(f"\n[done] {len(scenarios)} 个攻击向量")
out = os.path.join(ROOT, "results", "stealth_v2_vectors.json")
json.dump(scenarios, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"-> {out}")

# 验证: v2 的 i_head 量级
for sc in scenarios:
    if sc["target"] == 712 and sc["shift_pu"] == 0.012:
        i_bias = sc["bias_v2"].get("atk_i_head", 0)
        print(f"\n验证 i_head: v2偏置={i_bias} pu, 对应电流变化={i_bias*500:.1f} A")
        print(f"  负荷变化 {sc['delta_P_kW_3ph']} kW → 电流变化 ≈ {sc['delta_P_kW_3ph']*1000/3/7310:.1f} A")
        print(f"  匹配度: {abs(i_bias*500 - sc['delta_P_kW_3ph']*1000/3/7310) < 0.5}")
