# -*- coding: utf-8 -*-
"""Build single-phase IEEE33 equivalent for HIL402 — LoopGuard M1c v2.

v2 (2026-11-10 下午): per-channel export chain for ALL channels:
  Vrms/Irms -> Gain(pu) -> [true Probe v{b}_pu]
                       \-> Sum3(true + Random noise + SCADA attack bias) -> Probe v{b}_rep
  + load-step contactor on bus 724 (SCADA sw_load724)
Removes v1's single-channel tamper prototype.
Terminal facts: p_node/n_node/node, SPSC a_in/a_out/ctrl_in (ctrl_src="Model"),
Sum inputs in/in1/in2, RMS needs sig_output=True, signals need explicit
execution_rate, Probe exposes signals. (TyphoonHIL_API控制指南.md §17)
"""
import json, os, sys, time, traceback

sys.stdout.reconfigure(encoding="utf-8")
HIL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HIL_DIR)
from ieee33_params import F, V_LL, V_LN, LOAD, BRANCH, BRANCH_L, W

from typhoon.api.schematic_editor import model as mdl

TSE = os.path.join(HIL_DIR, "ieee33", "ieee33_1ph.tse")
OUT = os.path.join(HIL_DIR, "mk1ph.json")

FROM_TO = {}
for b in range(701, 718):
    FROM_TO[b] = (b, b + 1)
FROM_TO.update({718: (702, 719), 719: (719, 720), 720: (720, 721), 721: (721, 722),
                722: (703, 723), 723: (723, 724), 724: (724, 725),
                725: (706, 726)})
for b in range(726, 733):
    FROM_TO[b] = (b, b + 1)
BUSES = list(range(701, 734))

V_BUSES = [701, 703, 706, 709, 712, 715, 718, 725, 730, 733]
I_BASE = 500.0
NOISE_SIGMA = 0.003          # pu, per-channel measurement noise std
LOAD_STEP_BUS = 724

rep = {"built": [], "errors": []}
JH, CH = {}, {}

def C(t, n, pos, **props):
    c = mdl.create_component(t, name=n, position=pos)
    for k, v in props.items():
        try:
            mdl.set_component_property(n, k, v)
        except Exception:
            rep["errors"].append(f"prop {n}.{k}={v}: {traceback.format_exc()[-120:]}")
    rep["built"].append(n)
    CH[n] = c
    return c

def conn(a, b):
    ha = JH[a[0]] if a[1] == "" else mdl.term(CH[a[0]], a[1])
    hb = JH[b[0]] if b[1] == "" else mdl.term(CH[b[0]], b[1])
    mdl.create_connection(ha, hb)

mdl.create_new_model(name="ieee33_1ph")
for k, v in [("hil_device", "HIL402"), ("hil_configuration_id", "1")]:
    try:
        mdl.set_model_property_value(k, v)
    except Exception:
        rep["errors"].append(f"model prop {k}: " + traceback.format_exc()[-120:])
try:
    mdl.set_simulation_time_step(1e-5)
except Exception:
    rep["errors"].append("time step: " + traceback.format_exc()[-120:])

X0, DX = 400, 90
YP, YN, YV, YG, YS = 1000, 1600, 640, 460, 300

# ---- source / ground / contactor / head current --------------------------
C("core/Voltage Source", "Vs", (X0 - 200, YP - 40),
  param_set="1phase", init_source_nature="Sine",
  init_rms_value=str(V_LN), init_frequency=str(F), init_phase="0")
C("core/Ground", "gnd", (X0 - 260, YN))
C("core/Single Pole Single Throw Contactor", "SW1", (X0 - 100, YP),
  ctrl_src="Model", resistance="0.01")
C("core/SCADA Input", "sw_ctrl", (X0 - 160, YP - 120),
  def_value="1", min="0", max="1", override_signal_name="True", signal_name="sw_ctrl")
C("core/Current RMS", "Ihead", (X0 + 20, YP),
  frequency=str(F), sig_output="True", execution_rate="1e-4")
C("core/Digital Input", "DI1", (X0 - 100, YP - 200), addr="1")

JH["N0"] = mdl.create_junction(name="N0", position=(X0 - 160, YN))
conn(("Vs", "n_node"), ("gnd", "node"))
conn(("gnd", "node"), ("N0", ""))
conn(("Vs", "p_node"), ("SW1", "a_in"))
conn(("sw_ctrl", "out"), ("SW1", "ctrl_in"))
conn(("SW1", "a_out"), ("Ihead", "p_node"))

for b in BUSES:
    JH[f"P{b}"] = mdl.create_junction(name=f"P{b}", position=(X0 + DX * (b - 700), YP))
    JH[f"N{b}"] = mdl.create_junction(name=f"N{b}", position=(X0 + DX * (b - 700), YN))
conn(("Ihead", "n_node"), ("P701", ""))
for b in BUSES[:-1]:
    conn((f"N{b}", ""), (f"N{b + 1}", ""))
conn(("N0", ""), ("N701", ""))

ISO_BRANCH = 725               # DI4(=告警GPIO3)驱动的支路隔离: 切除 726-733 子树
for b, (f, t) in FROM_TO.items():
    r, x = BRANCH[str(b)]
    xm = X0 + DX * (b - 700) + DX // 2
    C("core/Resistor", f"Rb{b}", (xm, YP - 30), resistance=str(r))
    C("core/Inductor", f"Lb{b}", (xm, YP + 30), inductance=str(BRANCH_L[str(b)]))
    JH[f"M{b}"] = mdl.create_junction(name=f"M{b}", position=(xm, YP))
    if b == ISO_BRANCH:
        C("core/Single Pole Single Throw Contactor", "SWiso", (xm - 40, YP - 60),
          ctrl_src="Digital input", Sa="4", Sa_logic="active low", resistance="0.01")  # 健康DI4=0->闭合; 告警=1->跳闸隔离
        conn((f"P{f}", ""), ("SWiso", "a_in"))
        conn(("SWiso", "a_out"), (f"Rb{b}", "p_node"))
    else:
        conn((f"P{f}", ""), (f"Rb{b}", "p_node"))
    conn((f"Rb{b}", "n_node"), (f"M{b}", ""))
    conn((f"M{b}", ""), (f"Lb{b}", "p_node"))
    conn((f"Lb{b}", "n_node"), (f"P{t}", ""))

for b in BUSES:
    P3, Q3 = LOAD[str(b)]
    if P3 < 1e3:
        continue
    Pp, Qp = P3 / 3.0, Q3 / 3.0
    Rl = V_LN ** 2 / (Pp if Qp <= 1 else (Pp ** 2 + Qp ** 2) ** 0.5)
    Ll = V_LN ** 2 / (Qp * W) if Qp > 1 else 1e6
    xl = X0 + DX * (b - 700)
    if b == LOAD_STEP_BUS:
        # load-step contactor in series with the R half (P part) of the load
        C("core/Single Pole Single Throw Contactor", f"SWl{b}", (xl, (YP + YN) // 2 - 60),
          ctrl_src="Model", resistance="0.01")
        C("core/SCADA Input", f"swload{b}", (xl - 60, (YP + YN) // 2 - 140),
          def_value="1", min="0", max="1", override_signal_name="True",
          signal_name=f"sw_load{b}")
        conn((f"swload{b}", "out"), (f"SWl{b}", "ctrl_in"))
        conn((f"P{b}", ""), (f"SWl{b}", "a_in"))
        C("core/Resistor", f"Rl{b}", (xl, (YP + YN) // 2 - 20), resistance=str(round(Rl, 3)))
        conn((f"SWl{b}", "a_out"), (f"Rl{b}", "p_node"))
    else:
        C("core/Resistor", f"Rl{b}", (xl, (YP + YN) // 2 - 20), resistance=str(round(Rl, 3)))
        conn((f"P{b}", ""), (f"Rl{b}", "p_node"))
    conn((f"Rl{b}", "n_node"), (f"N{b}", ""))
    # v1 简化:负载 Q 并入 R(Z_load 取 |S| 而非 P),省 32 个电感态(核心0矩阵内存 100.83% 超载)
    # Q 独立建模留给 M2(Constant Power Load / Power Meter 轮)

# ---- export chains ---------------------------------------------------------
def export_chain(src_comp, src_term, tag, base):
    """true probe + (noise+bias) sum + reported probe for one channel."""
    xb = X0 + DX * (base - 700) if base else X0 + 40
    C("core/Gain", f"G_{tag}", (xb, YG),
      gain=str(round(1.0 / (V_LN if tag.startswith("v") else I_BASE), 8)),
      execution_rate="1e-4")
    conn((src_comp, src_term), (f"G_{tag}", "in"))
    C("core/Probe", f"Pt_{tag}", (xb, YG - 80),
      override_signal_name="True", signal_name=f"{tag}_pu")
    conn((f"G_{tag}", "out"), (f"Pt_{tag}", "in"))
    C("core/Random Source", f"Nz_{tag}", (xb, YS),
      mean="0", stddev=str(NOISE_SIGMA), execution_rate="1e-4")
    C("core/SCADA Input", f"Atk_{tag}", (xb + 60, YS),
      def_value="0", min="-0.5", max="0.5", override_signal_name="True",
      signal_name=f"atk_{tag}")
    C("core/Sum", f"Rp_{tag}", (xb, YS + 100),
      signs="+++", execution_rate="1e-4")
    conn((f"G_{tag}", "out"), (f"Rp_{tag}", "in"))
    conn((f"Nz_{tag}", "out"), (f"Rp_{tag}", "in1"))
    conn((f"Atk_{tag}", "out"), (f"Rp_{tag}", "in2"))
    C("core/Probe", f"Pr_{tag}", (xb, YS + 180),
      override_signal_name="True", signal_name=f"{tag}_rep")
    conn((f"Rp_{tag}", "out"), (f"Pr_{tag}", "in"))

for b in V_BUSES:
    xb = X0 + DX * (b - 700)
    C("core/Voltage RMS", f"Vr{b}", (xb, YV),
      frequency=str(F), sig_output="True", execution_rate="1e-4")
    conn((f"P{b}", ""), (f"Vr{b}", "p_node"))
    conn((f"N{b}", ""), (f"Vr{b}", "n_node"))
    export_chain(f"Vr{b}", "out", f"v{b}", b)
export_chain("Ihead", "out", "i_head", 0)

mdl.save_as(TSE)
mdl.close_model()
rep["n_components"] = len(rep["built"])
print(f"[build] {rep['n_components']} components, errors={len(rep['errors'])}")
for e in rep["errors"][:10]:
    print("   ERR", e)

# ---- compile ---------------------------------------------------------------
t0 = time.time()
info = {"tse": TSE, "n_components": rep["n_components"], "build_errors": rep["errors"]}
try:
    mdl.load(TSE)
    ok = mdl.compile()
    msgs = mdl.last_compilation_messages()
    info.update({"compile_ok": bool(ok), "elapsed_s": round(time.time() - t0, 1),
                 "messages": str(msgs)[-2500:]})
    print("[compile]", ok, info["elapsed_s"], "s")
    print(str(msgs)[-1000:])
except Exception:
    info["compile_error"] = traceback.format_exc()[-2000:]
    print("[compile] EXCEPTION")
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(info, fh, ensure_ascii=False, indent=1)
print("[done] ->", OUT)
