# -*- coding: utf-8 -*-
"""C2 台架原生模型闭环实验(HIL-only). 前提: C2固件已LOAD."""
import json, os, sys, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
HIL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HIL_DIR, "..")
TSE = os.path.join(HIL_DIR, "ieee33", "ieee33_1ph.tse")
SIG7 = ["v701", "v706", "v712", "v718", "v725", "v733", "i_head"]
CAP = [s + "_rep" for s in SIG7] + ["HIL0 digital input 4"]

tag, biasf = sys.argv[1], sys.argv[2]
dur, onset, hold = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
bias = json.load(open(biasf, encoding="utf-8"))

from typhoon.api import hil
from typhoon.api.schematic_editor import model as mdl
import typhoon.test.capture as cap

mdl.load(TSE)
mdl.compile()
cpd = mdl.get_compiled_model_file(TSE)
if not hil.load_model(str(cpd)):
    raise SystemExit("load_model failed")
hil.start_simulation()
hil.wait_sec(2)
for k, sig in enumerate(SIG7):
    hil.set_analog_output(k + 1, name=sig + "_rep", scaling=1.0, offset=0.0)
for s in ("atk_v701","atk_v703","atk_v706","atk_v709","atk_v712","atk_v715",
          "atk_v718","atk_v725","atk_v730","atk_v33","atk_v733","atk_i_head"):
    hil.set_scada_input_value(s, 0.0)
hil.set_scada_input_value("sw_load724", 1.0)
time.sleep(3)

# 标定门禁
CAL_MU = [0.999298, 0.960948, 0.942742, 0.933489, 0.974254, 0.934239, 0.398344]
CAL_Z  = [3274.866, 3133.090, 3308.696, 3243.529, 3378.359, 3319.263, 3511.054]
for i, c in enumerate(SIG7):
    x = hil.read_analog_signal(c + "_rep")
    z10 = (x - CAL_MU[i]) * CAL_Z[i] / 10.0
    if abs(z10) > 2.0:
        raise SystemExit(f"gate fail: {c} z={z10:.2f}")
print("[gate] OK", flush=True)
time.sleep(19)

t_start = time.time()
r = cap.start_capture(signals=CAP, duration=dur, rate=1000)
time.sleep(onset)
t_on = time.time() - t_start
for k, v in bias.items():
    hil.set_scada_input_value(k, float(v))
time.sleep(hold)
t_off = time.time() - t_start
for k in bias:
    hil.set_scada_input_value(k, 0.0 if k.startswith("atk") else 1.0)
time.sleep(6.0)
df = cap.get_capture_results(r)
arr = np.asarray(df)
np.savez_compressed(os.path.join(ROOT, "results", f"c2_{tag}.npz"),
                    data=arr.astype(np.float32), names=np.array(CAP),
                    meta=np.array(json.dumps({
                        "t_onset": round(t_on, 2), "t_offset": round(t_off, 2),
                        "fs": arr.shape[0] / dur, "bias": bias})))
di4 = arr[:, CAP.index("HIL0 digital input 4")]
i_on = int(t_on * arr.shape[0] / dur)
i_off = int(t_off * arr.shape[0] / dur)
print(f"[{tag}] shape={arr.shape} pre={di4[:i_on].mean():.2f} "
      f"in={di4[i_on:i_off].mean():.2f} post={di4[i_off:].mean():.2f}", flush=True)
hil.stop_simulation()
