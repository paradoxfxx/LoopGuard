# LoopGuard

Streaming False-Data-Injection Detection and Closed-Loop Automatic Response on a Distribution-Feeder Hardware-in-the-Loop Testbed

## What This Is

LoopGuard is a streaming FDIA detector that runs entirely on a TMS320F28377D DSP (200 MHz C28x core) connected to a Typhoon HIL402 real-time simulator. The closed loop is:

```
HIL402 (33-bus feeder + tamper layer)
  → analog outputs (7 ch, pu voltages)
    → DSP ADC (AD7606, 1 kHz)
      → int8 quantized causal GRU (30,245 params)
        → alarm GPIO
          → digital input → isolation contactor
```

## Repository Structure

- `hil/` - HIL testbed: model generator, feeder parameters, calibration, episode capture
- `dsp/fw/` - F28377D firmware source, linker script, int8 weight headers
- `dsp/dss/` - Resident JTAG debug probe and host driver
- `scripts/` - Training, evaluation, stealth attack construction, figures
- `models/` - Model architectures (causal GRU, causal SSM, shared stem)
- `ckpt/` - Model checkpoints (deployed seed, bench-native C2, LSTM-AE baseline)
- `results/` - Derived metrics (JSON) backing every table in the paper
- `figures/` - Paper figures (vector PDF)

## Key Results

| Detector | v1 attack | Load step | On-chip |
|----------|-----------|-----------|---------|
| z-score | FIRE | FIRE (false alarm) | yes |
| LSTM-AE | FIRE | FIRE (false alarm) | feasible |
| Sim-trained GRU | quiet | quiet (correct) | yes |
| **Bench-native GRU (C2)** | **FIRE** | **quiet (correct)** | **yes** |

## Requirements

- Python 3.14, PyTorch 2.11+, numpy, matplotlib, scikit-learn
- TI CGT cl2000 (COFF ABI), CCS scripting
- Typhoon HIL Control Center 2026.3
- Hardware: HIL402, F28377D board, AD7606, XDS100-class JTAG

## License

MIT

## Citation

If you use this code, please cite the associated manuscript.

## Note

Firmware runs RAM-resident only. No flash programming occurs at any point.
