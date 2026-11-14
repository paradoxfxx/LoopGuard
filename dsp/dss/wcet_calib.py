# -*- coding: utf-8 -*-
"""E5 刻度三互证之拍频路数: g_samp 长窗拍频(期望~1000Hz) + g_cyc 双档读取。"""
import sys, time
sys.path.insert(0, r"E:\NewPaper20261110\dsp\dss")
import lg_host as LH
A = LH.addrs()

def u32(n):
    w = LH.rd_words(A[n], 2)
    return (w[0] + 65536 * w[1]) if w else None

print("LOAD:", LH.Probe().cmd("LOAD"))
t0 = time.time(); s0 = u32("g_samp")
time.sleep(60)
t1 = time.time(); s1 = u32("g_samp")
dt = t1 - t0
rate = (s1 - s0) / dt
print(f"g_samp: {s0} -> {s1} over {dt:.2f}s => {rate:.2f} Hz (期望~1000; 偏差={100*(rate-1000)/1000:+.2f}%)")
for n in ("g_cyc", "g_cycmax"):
    print(n, "=", u32(n))
