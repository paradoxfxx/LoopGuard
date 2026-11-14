# -*- coding: utf-8 -*-
"""LoopGuard 常驻 DSS 探针主机驱动 (P8 命令文件协议):
   Z:/TEMP/loopguard_dss/lg_cmd.txt 下命令, lg_probe.log 收响应。
   用法: python lg_host.py [ping|load|status|knock|quit]"""
import json, os, subprocess, sys, time

sys.stdout.reconfigure(encoding="utf-8")
ZID = r"Z:\TEMP\loopguard_dss"
os.makedirs(ZID, exist_ok=True)
DSS = r"C:/ti/ccs1270/ccs/ccs_base/scripting/bin/dss.bat"
JS = r"E:\NewPaper20261110\dsp\dss\lg_probe.js"
LOG = os.path.join(ZID, "lg_probe.log")
CMD = os.path.join(ZID, "lg_cmd.txt")


def addrs():
    a = json.load(open(os.path.join(ZID, "lg_addr.json"), encoding="utf-8"))
    return {k.lstrip("_"): v for k, v in a.items()}


class Probe:
    def __init__(self):
        self.pos = os.path.getsize(LOG) if os.path.exists(LOG) else 0

    def _new_lines(self):
        if not os.path.exists(LOG) or os.path.getsize(LOG) <= self.pos:
            return []
        with open(LOG, "r", encoding="utf-8", errors="replace") as fh:
            fh.seek(self.pos)
            lines = fh.read().splitlines()
        self.pos = os.path.getsize(LOG)
        return lines

    def cmd(self, line, timeout=20.0):
        with open(CMD, "w", encoding="ascii") as fh:
            fh.write(line)
        t0 = time.time()
        seen_cmd_echo = False
        while time.time() - t0 < timeout:
            time.sleep(0.15)
            for l in self._new_lines():
                if l.startswith("CMD "):
                    seen_cmd_echo = True
                elif seen_cmd_echo and not l.startswith("ALIVE"):
                    return l
        return None

def start():
    subprocess.Popen([DSS, JS], cwd=ZID,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < 120:
        time.sleep(1.0)
        if os.path.exists(LOG):
            with open(LOG, "r", encoding="utf-8", errors="replace") as fh:
                t = fh.read()
            if "READY" in t:
                print("[probe] READY in %.0fs" % (time.time() - t0))
                return True
    print("[probe] NOT READY")
    return False


def rd_words(a, n):
    p = Probe()
    r = p.cmd("RD %X %d" % (a, n))
    if r and r.startswith("RD"):
        return [int(x, 16) for x in r.split(":", 1)[1].split()]
    return None


def main():
    act = sys.argv[1] if len(sys.argv) > 1 else "status"
    A = addrs()
    if act == "start":
        start()
        return
    p = Probe()
    if act == "ping":
        print(p.cmd("PING"))
    elif act == "load":
        print(p.cmd("LOAD"))
    elif act == "status":
        def u(addr):                      # 32位计数器
            w = rd_words(addr, 2)
            return (w[0] + 65536 * w[1]) if w else None
        for n in ("g_samp", "g_tick", "g_alarms"):
            print(n, "=", u(A[n]))
        for n in ("g_pred", "g_alarm", "g_atk_run"):
            r = rd_words(A[n], 1)
            print(n, "=", r[0] if r else None)
        r = p.cmd("F %X 12" % A["g_x"])
        print("g_x =", r.split(":", 1)[1] if r else None)
        print("g_cyc/cycmax =", u(A["g_cyc"]), u(A["g_cycmax"]))
    elif act == "knock":
        print(p.cmd("WR %X 1" % A["g_knock"]))
        time.sleep(1.5)
        p2 = Probe()
        r = rd_words(A["g_alarm"], 1)
        r2 = rd_words(A["g_alarms"], 2)
        print("alarm =", r[0] if r else None,
              " alarms =", (r2[0] + 65536 * r2[1]) if r2 else None)
    elif act == "quit":
        print(p.cmd("QUIT"))


if __name__ == "__main__":
    main()
