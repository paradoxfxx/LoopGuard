/* lg_probe.js (LoopGuard, 从 p7probe.js 20261108 派生) — P8 三相 12 路 20261108 台架的常驻 DSS 会话（Rhino）
 * 铁律：只 loadProgram 到 RAM，绝不碰 Flash。
 * 镜像路径从 Z:/TEMP/iddk_test/p7_out.txt 第一行读（'|' 前是 .out）。
 * 符号地址从 Z:/TEMP/iddk_test/p7_addr.js 读。
 * ccxml 用反斜杠绝对路径：DSS 会把 "E:/xxx" 当驱动器相对路径拼到 CWD（今天踩过）。
 * 命令协议同 P7：PING|LOAD|RUN|HALT|ST|RD|WR|F|WF|U|Q|G|EP|RELOADADDR|QUIT
 */
importPackage(Packages.com.ti.ccstudio.scripting.environment);
var script = ScriptingEnvironment.instance();
var Z = "Z:/TEMP/loopguard_dss/";
var flog = new java.io.FileWriter(Z + "lg_probe.log", true);
function log(x) { flog.write(x + "\n"); flog.flush(); }
function sleep(ms) { java.lang.Thread.sleep(ms); }
function readLine(path) {
    var br = new java.io.BufferedReader(new java.io.FileReader(path));
    var l = br.readLine(); br.close(); return l;
}
var PG = 1;
function rd(a) { return Number(s.memory.readWord(PG, a)) & 0xFFFF; }
function wr16(a, v) { s.memory.writeWord(PG, a, v & 0xFFFF); }
function hex(v) { return "0x" + ((v >>> 0).toString(16)); }
function ul(a) { return rd(a) + 65536 * rd(a + 1); }
function bits(f) { return java.lang.Float.floatToRawIntBits(f); }
function f2w(a, f) { var b = bits(f); wr16(a, b & 0xFFFF); wr16(a + 1, (b >>> 16) & 0xFFFF); }
function w2f(a) { var b = (rd(a) + 65536 * rd(a + 1)) | 0;
    return java.lang.Float.intBitsToFloat(b); }

load(Z + "lg_addr.js");
var DAT = 0x7F00;
function A_(n) { if (typeof A[n] === "undefined") throw "地址表里没有 " + n; return A[n]; }

var ds = script.getServer("DebugServer.1");
ds.setConfig("C:\\ti\\c2000\\C2000Ware_5_02_00_00\\device_support\\f2837xd\\common\\targetConfigs\\TMS320F28377D.ccxml");
var s = ds.openSession(".*C28xx_CPU1");
log("session opened");
s.target.connect();
s.target.halt();
log("CONNECT OK + HALTED");

function loadNow() {
    var line = String(readLine(Z + "lg_out.txt") || "");
    var out = line.split("|")[0];
    var t0 = java.lang.System.currentTimeMillis();
    s.target.halt();
    s.memory.loadProgram(out);
    var ms = java.lang.System.currentTimeMillis() - t0;
    log("LOAD_MS " + ms + "  " + out);
    load(Z + "lg_addr.js");
    s.target.runAsynch();
    log("RUNNING");
}
loadNow();
log("READY");

var alive = true, guard = 0;
while (alive && guard++ < 200000) {
    sleep(120);
    var cf = new java.io.File(Z + "lg_cmd.txt");
    if (!cf.exists()) { continue; }
    var br2 = new java.io.BufferedReader(new java.io.FileReader(cf));
    var c = br2.readLine(); br2.close();
    cf["delete"]();
    c = (c === null) ? "" : "" + c;
    c = c.replace(/^[\s]+/, '').replace(/[\s]+$/, '');
    if (c.length > 0) {
        log("CMD " + c);
        var p = c.split(" ");
        try {
            if (p[0] === "PING") { log("PONG"); }
            else if (p[0] === "LOAD") { loadNow(); }
            else if (p[0] === "RELOADADDR") { load(Z + "lg_addr.js"); log("RELOADADDR ok g_stat=" + A_("g_stat")); }
            else if (p[0] === "RUN") { s.target.runAsynch(); log("RUN set"); }
            else if (p[0] === "HALT") { s.target.halt(); log("HALTED"); }
            else if (p[0] === "ST") {
                log("ST evt_lo=" + (rd(A_("g_stat") + 4) + 65536 * rd(A_("g_stat") + 5)) + " per_lo=" + (rd(A_("g_stat") + 8) + 65536 * rd(A_("g_stat") + 9)));
            }
            else if (p[0] === "RD") {
                var a0 = parseInt(p[1], 16), n0 = Number(p[2]), hs = [];
                for (var i = 0; i < n0; i++) hs.push(hex(rd(a0 + i)));
                log("RD " + p[1] + " " + n0 + ": " + hs.join(" "));
            }
            else if (p[0] === "WR") { var aw = parseInt(p[1], 16); wr16(aw, Number(p[2])); log("WR " + p[1] + " = " + p[2] + " 回读 " + rd(aw)); }
            else if (p[0] === "F") {
                var af = parseInt(p[1], 16), nf = Number(p[2]), vs = [];
                for (i = 0; i < nf; i++) vs.push(Number(w2f(af + 2 * i)).toPrecision(8));
                log("F " + p[1] + " " + nf + ": " + vs.join(" "));
            }
            else if (p[0] === "U") {
                var au = parseInt(p[1], 16), nu = Number(p[2]), us = [];
                for (i = 0; i < nu; i++) us.push(ul(au + 2 * i));
                log("U " + p[1] + " " + nu + ": " + us.join(" "));
            }
            else if (p[0] === "WF") {
                var wb = parseInt(p[1], 16);
                for (i = 2; i < p.length; i++) f2w(wb + 2 * (i - 2), Number(p[i]));
                var rb = [];
                for (i = 2; i < p.length; i++) rb.push(Number(w2f(wb + 2 * (i - 2))).toPrecision(8));
                log("WF " + p[1] + " 写 " + (p.length - 2) + " 个 float 回读 " + rb.join(" "));
            }
            else if (p[0] === "Q") {
                var aq = parseInt(p[1], 16), nq = Number(p[2]), qs = [];
                for (i = 0; i < nq; i++) {
                    var lo = rd(aq + 4 * i) + 65536 * rd(aq + 4 * i + 1);
                    var hi = rd(aq + 4 * i + 2) + 65536 * rd(aq + 4 * i + 3);
                    qs.push(lo + 4294967296 * hi);
                }
                log("Q " + p[1] + " " + nq + ": " + qs.join(" "));
            }
            else if (p[0] === "QUIT") { alive = false; }
            else { log("未知命令 " + c); }
        } catch (e) { log("命令异常 " + e); }
    }
    if (guard % 300 === 0) { log("ALIVE guard=" + guard); }
    sleep(120);
}
log("BYE");
