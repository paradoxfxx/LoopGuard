# -*- coding: utf-8 -*-
"""LoopGuard M3 固件无头构建:编译(-O0) -> RAM 重链 -> 符号地址表导出。
配方照抄 P7 (p7_bench_build.py, 已验证); 铁律: 只 RAM 镜像。"""
import sys, os, re, subprocess, datetime, json, io
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SD = os.path.dirname(os.path.abspath(__file__))
CC = 'C:/ti/ti-cgt-c2000_22.6.0.LTS/bin/cl2000'
CGT = 'C:/ti/ti-cgt-c2000_22.6.0.LTS'
CW = 'C:/ti/c2000/C2000Ware_5_02_00_00'
RTW = r'E:/Paper2026Win/P1_LADRC_TD3/30_dsp/MBD_shell/P1_RL_ADRC_37xD_ert_rtw'
CMD = os.path.join(SD, 'ram_lnk_stream.cmd')
PERIPH = 'C:/Matlab/R2025b/toolbox/c2b/tic2000_shared/src/c2837xDPeripherals.cmd'
ZID = r'Z:\TEMP\loopguard_dss'
os.makedirs(ZID, exist_ok=True)

INC = ['-I' + CGT + '/include',
       '-I' + CW + '/device_support/f2837xd/common/include',
       '-I' + CW + '/device_support/f2837xd/headers/include',
       '-I' + CW + '/device_support/f2837xd/common',
       '-I' + CW + '/driverlib/f2837xd/driverlib', '-I' + SD]
DEF = ['-DF2837X_REG_FORMAT', '-DF28377D', '-DCPU1', '-DSTACK_SIZE=1024']
BASE = ['--abi=coffabi', '--compile_only', '--preproc_with_compile', '--large_memory_model',
        '-v28', '--float_support=fpu32', '-ml', '--tmu_support=tmu0', '--fp_mode=relaxed']
DRIVER_OBJS = ['F2837xD_SysCtrl.obj', 'F2837xD_Gpio.obj', 'F2837xD_PieCtrl.obj',
               'F2837xD_GlobalVariableDefs.obj', 'F2837xD_CodeStartBranch.obj',
               'F2837xD_usDelay.obj', 'F2837xD_CpuTimers.obj', 'F2837xD_Adc.obj',
               'F2837xD_PieVect.obj', 'F2837xD_DefaultISR.obj', 'memcpy_fast.obj']
missing = [o for o in DRIVER_OBJS if not os.path.exists(os.path.join(RTW, o))]
if missing:
    raise SystemExit('missing driver objs: %s' % missing)

OPT = '-O2' if '--opt2' in sys.argv else '-O0'
SUF = '_o2' if OPT == '-O2' else ''
d = os.path.join(SD, 'obj' + (SUF or '_O0'))
os.makedirs(d, exist_ok=True)

def compile_(src):
    obj = os.path.join(d, os.path.basename(src).replace('.c', '.obj'))
    if os.path.exists(obj):
        os.remove(obj)
    r = subprocess.run([CC] + BASE + DEF + [OPT, '-fs' + d, '-fr' + d] + INC + [src],
                       cwd=SD, capture_output=True, text=True, errors='replace')
    msg = (r.stdout or '') + (r.stderr or '')
    ok = os.path.exists(obj)
    bad = [l[:180] for l in msg.split('\n') if re.search(r'error|abort|invalid', l, re.I)]
    print('  compile %-20s rc=%d obj=%s %s' % (os.path.basename(src), r.returncode,
          'OK' if ok else 'MISSING', '' if not bad else '| ' + ' ; '.join(bad[:4])))
    if not ok:
        print(msg[:3000]); raise SystemExit(src + ' compile failed')
    return obj

objs = [compile_(os.path.join(SD, 'stream_det.c')),
        compile_(os.path.join(CW, 'device_support/f2837xd/common/source/F2837xD_Emif.c'))]

out = os.path.join(SD, 'stream_det_ram%s.out' % SUF)
mapf = out.replace('.out', '.map')
lk = [CC, '--abi=coffabi', '-z', '-I' + CGT + '/lib', '--stack_size=1024',
      '--warn_sections', '--reread_libs', '--rom_model', '--priority',
      '--output_file=' + out.replace('\\', '/'), '-m' + mapf.replace('\\', '/'),
      '-lrts2800_fpu32.lib', '--define', 'F28377D', '--define', 'CPU1',
      '--define', 'BOOT_FROM_FLASH=0', '--define', 'BOOT_USING_BL=0'] + \
     [PERIPH, CMD.replace('\\', '/')] + [o.replace('\\', '/') for o in objs] + \
     [os.path.join(RTW, o).replace('\\', '/') for o in DRIVER_OBJS]
r = subprocess.run(lk, cwd=SD, capture_output=True, text=True, errors='replace')
msg = (r.stdout or '') + (r.stderr or '')
ok = os.path.exists(out)
print('  link rc=%d image=%s' % (r.returncode, 'OK' if ok else 'FAILED'))
if not ok:
    print(msg[:3000]); raise SystemExit('link failed')

txt = io.open(mapf, encoding='utf-8', errors='replace').read()
flash = [m.group(1) for m in re.finditer(r'(?m)^\S[^\n]*?\s+([01])\s+(00[0-9a-f]{6})\s', txt)
         if int(m.group(2), 16) >= 0x80000]
print('  flash-zone sections: %d (must be 0)' % len(flash))
for sec in ('.text', '.econst', '.ebss', '.cinit', '.stack'):
    i = txt.find('\n' + sec + ' ')
    print('  %-9s %s' % (sec, ' '.join(txt[i:i + 70].split()) if i > 0 else '(none)'))

addr = {}
for m in re.finditer(r'(?m)^([0-9a-f]{8})\s+[0-9a-f]{3}\s+\([^)]*\)\s+_(g_\w+)$', txt):
    addr[m.group(2)] = int(m.group(1), 16)
need = ['g_run', 'g_pred', 'g_alarm', 'g_atk_run', 'g_knock', 'g_tick', 'g_samp',
        'g_alarms', 'g_cyc', 'g_cycmax', 'g_raw', 'g_x', 'g_h0', 'g_h1',
        'g_logits', 'g_vscale']
miss = [n for n in need if n not in addr]
if miss:
    raise SystemExit('missing symbols: %s (got %d)' % (miss, len(addr)))
io.open(os.path.join(ZID, 'lg_addr.json'), 'w', encoding='utf-8').write(json.dumps(addr, indent=1))
io.open(os.path.join(ZID, 'lg_out.txt'), 'w', encoding='utf-8').write(out.replace('\\', '/'))
print('  %d symbols -> %s' % (len(addr), os.path.join(ZID, 'lg_addr.json')))
print('  image %d bytes  %s' % (os.path.getsize(out),
      datetime.datetime.now().strftime('%m-%d %H:%M:%S')))
