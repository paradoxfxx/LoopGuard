/* stream_det.c — LoopGuard 边缘流式检测固件 (TMS320F28377D, RAM-only)
 * 2026-11-10 M3 v1: AD7606 采 HIL AO(7路 pu) -> DiffStem -> int8权重 GRU(d48,2层)
 *   -> 5类头 -> 告警FSM(连续5拍攻击类) -> GPIO3(->HIL DI4), 心跳 GPIO2(->DI3)
 * 节拍: 主循环轮询 Timer0 (1 kHz 采样, 100样本均值 -> 10 Hz 推理)
 *   (PIE 中断在本板 RAM 路径有不响应前科, 轮询规避 — F28377D指南 §12.2)
 * 权重: stream_det_w.h 由 scripts/export_weights.py 从 ckpt 生成 (int8/行尺度)
 * 铁律: 只 RAM 下载; 全部旋钮 volatile 全局.
 */
#include "F2837xD_device.h"
#include "F2837xD_Examples.h"
#include "F2837xD_gpio.h"
#include "F2837xD_sysctrl.h"
#include "F2837xD_cputimer.h"
#include <math.h>
#include "stream_det_w.h"
#include "stream_det_cal.h"

#define AD_BASE      0x300000        /* EMIF CS3, AD7606 (16-bit 读序 = 通道序) */
#define AD_CONVST    14              /* GPIO14 */
#define AD_RESET     15              /* GPIO15 */
#define AD_CS4       44              /* GPIO44 常高 */
#define GPIO_ALRM    3               /* GPIO3 -> HIL DI4 (门极接线惯例 GPIOk->DIk+1) */
#define GPIO_HB      2               /* GPIO2 -> HIL DI3 */

#define D           24  /* C2 bench-native d=24 */
#define M           12
#define CIN         (3*M)
#define DECIM       100              /* 1 kHz / 100 = 10 Hz 推理 */

/* ---------- 旋钮/读出 (volatile, JTAG 可读写) ---------- */
volatile unsigned int g_run   = 1;
volatile unsigned int g_pred  = 0;        /* 最近一拍类别 */
volatile unsigned int g_alarm = 0;        /* 告警锁存(1=在告警态) */
volatile unsigned int g_atk_run = 0;      /* 连续攻击类计数 */
volatile unsigned int g_knock = 0;        /* 置1 => 强制告警一拍(链路自检) */
volatile unsigned int g_inhibit = 0;      /* 置1 => 告警旁路(维护/标定用):FSM与GPIO3全压低 */
volatile unsigned int g_ack = 0;          /* 运维确认: 写1 => 告警锁存解除+输入解冻+环清零重启 */
volatile unsigned int g_frozen = 0;       /* 告警后输入冻结标志(只读) */
volatile unsigned int g_frz_run = 0;      /* 冻结期拍计数(自动确认用) */
volatile unsigned int g_maint = 0;        /* 死输入维护态(AO未绑定) */
volatile unsigned int g_debounce = 25;    /* 告警去抖拍数: 表1实测preFA步率15-31%,5拍撑不住
                                             (0.15^5每秒级误锁),25拍=2.5s持续判攻,真实攻击分钟级 */
volatile unsigned int g_tick  = 0;        /* 10 Hz 拍计数 */
volatile unsigned int g_samp  = 0;        /* 1 kHz 采样计数 */
volatile unsigned int g_alarms = 0;       /* 历史告警沿数 */
volatile unsigned long g_cyc  = 0;        /* 最近一次推理的 CpuTimer1 周期数 */
volatile unsigned long g_cycmax = 0;
volatile float g_raw[16];                 /* 最近一次 AD7606 原始码(电压) */
volatile float g_x[M];                    /* 最近一拍输入(已归一) */
volatile float g_h0[D], g_h1[D];          /* GRU 隐态(可视) */
volatile float g_logits[5];
volatile float g_vscale[M];               /* 每通道量程(码->V), 默认 10.0/32768 */
volatile float g_zgain = 1.0f;            /* 总增益旋钮;CAL_Z已含池化因子(10/sigma_raw) */

#define L_RING 128
#define BASE_Q 32                  /* 训练stem的基线=窗口最旧1/4 */
/* ---------- GRU 状态 ---------- */
static float h0[D], h1[D];
static float xring[L_RING][M];     /* 最近128拍输入环(训练口径stem用) */
static int ring_n = 0;             /* 已填充数(<128时为预热) */

/* ---------- tanh LUT (129 点, 启动时填) ---------- */
static float lut_tanh[129];
static void lut_init(void)
{
    int i;
    for (i = 0; i < 129; ++i) {
        float t = -4.0f + 8.0f * i / 128.0f;
        lut_tanh[i] = tanhf(t);
    }
}
static float tanh_lut(float x)             /* ±4 饱和, 过渡带线性插值 */
{
    float t;
    if (x <= -4.0f) return -1.0f;
    if (x >= 4.0f) return 1.0f;
    t = (x + 4.0f) * 16.0f;
    {
        int i = (int)t;
        float f = t - i;
        return lut_tanh[i] + f * (lut_tanh[i + 1] - lut_tanh[i]);
    }
}
static inline float sigmoid_lut(float x) { return 0.5f * (tanh_lut(0.5f * x) + 1.0f); }

/* ---------- int8 权重矩阵向量积: y = b + s .* (Q * x) ---------- */
static void matvec_i8(const signed char *q, const float *s, const float *b,
                      const float *x, int rows, int cols, float *y)
{
    int i, j;
    for (i = 0; i < rows; ++i) {
        float acc = b ? b[i] : 0.0f;
        const signed char *qr = q + (long)i * cols;
        for (j = 0; j < cols; ++j)
            acc += (float)qr[j] * x[j];
        y[i] = s[i] * acc;
    }
}

/* ---------- GRU 层 (torch 门槛序 r,z,n; n 的 r 只乘隐态支路) ---------- */
static void gru_layer(const signed char *wih, const signed char *whh,
                      const float *sih, const float *shh,
                      const float *bih, const float *bhh,
                      const float *x, float *h)
{
    static float gi[3 * D], gh[3 * D];
    int i;
    matvec_i8(wih, sih, bih, x, 3 * D, D, gi);      /* 输入支路: [r z n] */
    matvec_i8(whh, shh, bhh, h, 3 * D, D, gh);      /* 隐态支路 */
    for (i = 0; i < D; ++i) {
        float r = sigmoid_lut(gi[i] + gh[i]);
        float z = sigmoid_lut(gi[D + i] + gh[D + i]);
        float n = tanh_lut(gi[2 * D + i] + r * gh[2 * D + i]);
        h[i] = (1.0f - z) * n + z * h[i];
    }
}

/* ---------- 一次推理 (10 Hz) ---------- */
static void infer(const float *x)
{
    static float stem[CIN], proj[D], logits[5];
    int i, pred, atk;
    unsigned long c0, c1;

    /* 告警锁存-冻结-确认机制(2026-11-11 rev): 告警后冻结输入环,仅心跳;
       g_ack 解锁并清环重启(防隔离反馈闭锁: E6 v3 发现) */
    if (g_frozen) {
        if (++g_frz_run > 300) g_ack = 1;   /* 自动确认: 冻结30s后自动重合闸式复位(2026-11-11) */
        if (g_ack) {
            g_alarm = 0; g_frozen = 0; g_atk_run = 0; g_ack = 0; g_frz_run = 0;
            ring_n = 0;                     /* 环清零 -> 重新预热,窗口不含冻结期历史 */
            GpioDataRegs.GPACLEAR.all |= (1UL << GPIO_ALRM);  /* 立即拉低:隔离必须马上复归,
                                                    否则重预热期i_head停在-424sigma(2026-11-11真凶) */
        }
        g_tick++;
        if (g_tick & 1) GpioDataRegs.GPASET.all |= (1UL << GPIO_HB);
        else            GpioDataRegs.GPACLEAR.all |= (1UL << GPIO_HB);
        return;
    }
    /* DiffStem(训练口径,逐位一致): 环写入 -> dx 与 基线(最旧BASE_Q拍均值) */
    {
        int pos = ring_n % L_RING;
        int old_pos, bcount;
        float base[M];
        for (i = 0; i < M; ++i) xring[pos][i] = x[i];
        ring_n++;
        if (ring_n < L_RING + BASE_Q) {         /* 预热:窗口未满+基线组未满(下位机不告警) */
            if (!g_alarm) GpioDataRegs.GPACLEAR.all |= (1UL << GPIO_ALRM);
            return;
        }
        bcount = BASE_Q;
        for (i = 0; i < M; ++i) base[i] = 0.0f;
        for (i = 0; i < bcount; ++i) {
            int src = (ring_n - L_RING + i) % L_RING;
            int j;
            for (j = 0; j < M; ++j) base[j] += xring[src][j];
        }
        old_pos = (ring_n - 2 + L_RING * 2) % L_RING;   /* 上一拍 */
        for (i = 0; i < M; ++i) {
            base[i] /= (float)bcount;
            stem[i] = x[i];
            stem[M + i] = x[i] - xring[old_pos][i];
            stem[2 * M + i] = x[i] - base[i];
        }
    }

    c0 = (unsigned long)CpuTimer1Regs.TIM.all;
    matvec_i8(W_in_q, W_in_s, b_in, stem, D, CIN, proj);
    gru_layer(Wih0_q, Whh0_q, sih0, shh0, bih0, bhh0, proj, h0);
    gru_layer(Wih1_q, Whh1_q, sih1, shh1, bih1, bhh1, h0, h1);
    matvec_i8(W_hd_q, W_hd_s, b_hd, h1, 5, D, logits);
    c1 = (unsigned long)CpuTimer1Regs.TIM.all;
    /* 递减秒表跨回绕: TIM 回绕到 PRD(而非0xFFFFFFFF), 差值按 PRD 折算(2026-11-11修) */
    {
        unsigned long prd = ((unsigned long)CpuTimer1Regs.PRD.all) & 0xFFFFFFFFUL;
        g_cyc = (c0 >= c1) ? (c0 - c1) : (c0 + prd - c1);
    }
    if (g_cyc > g_cycmax) g_cycmax = g_cyc;

    for (i = 0; i < 5; ++i) g_logits[i] = logits[i];
    pred = 0;
    if (logits[1] > logits[pred]) pred = 1;
    if (logits[2] > logits[pred]) pred = 2;
    if (logits[3] > logits[pred]) pred = 3;
    if (logits[4] > logits[pred]) pred = 4;
    g_pred = (unsigned int)pred;

    atk = (pred >= 2);
    if (g_knock) atk = 1;
    if (g_inhibit) atk = 0;               /* 旁路: 告警链路全静默 */
    g_atk_run = atk ? (g_atk_run + 1) : 0;
    if (g_atk_run >= g_debounce && !g_alarm) {
        g_alarm = 1; g_alarms++; g_frozen = 1;   /* 锁存+输入冻结,只由 g_ack 解除 */
    }
    /* 告警不再自动清除(锁存制,防隔离反馈闭锁) */
    if (g_knock) g_knock = 0;
    g_tick++;

    /* 执行器交付 */
    if (g_alarm) GpioDataRegs.GPASET.all |= (1UL << GPIO_ALRM);
    else         GpioDataRegs.GPACLEAR.all |= (1UL << GPIO_ALRM);
    if (g_tick & 1) GpioDataRegs.GPASET.all |= (1UL << GPIO_HB);
    else            GpioDataRegs.GPACLEAR.all |= (1UL << GPIO_HB);
}

/* ---------- EMIF / AD7606 ---------- */
extern void setup_emif1_pinmux_async_16bit(Uint16);
static void emif_ad_init(void)
{
    volatile Uint16 *ASYNC_CS3_CR = (volatile Uint16 *)0x0000B121; /* EMIF1_ASYNC_CS3_CR(见C2000Ware) */
    EALLOW;
    /* EMIF1 主锁三道门 (F28377D指南 §11.2): MSEL -> ACCPROT0 -> COMMIT */
    *(volatile Uint32 *)0x0000B350 = 0x93A5CE71;   /* EMIF1MSEL.MSEL_EMIF1 = 1 */
    *(volatile Uint32 *)0x0000B354 = 0;            /* EMIF1ACCPROT0 = 0 */
    *(volatile Uint32 *)0x0000B35C = 1;            /* EMIF1COMMIT = 1 */
    ClkCfgRegs.PERCLKDIVSEL.bit.EMIF1CLKDIV = 1;   /* EMIFCLK = SYSCLK/2 = 100 MHz */
    EDIS;
    setup_emif1_pinmux_async_16bit(0);
    /* CS3: 扩展 strobe 模式, ASYNC_CS3_CR: W_STROBE=R_STROBE=4 (保守, §11.9 ws>=2 才正确) */
    EALLOW;
    *(volatile Uint32 *)0x0000B120 = 0x04122214;   /* CS3 CR 低字(与台账 ws=4 档同族, 若读错改 0x04122a14) */
    EDIS;
}

static Uint16 ad_read_ch(int idx)
{
    volatile Uint16 *p = (volatile Uint16 *)AD_BASE;
    return p[idx];
}

static void ad_convst(void)
{
    GpioDataRegs.GPACLEAR.all |= (1UL << AD_CONVST);
    GpioDataRegs.GPASET.all   |= (1UL << AD_CONVST);   /* 上升沿启动转换 */
    DELAY_US(6);                                       /* 转换时间余量 */
}

/* ---------- main ---------- */
void main(void)
{
    static float acc[16];
    int i, n = 0;

    InitSysCtrl();                      /* PLL 200 MHz (§13.1: 本镜像刻度=200.00 MHz) */
    DINT;
    InitPieCtrl();
    IER = 0x0000; IFR = 0x0000;
    InitPieVectTable();

    lut_init();
    for (i = 0; i < M; ++i) { g_x[i] = 0; g_vscale[i] = 10.0f / 32768.0f; }
    for (i = 0; i < L_RING; ++i) { int j; for (j = 0; j < M; ++j) xring[i][j] = 0.0f; }
    for (i = 0; i < D; ++i) { h0[i] = 0; h1[i] = 0; g_h0[i] = 0; g_h1[i] = 0; }
    for (i = 0; i < 16; ++i) acc[i] = 0;

    /* GPIO: CONVST/RESET/CS4 + 告警/心跳 */
    EALLOW;
    GpioCtrlRegs.GPAMUX1.bit.GPIO2 = 0;  GpioCtrlRegs.GPADIR.bit.GPIO2 = 1;
    GpioCtrlRegs.GPAMUX1.bit.GPIO3 = 0;  GpioCtrlRegs.GPADIR.bit.GPIO3 = 1;
    GpioCtrlRegs.GPAMUX1.bit.GPIO14 = 0; GpioCtrlRegs.GPADIR.bit.GPIO14 = 1;
    GpioCtrlRegs.GPAMUX1.bit.GPIO15 = 0; GpioCtrlRegs.GPADIR.bit.GPIO15 = 1;
    GpioCtrlRegs.GPBMUX1.bit.GPIO44 = 0; GpioCtrlRegs.GPBDIR.bit.GPIO44 = 1;  /* GPB: GPIO32-63 */
    EDIS;
    GpioDataRegs.GPBCLEAR.all |= (1UL << (AD_CS4 - 32));
    GpioDataRegs.GPBSET.all   |= (1UL << (AD_CS4 - 32));
    /* 上电即安全态(2026-11-11 E6v5复盘): 预热16s内不进FSM/GPIO写,
       GPIO3必须先拉低,否则上一镜像的告警电平会让隔离保持断开、毒化首窗 */
    GpioDataRegs.GPACLEAR.all |= (1UL << GPIO_ALRM);
    GpioDataRegs.GPACLEAR.all |= (1UL << GPIO_HB);
    /* AD7606 复位: RESET 高 >=10 ms 再拉低 (§9.1) */
    GpioDataRegs.GPACLEAR.all |= (1UL << AD_RESET);
    GpioDataRegs.GPASET.all   |= (1UL << AD_RESET);
    DELAY_US(20000);
    GpioDataRegs.GPACLEAR.all |= (1UL << AD_RESET);

    emif_ad_init();

    InitCpuTimers();
    ConfigCpuTimer(&CpuTimer0, 200, 1000);   /* 1 ms */
    ConfigCpuTimer(&CpuTimer1, 200, 1000000);/* 自由跑秒表, 1e6 µs 周期 */
    CpuTimer0Regs.TCR.bit.TIF = 1;           /* 清标志 */
    CpuTimer0Regs.TCR.bit.TSS = 0;           /* 启动 Timer0 */
    CpuTimer1Regs.TCR.bit.TIF = 1;
    CpuTimer1Regs.TCR.bit.TSS = 0;           /* 秒表自由跑 */
    EINT; ERTM;

    while (g_run) {
        if (CpuTimer0Regs.TCR.bit.TIF == 1) {      /* 1 kHz 轮询节拍 */
            CpuTimer0Regs.TCR.bit.TIF = 1;         /* 写 1 清标志 */
            ad_convst();
            for (i = 0; i < 16; ++i) {
                Uint16 code = ad_read_ch(i);
                float v = ((float)(short)code) * g_vscale[0]; /* 量程同板统一 ±10V */
                acc[i] += v;
                g_raw[i] = v;
            }
            g_samp++;
            if (++n >= DECIM) {
                /* 7 路 AO 输入映射: 本台架实测 2026-11-10 E6b 校准 = AO_k -> AD元素k-1
                 * (元素8-14为镜像组,勿用; P1 台架的{13,14,15,8..}表在本台架无效) */
                const int map7[7] = {0, 1, 2, 3, 4, 5, 6};
                float xm[M];
                for (i = 0; i < M; ++i) xm[i] = 0.0f;
                /* 逐通道死通道掩蔽(2026-11-11修): 判活用原始电压(非z分数!),
                   死通道(<0.3V,如AO未绑定/部分绑定竞态)按无信息z=0处理,不再打出-3300sigma;
                   7路全死 -> 维护态(不推理不填环) */
                {
                    int alive = 0;
                    for (i = 0; i < 7; ++i) {
                        float rawv = acc[map7[i]] / DECIM;
                        if (rawv > 0.5f) alive = 1;
                        xm[i] = (rawv < 0.3f) ? 0.0f
                                : g_zgain * (rawv - CAL_MU[i]) * CAL_Z[i];
                    }
                    if (!alive) { g_maint = 1; for (i = 0; i < 16; ++i) acc[i] = 0.0f; n = 0; continue; }
                    g_maint = 0;
                }
                for (i = 0; i < M; ++i) g_x[i] = xm[i];
                infer(xm);
                for (i = 0; i < 16; ++i) acc[i] = 0;
                n = 0;
            }
        }
    }
    while (1) { ; }
}
