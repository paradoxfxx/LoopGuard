/* LoopGuard stream_det 固件 RAM 链接（配方照 P7/P1，两条硬规则沿用：
 *   代码段只放 0x8000–0xBFFF（RAMLS0–6，取指窗口）；
 *   数据段只放 0xC000–0xFFFF（16 位数据寻址窗口）。
 * 差异：权重 .econst 较大(~18k字) → 放 0x14000 起的程序空间 GS 区
 * （§11.1 已实测 GS14 0x1A000 的 const 读取可行，6.58 计数/MAC）。 */
MEMORY
{
PAGE 0 :
   RC_BEGIN  : origin = 0x000000, length = 0x000002
   RC_LS     : origin = 0x008000, length = 0x004000   /* 代码 RAMLS0–6 */
   RC_GSC    : origin = 0x014000, length = 0x008000   /* 程序空间常数(权重) GS8–GS15 */
   RC_RESET  : origin = 0x3FFFC0, length = 0x000002

PAGE 1 :
   RD_M1     : origin = 0x000122, length = 0x0006D6   /* RAMM0 尾 + RAMM1：栈 */
   RD_GS     : origin = 0x00C000, length = 0x003800   /* 数据窗口 0xC000–0xF7FF */

   EMIF1_CS2_MEMORY : origin = 0x00100000, length = 0x00200000
   EMIF1_CS3_MEMORY : origin = 0x00300000, length = 0x00080000
   EMIF1_CS4_MEMORY : origin = 0x00380000, length = 0x00060000
}

SECTIONS
{
   codestart          : > RC_BEGIN, PAGE = 0
   .text              : > RC_LS, PAGE = 0
   .switch            : > RC_LS, PAGE = 0
   .cinit             : > RC_LS, PAGE = 0
   .pinit             : > RC_LS, PAGE = 0

   ramfuncs           : > RC_LS, PAGE = 0,
        LOAD_START(_RamfuncsLoadStart), LOAD_SIZE(_RamfuncsLoadSize), LOAD_END(_RamfuncsLoadEnd),
        RUN_START(_RamfuncsRunStart),   RUN_SIZE(_RamfuncsRunSize),   RUN_END(_RamfuncsRunEnd)
   .TI.ramfunc        : > RC_LS, PAGE = 0,
        LOAD_START(_MW_RamfuncsLoadStart), LOAD_SIZE(_MW_RamfuncsLoadSize), LOAD_END(_MW_RamfuncsLoadEnd),
        RUN_START(_MW_RamfuncsRunStart),   RUN_SIZE(_MW_RamfuncsRunSize),   RUN_END(_MW_RamfuncsRunEnd)

   .econst            : > RC_GSC, PAGE = 0
   .const             : > RC_GSC, PAGE = 0

   .ebss              : > RD_GS, PAGE = 1
   .bss               : > RD_GS, PAGE = 1
   .bss:cio           : > RD_GS, PAGE = 1
   .data              : > RD_GS, PAGE = 1
   .esysmem           : > RD_GS, PAGE = 1
   .sysmem            : > RD_GS, PAGE = 1
   .stack             : > RD_M1, PAGE = 1
   .reset             : > RC_RESET, PAGE = 0, TYPE = DSECT
}
