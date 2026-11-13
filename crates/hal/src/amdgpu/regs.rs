//! GFX10.3 (RDNA2) register addresses and bitfields.
//!
//! Addresses are byte offsets as consumed by the PM4 `SET_*_REG` packets and
//! match Mesa's `src/amd/registers/gfx103.json` (which is generated from the
//! hardware spec and is what radeonsi programs).
#![allow(dead_code)]

/// Compile-time bitfield packer: `field(value, shift, mask)`.
#[inline(always)]
pub const fn f(v: u32, shift: u32, mask: u32) -> u32 {
    (v & mask) << shift
}

// Shader (SH) registers, SH base 0x2C00.
pub const SPI_SHADER_PGM_RSRC4_PS: u32 = 0x0B004;
pub const SPI_SHADER_PGM_RSRC3_PS: u32 = 0x0B01C;
pub const SPI_SHADER_PGM_LO_PS: u32 = 0x0B020;
pub const SPI_SHADER_PGM_HI_PS: u32 = 0x0B024;
pub const SPI_SHADER_PGM_RSRC1_PS: u32 = 0x0B028;
pub const SPI_SHADER_PGM_RSRC2_PS: u32 = 0x0B02C;
pub const SPI_SHADER_USER_DATA_PS_0: u32 = 0x0B030;

pub const SPI_SHADER_REQ_CTRL_PS: u32 = 0x0B0C0;
pub const SPI_SHADER_USER_ACCUM_PS_0: u32 = 0x0B0C8;
pub const SPI_SHADER_USER_ACCUM_PS_1: u32 = 0x0B0CC;
pub const SPI_SHADER_USER_ACCUM_PS_2: u32 = 0x0B0D0;
pub const SPI_SHADER_USER_ACCUM_PS_3: u32 = 0x0B0D4;

pub const SPI_SHADER_PGM_RSRC4_VS: u32 = 0x0B104;
pub const SPI_SHADER_PGM_RSRC3_VS: u32 = 0x0B118;
pub const SPI_SHADER_LATE_ALLOC_VS: u32 = 0x0B11C;
pub const SPI_SHADER_PGM_LO_VS: u32 = 0x0B120;
pub const SPI_SHADER_PGM_HI_VS: u32 = 0x0B124;
pub const SPI_SHADER_PGM_RSRC1_VS: u32 = 0x0B128;
pub const SPI_SHADER_PGM_RSRC2_VS: u32 = 0x0B12C;
pub const SPI_SHADER_USER_DATA_VS_0: u32 = 0x0B130;
pub const SPI_SHADER_REQ_CTRL_VS: u32 = 0x0B1C0;
pub const SPI_SHADER_USER_ACCUM_VS_0: u32 = 0x0B1C8;
pub const SPI_SHADER_USER_ACCUM_VS_1: u32 = 0x0B1CC;
pub const SPI_SHADER_USER_ACCUM_VS_2: u32 = 0x0B1D0;
pub const SPI_SHADER_USER_ACCUM_VS_3: u32 = 0x0B1D4;

// SPI_SHADER_PGM_RSRC1_* / _RSRC2_*.
pub const RSRC1_VGPRS_SHIFT: u32 = 0; // granule 8, value = vgprs/8 - 1
pub const RSRC1_VGPRS_MASK: u32 = 0x3F;
pub const RSRC2_SCRATCH_EN: u32 = 0;
pub const RSRC2_USER_SGPR_SHIFT: u32 = 1;
pub const RSRC2_USER_SGPR_MASK: u32 = 0x1F;
pub const RSRC2_USER_SGPR_MSB: u32 = 27;
/// `FLOAT_MODE.FP_16_64_DENORMS` — LLVM's default for graphics shaders.
pub const RSRC1_FLOAT_MODE_DENORM16_64: u32 = f(192, 12, 0xFF);
/// `DX10_CLAMP` — required for the GL/DX I/O conventions LLVM emits.
pub const RSRC1_DX10_CLAMP: u32 = f(1, 21, 1);
/// GFX10 shader memory ordering (bit differs between VS and PS).
pub const RSRC1_VS_MEM_ORDERED: u32 = 1 << 27;
pub const RSRC1_PS_MEM_ORDERED: u32 = 1 << 25;

pub const RSRC3_CU_EN_SHIFT: u32 = 0;
pub const RSRC3_CU_EN_MASK: u32 = 0xFFFF;
pub const RSRC3_WAVE_LIMIT_PS: u32 = f(0x3F, 16, 0x3F);
pub const RSRC3_WAVE_LIMIT_VS: u32 = f(0x3F, 16, 0x3F);

// Context registers, context base 0x28000.
pub const DB_RENDER_CONTROL: u32 = 0x28000;
pub const DB_COUNT_CONTROL: u32 = 0x28004;
pub const DB_DEPTH_VIEW: u32 = 0x28008;
pub const DB_RENDER_OVERRIDE: u32 = 0x2800C;
pub const DB_RENDER_OVERRIDE2: u32 = 0x28010;
pub const DB_HTILE_DATA_BASE: u32 = 0x28014;
pub const DB_DEPTH_SIZE_XY: u32 = 0x2801C;
pub const DB_STENCIL_CLEAR: u32 = 0x28028;
pub const DB_DEPTH_CLEAR: u32 = 0x2802C;
pub const PA_SC_SCREEN_SCISSOR_TL: u32 = 0x28030;
pub const PA_SC_SCREEN_SCISSOR_BR: u32 = 0x28034;
pub const DB_DFSM_CONTROL: u32 = 0x28038;
pub const DB_DEPTH_INFO: u32 = 0x2803C;
pub const DB_Z_INFO: u32 = 0x28040;
pub const DB_STENCIL_INFO: u32 = 0x28044;
pub const DB_Z_READ_BASE: u32 = 0x28048;
pub const DB_STENCIL_READ_BASE: u32 = 0x2804C;
pub const DB_Z_WRITE_BASE: u32 = 0x28050;
pub const DB_STENCIL_WRITE_BASE: u32 = 0x28054;
pub const DB_Z_READ_BASE_HI: u32 = 0x28068;
pub const TA_BC_BASE_ADDR: u32 = 0x28080;
pub const TA_BC_BASE_ADDR_HI: u32 = 0x28084;
pub const DB_RMI_L2_CACHE_CONTROL: u32 = 0x2807C;
pub const PA_SC_WINDOW_OFFSET: u32 = 0x28200;
pub const PA_SC_WINDOW_SCISSOR_TL: u32 = 0x28204;
pub const PA_SC_WINDOW_SCISSOR_BR: u32 = 0x28208;
pub const PA_SC_CLIPRECT_RULE: u32 = 0x2820C;
pub const PA_SC_EDGERULE: u32 = 0x28230;
pub const PA_SU_HARDWARE_SCREEN_OFFSET: u32 = 0x28234;
pub const CB_TARGET_MASK: u32 = 0x28238;
pub const CB_SHADER_MASK: u32 = 0x2823C;
pub const PA_SC_GENERIC_SCISSOR_TL: u32 = 0x28240;
pub const PA_SC_GENERIC_SCISSOR_BR: u32 = 0x28244;
pub const PA_SC_VPORT_SCISSOR_0_TL: u32 = 0x28250;
pub const PA_SC_VPORT_SCISSOR_0_BR: u32 = 0x28254;
pub const PA_SC_VPORT_ZMIN_0: u32 = 0x282D0;
pub const PA_SC_VPORT_ZMAX_0: u32 = 0x282D4;
pub const DB_DEPTH_CONTROL: u32 = 0x28800;
pub const DB_EQAA: u32 = 0x28804;
pub const CB_COLOR_CONTROL: u32 = 0x28808;
pub const DB_SHADER_CONTROL: u32 = 0x2880C;
pub const PA_CL_CLIP_CNTL: u32 = 0x28810;
pub const PA_SU_SC_MODE_CNTL: u32 = 0x28814;
pub const PA_CL_VTE_CNTL: u32 = 0x28818;
pub const PA_CL_VS_OUT_CNTL: u32 = 0x2881C;
pub const PA_SU_PRIM_FILTER_CNTL: u32 = 0x2882C;
pub const PA_CL_NGG_CNTL: u32 = 0x28838;
pub const PA_CL_VRS_CNTL: u32 = 0x28848;
pub const SPI_PS_INPUT_CNTL_0: u32 = 0x28644;
pub const SPI_PS_INPUT_CNTL_1: u32 = 0x28648;
pub const SPI_VS_OUT_CONFIG: u32 = 0x286C4;
pub const SPI_PS_INPUT_ENA: u32 = 0x286CC;
pub const SPI_PS_INPUT_ADDR: u32 = 0x286D0;
pub const SPI_INTERP_CONTROL_0: u32 = 0x286D4;
pub const SPI_PS_IN_CONTROL: u32 = 0x286D8;
pub const SPI_BARYC_CNTL: u32 = 0x286E0;
pub const SPI_TMPRING_SIZE: u32 = 0x286E8;
pub const SPI_SHADER_IDX_FORMAT: u32 = 0x28708;
pub const SPI_SHADER_POS_FORMAT: u32 = 0x2870C;
pub const SPI_SHADER_Z_FORMAT: u32 = 0x28710;
pub const SPI_SHADER_COL_FORMAT: u32 = 0x28714;
pub const SX_PS_DOWNCONVERT: u32 = 0x28754;
pub const SX_BLEND_OPT_EPSILON: u32 = 0x28758;
pub const SX_BLEND_OPT_CONTROL: u32 = 0x2875C;
pub const SX_MRT0_BLEND_OPT: u32 = 0x28760;
pub const SX_PS_DOWNCONVERT_CONTROL: u32 = 0x28750;
pub const CB_BLEND0_CONTROL: u32 = 0x28780;
pub const CB_BLEND_RED: u32 = 0x28414;
pub const CB_DCC_CONTROL: u32 = 0x28424;
pub const CB_RMI_GL2_CACHE_CONTROL: u32 = 0x28410;
pub const PA_SU_POINT_SIZE: u32 = 0x28A00;
pub const PA_SU_POINT_MINMAX: u32 = 0x28A04;
pub const PA_SU_LINE_CNTL: u32 = 0x28A08;
pub const PA_SC_LINE_STIPPLE: u32 = 0x28A0C;
pub const VGT_GS_MODE: u32 = 0x28A40;
pub const VGT_GS_ONCHIP_CNTL: u32 = 0x28A44;
pub const PA_SC_MODE_CNTL_0: u32 = 0x28A48;
pub const PA_SC_MODE_CNTL_1: u32 = 0x28A4C;
pub const VGT_PRIMITIVEID_EN: u32 = 0x28A84;
pub const VGT_ESGS_RING_ITEMSIZE: u32 = 0x28AAC;
pub const VGT_REUSE_OFF: u32 = 0x28AB4;
pub const VGT_HOS_MAX_TESS_LEVEL: u32 = 0x28A18;
pub const VGT_TESS_DISTRIBUTION: u32 = 0x28B50;
pub const VGT_SHADER_STAGES_EN: u32 = 0x28B54;
/// Multi-dword register: the value lives in dword index 1
/// (`Pm4::set_context_reg_idx`).
pub const IA_MULTI_VGT_PARAM: u32 = 0x28AA8;
pub const VGT_LS_HS_CONFIG: u32 = 0x28B58;
pub const VGT_TF_PARAM: u32 = 0x28B6C;
pub const VGT_STRMOUT_CONFIG: u32 = 0x28B94;
pub const VGT_STRMOUT_BUFFER_CONFIG: u32 = 0x28B98;
pub const PA_SC_AA_CONFIG: u32 = 0x28BE0;
pub const PA_SC_AA_MASK_X0Y0_X1Y0: u32 = 0x28C38;
pub const PA_SC_LINE_CNTL: u32 = 0x28BDC;
pub const PA_SU_VTX_CNTL: u32 = 0x28BE4;
pub const PA_CL_GB_VERT_CLIP_ADJ: u32 = 0x28BE8;
pub const PA_CL_GB_VERT_DISC_ADJ: u32 = 0x28BEC;
pub const PA_CL_GB_HORZ_CLIP_ADJ: u32 = 0x28BF0;
pub const PA_CL_GB_HORZ_DISC_ADJ: u32 = 0x28BF4;
pub const PA_SU_POLY_OFFSET_DB_FMT_CNTL: u32 = 0x28B78;
pub const PA_SU_POLY_OFFSET_CLAMP: u32 = 0x28B7C;
pub const PA_SU_POLY_OFFSET_FRONT_SCALE: u32 = 0x28B80;
pub const PA_SU_POLY_OFFSET_FRONT_OFFSET: u32 = 0x28B84;
pub const PA_SU_POLY_OFFSET_BACK_SCALE: u32 = 0x28B88;
pub const PA_SU_POLY_OFFSET_BACK_OFFSET: u32 = 0x28B8C;
pub const PA_SC_BINNER_CNTL_0: u32 = 0x28C44;
pub const PA_SC_BINNER_CNTL_1: u32 = 0x28C48;
pub const PA_SC_NGG_MODE_CNTL: u32 = 0x28C50;
pub const VGT_VERTEX_REUSE_BLOCK_CNTL: u32 = 0x28C58;
pub const PA_SU_SMALL_PRIM_FILTER_CNTL: u32 = 0x28830;
pub const PA_SC_CONSERVATIVE_RASTERIZATION_CNTL: u32 = 0x28C4C;
pub const CB_COLOR0_BASE: u32 = 0x28C60;
pub const CB_COLOR0_VIEW: u32 = 0x28C6C;
pub const CB_COLOR0_INFO: u32 = 0x28C70;
pub const CB_COLOR0_ATTRIB: u32 = 0x28C74;
pub const CB_COLOR0_DCC_CONTROL: u32 = 0x28C78;
pub const CB_COLOR0_CMASK: u32 = 0x28C7C;
pub const CB_COLOR0_FMASK: u32 = 0x28C84;
pub const CB_COLOR0_CLEAR_WORD0: u32 = 0x28C8C;
pub const CB_COLOR0_CLEAR_WORD1: u32 = 0x28C90;
pub const CB_COLOR0_DCC_BASE: u32 = 0x28C94;
pub const CB_COLOR0_BASE_EXT: u32 = 0x28E40;
pub const CB_COLOR0_CMASK_BASE_EXT: u32 = 0x28E60;
pub const CB_COLOR0_FMASK_BASE_EXT: u32 = 0x28E80;
pub const CB_COLOR0_DCC_BASE_EXT: u32 = 0x28EA0;
pub const CB_COLOR0_ATTRIB2: u32 = 0x28EC0;
pub const CB_COLOR0_ATTRIB3: u32 = 0x28EE0;
pub const CB_COLOR1_INFO: u32 = 0x28CAC;
/// Stride between `CB_COLOR<i>` blocks.
pub const CB_COLOR_STRIDE: u32 = 0x3C;

// User-config (UCONFIG) registers, base 0x30000.
pub const VGT_PRIMITIVE_TYPE: u32 = 0x30908;
pub const VGT_INDEX_TYPE: u32 = 0x3090C;
pub const GE_MIN_VTX_INDX: u32 = 0x30924;
pub const GE_INDX_OFFSET: u32 = 0x30928;
pub const GE_MULTI_PRIM_IB_RESET_EN: u32 = 0x3092C;
pub const GE_MAX_VTX_INDX: u32 = 0x30964;
pub const VGT_INSTANCE_BASE_ID: u32 = 0x30968;
pub const GE_CNTL: u32 = 0x3096C;
pub const GE_STEREO_CNTL: u32 = 0x3097C;
pub const GE_PC_ALLOC: u32 = 0x30980;
pub const GE_USER_VGPR_EN: u32 = 0x30988;
pub const PA_SU_LINE_STIPPLE_VALUE: u32 = 0x30A00;
pub const PA_SC_LINE_STIPPLE_STATE: u32 = 0x30A04;

// ============================================================================
// Field values
// ============================================================================

// PA_SC_WINDOW_SCISSOR_{TL,BR} / PA_SC_VPORT_SCISSOR_0_*.
pub const SCISSOR_TL_X_SHIFT: u32 = 0;
pub const SCISSOR_TL_Y_SHIFT: u32 = 16;
pub const WINDOW_OFFSET_DISABLE: u32 = 1 << 31;

// PA_SC_SCREEN_SCISSOR_{TL,BR}: 16-bit inclusive bounds.
pub const SCREEN_SCISSOR_X_MASK: u32 = 0xFFFF;
pub const SCREEN_SCISSOR_Y_SHIFT: u32 = 16;

// DB_RENDER_CONTROL.
pub const DB_CLEAR_DEPTH: u32 = 1 << 0;
pub const DB_CLEAR_STENCIL: u32 = 1 << 1;

// DB_DEPTH_CONTROL.
pub const DB_STENCIL_ENABLE: u32 = 1 << 0;
pub const DB_Z_ENABLE: u32 = 1 << 1;
pub const DB_Z_WRITE_ENABLE: u32 = 1 << 2;
pub const DB_ZFUNC_SHIFT: u32 = 4;
pub const DB_ZFUNC_MASK: u32 = 7;

// DB_SHADER_CONTROL.
pub const DB_Z_ORDER_SHIFT: u32 = 4;
pub const DB_LATE_Z: u32 = 0;
pub const DB_EARLY_Z_THEN_LATE_Z: u32 = 1;
pub const DB_KILL_ENABLE: u32 = 1 << 6;
pub const DB_DUAL_QUAD_DISABLE: u32 = 1 << 15;

// PA_CL_CLIP_CNTL.
pub const CLIP_DISABLE: u32 = 1 << 16;
pub const DX_CLIP_SPACE_DEF: u32 = 0; // 0 = OpenGL [-w, w]
pub const DX_LINEAR_ATTR_CLIP_ENA: u32 = 1 << 24;

// PA_SU_SC_MODE_CNTL.
pub const SC_CULL_FRONT: u32 = 1 << 0;
pub const SC_CULL_BACK: u32 = 1 << 1;
pub const SC_FACE: u32 = 1 << 2;
pub const SC_POLY_MODE: u32 = 1 << 3;
pub const SC_POLY_OFFSET_FRONT_ENABLE: u32 = 1 << 11;
pub const SC_POLY_OFFSET_BACK_ENABLE: u32 = 1 << 12;
pub const SC_POLY_OFFSET_PARA_ENABLE: u32 = 1 << 13;
pub const SC_PROVOKING_VTX_LAST: u32 = 1 << 19;
pub const SC_MULTI_PRIM_IB_ENA: u32 = 1 << 21;
pub const SC_KEEP_TOGETHER_ENABLE: u32 = 1 << 24;

// PA_CL_VTE_CNTL.
pub const VTE_VPORT_XY_SCALE_ENA: u32 = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3);
pub const VTE_VPORT_Z: u32 = (1 << 4) | (1 << 5);
pub const VTE_VTX_W0_FMT: u32 = 1 << 10;

// PA_SU_VTX_CNTL.
pub const VTX_PIX_CENTER: u32 = 1 << 0;
pub const VTX_ROUND_TO_EVEN: u32 = f(2, 1, 0x3);
pub const VTX_QUANT_16_8_1_256: u32 = f(5, 3, 0x7);

// PA_SU_SC_MODE_CNTL fill modes.
pub const POLY_FRONT_PTYPE_SHIFT: u32 = 5;
pub const POLY_BACK_PTYPE_SHIFT: u32 = 8;
pub const DRAW_TRIANGLES: u32 = 2;

// VGT_SHADER_STAGES_EN.
pub const VS_STAGE_REAL: u32 = f(0, 6, 0x3);
pub const VS_W32_EN: u32 = 1 << 23;
pub const GS_FAST_LAUNCH: u32 = f(1, 19, 0x3);
pub const MAX_PRIMGRP_IN_WAVE_SHIFT: u32 = 15;
pub const MAX_PRIMGRP_IN_WAVE_MASK: u32 = 0xF;

// IA_MULTI_VGT_PARAM (dword 1 of R_028AA8).
pub const IA_PRIMGROUP_SIZE: u32 = 128;
pub const IA_WD_SWITCH_ON_EOP: u32 = 1 << 20;

// GE_CNTL.
pub const GE_PRIM_GRP_SIZE_SHIFT: u32 = 0;
pub const GE_PRIM_GRP_SIZE_MASK: u32 = 0x1FF;
pub const GE_VERT_GRP_SIZE_SHIFT: u32 = 9;
pub const GE_VERT_GRP_SIZE_MASK: u32 = 0x1FF;

// VGT_PRIMITIVE_TYPE.
pub const DI_PT_TRILIST: u32 = 4;

// VGT_INDEX_TYPE.
pub const VGT_INDEX_16: u32 = 0;
pub const VGT_INDEX_32: u32 = 1;

// VGT_GS_MODE.
pub const GS_OFF: u32 = 0;

// SPI_VS_OUT_CONFIG.
pub const VS_EXPORT_COUNT_SHIFT: u32 = 1;
pub const VS_EXPORT_COUNT_MASK: u32 = 0x1F;
pub const VS_NO_PC_EXPORT: u32 = 1 << 7;

// SPI_SHADER_POS_FORMAT / SPI_SHADER_{Z,COL}_FORMAT / CB_SHADER_MASK.
pub const SPI_SHADER_4COMP: u32 = 4;
pub const SPI_SHADER_32_ABGR: u32 = 9;
pub const SPI_SHADER_NONE: u32 = 0;
/// Fully-enabled `CB_SHADER_MASK.OUTPUT0_ENABLE`.
pub const CB_SHADER_MASK_RGBA: u32 = 0xF;

// SPI_PS_INPUT_ENA / SPI_PS_INPUT_ADDR.
pub const PERSP_CENTER_ENA: u32 = 1 << 1;

// SPI_PS_IN_CONTROL.
pub const PS_IN_NUM_INTERP_SHIFT: u32 = 0;
pub const PS_IN_NUM_INTERP_MASK: u32 = 0x3F;
pub const PS_IN_PS_W32_EN: u32 = 1 << 15;

// SPI_PS_INPUT_CNTL_n.
pub const PS_INPUT_OFFSET_SHIFT: u32 = 0;
pub const PS_INPUT_OFFSET_MASK: u32 = 0x3F;
pub const PS_INPUT_FLAT_SHADE: u32 = 1 << 10;

// CB_COLOR0_INFO.
pub const CB_ENDIAN_NONE: u32 = f(0, 0, 0x3);
pub const CB_FORMAT_SHIFT: u32 = 2;
pub const CB_FORMAT_MASK: u32 = 0x1F;
pub const CB_FORMAT_8_8_8_8: u32 = 10;
pub const CB_NUMBER_UNORM: u32 = f(0, 8, 0x7);
pub const CB_SWAP_STD: u32 = f(0, 11, 0x3);
pub const CB_SWAP_ALT: u32 = f(1, 11, 0x3);
pub const CB_COMP_SWAP_SHIFT: u32 = 11;
pub const CB_BLEND_CLAMP: u32 = 1 << 15;
pub const CB_ROUND_MODE: u32 = 1 << 18;
pub const CB_SIMPLE_FLOAT: u32 = 1 << 17;
pub const CB_LINEAR_GENERAL: u32 = 1 << 7;

// CB_COLOR0_ATTRIB2.
pub const CB_MIP0_HEIGHT_SHIFT: u32 = 0;
pub const CB_MIP0_HEIGHT_MASK: u32 = 0x3FFF;
pub const CB_MIP0_WIDTH_SHIFT: u32 = 14;
pub const CB_MIP0_WIDTH_MASK: u32 = 0x3FFF;

// CB_COLOR0_ATTRIB3.
pub const CB_MIP0_DEPTH_SHIFT: u32 = 0;
pub const CB_MIP0_DEPTH_MASK: u32 = 0x1FFF;
pub const CB_COLOR_SW_MODE_SHIFT: u32 = 14;
pub const CB_COLOR_SW_MODE_MASK: u32 = 0x1F;
pub const CB_RESOURCE_TYPE_SHIFT: u32 = 24;
pub const CB_RESOURCE_TYPE_MASK: u32 = 0x3;
pub const CB_RESOURCE_TYPE_2D: u32 = 1;
pub const CB_RESOURCE_LEVEL: u32 = 1 << 27;
/// `SW_LINEAR`.
pub const SW_LINEAR: u32 = 0;

// CB_COLOR0_VIEW.
pub const CB_VIEW_SLICE_START_SHIFT: u32 = 0;
pub const CB_VIEW_SLICE_MAX_SHIFT: u32 = 13;
pub const CB_VIEW_MIP_LEVEL_SHIFT: u32 = 26;
/// A linear 2D surface needs at most one slice/mip.
pub const CB_VIEW_SLICE_MAX: u32 = 0x1FFF;

// CB_COLOR_CONTROL.
pub const CB_MODE_NORMAL: u32 = f(1, 4, 0x7);
pub const CB_ROP3_COPY: u32 = f(204, 16, 0xFF);

// CB_BLEND0_CONTROL.
pub const BLEND_COLOR_SRCBLEND_SHIFT: u32 = 0;
pub const BLEND_COLOR_COMB_FCN_SHIFT: u32 = 5;
pub const BLEND_COLOR_DESTBLEND_SHIFT: u32 = 8;
pub const BLEND_ALPHA_SRCBLEND_SHIFT: u32 = 16;
pub const BLEND_ALPHA_COMB_FCN_SHIFT: u32 = 21;
pub const BLEND_ALPHA_DESTBLEND_SHIFT: u32 = 24;
pub const BLEND_SEPARATE_ALPHA: u32 = 1 << 29;
pub const BLEND_ENABLE: u32 = 1 << 30;

/// CB blend factors as `(GL enum -> hardware value)`.
pub fn blend_factor(gl_factor: u32) -> u32 {
    use volt_raster::gl;
    match gl_factor {
        gl::ZERO => 0,
        gl::ONE => 1,
        gl::SRC_COLOR => 2,
        gl::ONE_MINUS_SRC_COLOR => 3,
        gl::SRC_ALPHA => 4,
        gl::ONE_MINUS_SRC_ALPHA => 5,
        gl::DST_ALPHA => 6,
        gl::ONE_MINUS_DST_ALPHA => 7,
        gl::DST_COLOR => 8,
        gl::ONE_MINUS_DST_COLOR => 9,
        gl::SRC_ALPHA_SATURATE => 10,
        gl::CONSTANT_COLOR | gl::CONSTANT_ALPHA => 13,
        gl::ONE_MINUS_CONSTANT_COLOR | gl::ONE_MINUS_CONSTANT_ALPHA => 14,
        _ => 1,
    }
}

/// Depth comparison (`GL enum -> DB_DEPTH_CONTROL.ZFUNC`).
pub fn depth_func(gl_func: u32) -> u32 {
    use volt_raster::gl;
    match gl_func {
        gl::NEVER => 0,
        gl::LESS => 1,
        gl::EQUAL => 2,
        gl::LEQUAL => 3,
        gl::GREATER => 4,
        gl::NOTEQUAL => 5,
        gl::GEQUAL => 6,
        _ => 7,
    }
}

// PA_SC_MODE_CNTL_0.
pub const SC_MODE_MSAA_ENABLE: u32 = 1 << 0;
pub const SC_MODE_VPORT_SCISSOR_ENABLE: u32 = 1 << 1;
pub const SC_MODE_ALTERNATE_RBS_PER_TILE: u32 = 1 << 5;

// PA_SC_MODE_CNTL_1 (reduces to "sane defaults" for linear targets).
pub const SC_MODE1_WALK_SIZE: u32 = 1 << 0;
pub const SC_MODE1_WALK_FENCE_SIZE: u32 = f(3, 4, 0x7);
pub const SC_MODE1_SUPERTILE_WALK_ORDER: u32 = 1 << 7;
pub const SC_MODE1_TILE_WALK_ORDER: u32 = 1 << 8;
pub const SC_MODE1_MULTI_SHADER_ENGINE_PRIM_DISCARD: u32 = 1 << 17;
pub const SC_MODE1_FORCE_EOV_CNTDWN: u32 = 1 << 25;
pub const SC_MODE1_FORCE_EOV_REZ: u32 = 1 << 26;

// PA_SU_SC_MODE_CNTL blend-related / PA_SU_POINT_SIZE.
pub const POINT_SIZE_1PX: u32 = f(8, 0, 0xFFFF) | f(8, 16, 0xFFFF);
pub const LINE_WIDTH_1PX: u32 = f(8, 0, 0xFFFF);
pub const EDGERULE_GL: u32 = f(0xA, 0, 0xF)   // ER_TRI
    | f(0x6, 4, 0xF)                          // ER_POINT
    | f(0xA, 8, 0xF)                          // ER_RECT
    | f(0x19, 12, 0x3F)                       // ER_LINE_LR
    | f(0x25, 18, 0x3F)                       // ER_LINE_RL
    | f(0xA, 24, 0xF)                         // ER_LINE_TB
    | f(0xA, 28, 0xF); // ER_LINE_BT
pub const NGG_CNTL_LEGACY: u32 = f(30, 2, 0xFF);
