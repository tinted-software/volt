//! GFX10.3 hardware state emission for the fixed-function pipeline.
//!
//! Emits the preamble (once per IB) and the per-draw state (programs,
//! user SGPRs, rasterizer, blend, color target, and draw packet).
#![allow(dead_code)]

use super::pm4::*;
use super::regs::*;
use volt_raster::gl;
use volt_shader::amdgcn::{PS_USER_SGPRS, PsConstants, VS_PARAM_EXPORTS, VS_USER_SGPRS};

/// GCR_CNTL bitmask: invalidate and write back all GL2/GL1/vector/scalar/
/// instruction caches across the chip.
pub const GCR_CNTL_INVALIDATE_ALL: u32 = (1 << 0)   // GL2_INV
    | (1 << 1)                                       // GL2_WB
    | (1 << 2)                                       // GLM_INV
    | (1 << 3)                                       // GLM_WB
    | (1 << 4)                                       // GL1_INV
    | (1 << 5)                                       // GLV_INV
    | (1 << 6)                                       // GLK_INV
    | (1 << 7); // GLI_INV

/// Emit the per-IB preamble: hardware defaults, DPBB disable, cache sync.
pub fn emit_preamble(pm4: &mut Pm4) {
    pm4.context_control();
    pm4.clear_state();

    // CLEAR_STATE leaves SH registers untouched. Match Mesa's GFX10 preamble.
    pm4.set_sh_reg(SPI_SHADER_REQ_CTRL_PS, 1 | (3 << 1)); // grouping; 4 requests/CU
    pm4.set_sh_reg_seq(SPI_SHADER_USER_ACCUM_PS_0, &[0; 4]);
    pm4.set_sh_reg(SPI_SHADER_REQ_CTRL_VS, 0);
    pm4.set_sh_reg_seq(SPI_SHADER_USER_ACCUM_VS_0, &[0; 4]);

    // Disable DPBB binning: straight-through rasterization matching the CPU.
    // BIN_SIZE 128x128, DISABLE_BINNING_USE_NEW_SC (10.3).
    let binner_cntl_0 = f(2, 0, 0x3) // BINNING_MODE = DISABLE_BINNING_USE_NEW_SC
        | f(2, 4, 0x7)               // BIN_SIZE_X_EXTEND = 2 (128 px)
        | f(2, 7, 0x7)               // BIN_SIZE_Y_EXTEND = 2 (128 px)
        | (1 << 18)                  // DISABLE_START_OF_PRIM
        | f(63, 19, 0xFF)            // FPOVS_PER_BATCH
        | (1 << 27)                  // OPTIMAL_BIN_SELECTION
        | (1 << 28); // FLUSH_ON_BINNING_TRANSITION
    pm4.set_context_reg(PA_SC_BINNER_CNTL_0, binner_cntl_0);
    // MAX_ALLOC_COUNT = 84 (pc_lines 256 / 3 - 1), MAX_PRIM_PER_BATCH = 1023.
    pm4.set_context_reg(PA_SC_BINNER_CNTL_1, f(84, 0, 0xFFFF) | f(1023, 16, 0xFFFF));
    pm4.set_context_reg(PA_SC_NGG_MODE_CNTL, f(512, 0, 0x7FF));

    // Preamble registers from radeonsi (ac_cmdbuf.c / si_state.c).
    pm4.set_reg(SPI_SHADER_IDX_FORMAT, f(1, 0, 0xF)); // 1COMP
    pm4.set_reg(
        PA_CL_VRS_CNTL,
        f(1, 0, 0x7) | f(1, 9, 0x7), // OVERRIDE
    );
    pm4.set_reg(
        PA_SU_PRIM_FILTER_CNTL,
        (1 << 30) | (1 << 31), // XMAX_RIGHT / YMAX_BOTTOM_EXCLUSION
    );
    pm4.set_reg(DB_DFSM_CONTROL, f(2, 0, 0x3)); // PUNCHOUT_MODE = FORCE_OFF
    // L2 cache write-through/stream policies for color and Z.
    pm4.set_reg(
        CB_RMI_GL2_CACHE_CONTROL,
        f(1, 6, 0x3) | f(1, 22, 0x3), // COLOR_WR_POLICY / RD_POLICY = STREAM
    );
    pm4.set_reg(
        DB_RMI_L2_CACHE_CONTROL,
        f(1, 0, 0x3) | f(1, 16, 0x3), // Z STREAM
    );
    pm4.set_reg(PA_SU_SMALL_PRIM_FILTER_CNTL, 1);
    pm4.set_reg(SX_PS_DOWNCONVERT_CONTROL, 0xFF);
    pm4.set_reg(VGT_VERTEX_REUSE_BLOCK_CNTL, 14);

    // Initial UCONFIG defaults.
    pm4.set_uconfig_reg(GE_MIN_VTX_INDX, 0);
    pm4.set_uconfig_reg(GE_INDX_OFFSET, 0);
    pm4.set_uconfig_reg(GE_MAX_VTX_INDX, 0xFFFFFFFF);
    pm4.set_uconfig_reg(VGT_INSTANCE_BASE_ID, 0);
    pm4.set_uconfig_reg(GE_STEREO_CNTL, 0);
    pm4.set_uconfig_reg(GE_USER_VGPR_EN, 0);
    pm4.set_uconfig_reg(PA_SU_LINE_STIPPLE_VALUE, 0);
    pm4.set_uconfig_reg(PA_SC_LINE_STIPPLE_STATE, 0);

    // Full screen scissor bounds default to the hardware max (16384).
    pm4.set_reg(
        PA_SC_SCREEN_SCISSOR_BR,
        f(16384, 0, SCREEN_SCISSOR_X_MASK) | (16384 << SCREEN_SCISSOR_Y_SHIFT),
    );
    pm4.set_reg(PA_SC_WINDOW_SCISSOR_TL, WINDOW_OFFSET_DISABLE);
    pm4.set_reg(PA_SC_GENERIC_SCISSOR_TL, WINDOW_OFFSET_DISABLE);
    pm4.set_reg(
        PA_SC_GENERIC_SCISSOR_BR,
        f(16384, 0, 0x7FFF) | (16384 << 16),
    );
    pm4.set_reg(PA_SC_CLIPRECT_RULE, 0xFFFF);
    pm4.set_reg(PA_SC_WINDOW_OFFSET, 0);

    // Initial VGT / GE state for legacy VS/PS.
    let vgt_stages = VS_W32_EN | f(2, MAX_PRIMGRP_IN_WAVE_SHIFT, MAX_PRIMGRP_IN_WAVE_MASK);
    pm4.set_reg(VGT_SHADER_STAGES_EN, vgt_stages);
    pm4.set_reg(VGT_REUSE_OFF, 0);
    // GE_CNTL: 128 prims per group (recommended for non-tess non-GS).
    let ge_cntl = f(128, GE_PRIM_GRP_SIZE_SHIFT, GE_PRIM_GRP_SIZE_MASK);
    pm4.set_uconfig_reg(GE_CNTL, ge_cntl);
    pm4.set_uconfig_reg(VGT_PRIMITIVE_TYPE, DI_PT_TRILIST);
    // Disable streamout explicitly; CLEAR_STATE may leave streamout routing undefined.
    pm4.set_context_reg_seq(VGT_STRMOUT_CONFIG, &[0, 0]);

    // Line/point sizes, screen offset.
    pm4.set_reg(PA_SU_POINT_SIZE, POINT_SIZE_1PX);
    pm4.set_reg(PA_SU_POINT_MINMAX, POINT_SIZE_1PX);
    pm4.set_reg(PA_SU_LINE_CNTL, LINE_WIDTH_1PX);
    pm4.set_reg(PA_SU_HARDWARE_SCREEN_OFFSET, 0);
}

/// Helper: dispatch to `set_context_reg` with an address bounds check.
trait Pm4RegExt {
    fn set_reg(&mut self, addr: u32, val: u32);
}

impl Pm4RegExt for Pm4 {
    #[inline(always)]
    fn set_reg(&mut self, addr: u32, val: u32) {
        self.set_context_reg(addr, val);
    }
}

/// Colorbuffer target (only MRT0 is used; MRT1..7 are set to INVALID).
pub fn emit_color_target(
    pm4: &mut Pm4,
    color_va: u64,
    pitch_bytes: u32,
    width: u32,
    height: u32,
    is_bgra: bool,
) {
    let bpp = 4u32;
    let pitch_pixels = pitch_bytes / bpp;
    debug_assert!(
        pitch_bytes.is_multiple_of(256),
        "pitch must be a 256B multiple"
    );

    // Word 0..13 for CB_COLOR0_BASE..CB_COLOR0_DCC_BASE.
    let base_lo = (color_va >> 8) as u32;
    let base_hi = (color_va >> 40) as u32;
    let view = 0; // slice 0, mip 0
    let swap = if is_bgra { CB_SWAP_ALT } else { CB_SWAP_STD };
    let info = CB_ENDIAN_NONE
        | f(CB_FORMAT_8_8_8_8, CB_FORMAT_SHIFT, CB_FORMAT_MASK)
        | CB_NUMBER_UNORM
        | swap
        | CB_BLEND_CLAMP
        | CB_SIMPLE_FLOAT;
    let attrib = 0; // 1 sample, 1 fragment
    let dcc_control = 0;
    let attrib2 = f(
        height.saturating_sub(1),
        CB_MIP0_HEIGHT_SHIFT,
        CB_MIP0_HEIGHT_MASK,
    ) | f(
        pitch_pixels.saturating_sub(1),
        CB_MIP0_WIDTH_SHIFT,
        CB_MIP0_WIDTH_MASK,
    );
    let attrib3 = f(1, CB_MIP0_DEPTH_SHIFT, CB_MIP0_DEPTH_MASK) // one layer
        | f(SW_LINEAR, CB_COLOR_SW_MODE_SHIFT, CB_COLOR_SW_MODE_MASK)
        | f(CB_RESOURCE_TYPE_2D, CB_RESOURCE_TYPE_SHIFT, CB_RESOURCE_TYPE_MASK)
        | (1 << 26) // CMASK_PIPE_ALIGNED (Mesa GFX10 mutable surface)
        | CB_RESOURCE_LEVEL;

    // R_028C60_CB_COLOR0_BASE: 14 sequential context registers.
    pm4.set_context_reg_seq(
        CB_COLOR0_BASE,
        &[
            base_lo,
            0, // hole (CB_COLOR0_PITCH unused on GFX10)
            0, // hole
            view,
            info,
            attrib,
            dcc_control,
            0, // CB_COLOR0_CMASK
            0, // hole
            0, // CB_COLOR0_FMASK
            0, // hole
            0, // CLEAR_WORD0
            0, // CLEAR_WORD1
            0, // CB_COLOR0_DCC_BASE
        ],
    );
    pm4.set_context_reg(CB_COLOR0_BASE_EXT, base_hi);
    pm4.set_context_reg(CB_COLOR0_ATTRIB2, attrib2);
    pm4.set_context_reg(CB_COLOR0_ATTRIB3, attrib3);

    // Disable unwritten colorbuffers MRT1..7.
    for i in 1..8 {
        pm4.set_context_reg(CB_COLOR0_INFO + i * CB_COLOR_STRIDE, 0);
    }

    // Framebuffer bounds for window scissor.
    pm4.set_context_reg(
        PA_SC_WINDOW_SCISSOR_BR,
        f(width, 0, 0x7FFF) | (height << 16),
    );
}

/// Blend, color mask, and rasterizer state for one draw.
pub fn emit_raster_and_blend(
    pm4: &mut Pm4,
    pipe: &crate::Pipeline,
    viewport: (i32, i32, u32, u32),
    scissor: Option<(i32, i32, u32, u32)>,
    fb_w: u32,
    fb_h: u32,
    uses_kill: bool,
) {
    // 1. Viewport: X/Y scale & offset, Z scale & offset.
    // Negative Y scale flips the OpenGL NDC bottom-left to top-down row 0.
    let (vx, vy, vw, vh) = viewport;
    let hw = (vw as f32) * 0.5;
    let hh = (vh as f32) * 0.5;
    let x_scale = hw.to_bits();
    let x_offset = ((vx as f32) + hw).to_bits();
    let y_scale = (-hh).to_bits();
    let y_offset = ((vy as f32) + hh).to_bits();
    let z_scale = (0.5f32).to_bits();
    let z_offset = (0.5f32).to_bits();

    // PA_CL_VPORT_XSCALE: 6 contiguous context registers starting at 0x2843C.
    pm4.set_context_reg_seq(
        0x2843C,
        &[x_scale, x_offset, y_scale, y_offset, z_scale, z_offset],
    );
    pm4.set_context_reg(PA_SC_VPORT_ZMIN_0, 0);
    pm4.set_context_reg(PA_SC_VPORT_ZMAX_0, (1.0f32).to_bits());

    // 2. Scissor: intersect requested scissor with the framebuffer bounds.
    let (sx, sy, sw, sh) = scissor.unwrap_or((0, 0, fb_w, fb_h));
    let min_x = sx.max(0) as u32;
    let min_y = sy.max(0) as u32;
    let max_x = ((sx + sw as i32).max(0) as u32).min(fb_w);
    let max_y = ((sy + sh as i32).max(0) as u32).min(fb_h);
    let vport_scissor_tl =
        (min_x & 0x7FFF) | ((min_y & 0x7FFF) << SCISSOR_TL_Y_SHIFT) | WINDOW_OFFSET_DISABLE;
    let vport_scissor_br = (max_x & 0x7FFF) | ((max_y & 0x7FFF) << 16);
    pm4.set_context_reg(PA_SC_VPORT_SCISSOR_0_TL, vport_scissor_tl);
    pm4.set_context_reg(PA_SC_VPORT_SCISSOR_0_BR, vport_scissor_br);

    // Guardband 1.0 = clip exactly at the viewport boundary.
    pm4.set_context_reg(
        PA_SU_VTX_CNTL,
        VTX_PIX_CENTER | VTX_ROUND_TO_EVEN | VTX_QUANT_16_8_1_256,
    );
    let one_f = (1.0f32).to_bits();
    pm4.set_context_reg_seq(PA_CL_GB_VERT_CLIP_ADJ, &[one_f, one_f, one_f, one_f]);

    // 3. Culling and face winding.
    // With negative Y-scale, OpenGL CCW front faces evaluate to clockwise
    // in top-down screen space, matching the CPU rasterizer.
    let cull_front = pipe.cull_mode == gl::FRONT || pipe.cull_mode == gl::FRONT_AND_BACK;
    let cull_back = pipe.cull_mode == gl::BACK || pipe.cull_mode == gl::FRONT_AND_BACK;
    let mut su_mode = 0u32;
    if cull_front {
        su_mode |= SC_CULL_FRONT;
    }
    if cull_back {
        su_mode |= SC_CULL_BACK;
    }
    // With negative Y-scale in the viewport, OpenGL CCW front-facing triangles
    // evaluate to clockwise in top-down window space, so CW is front.
    if pipe.front_face_ccw {
        su_mode |= SC_FACE;
    }
    su_mode |= f(DRAW_TRIANGLES, POLY_FRONT_PTYPE_SHIFT, 0x7)
        | f(DRAW_TRIANGLES, POLY_BACK_PTYPE_SHIFT, 0x7)
        | SC_PROVOKING_VTX_LAST;
    pm4.set_context_reg(PA_SU_SC_MODE_CNTL, su_mode);
    pm4.set_context_reg(PA_CL_NGG_CNTL, NGG_CNTL_LEGACY);
    pm4.set_context_reg(PA_SC_EDGERULE, EDGERULE_GL);
    // Multi-primitive IA/VGT grouping (per draw in radeonsi). 128 primitives
    // per group is the recommendation without GS/tessellation; WD_SWITCH_ON_EOP
    // is required on GFX7+ whenever the work distributor would otherwise wait
    // for a signal it never gets (chips with < 4 shader engines).
    pm4.set_context_reg_idx(
        IA_MULTI_VGT_PARAM,
        1,
        f(IA_PRIMGROUP_SIZE - 1, 0, 0xFFFF) | IA_WD_SWITCH_ON_EOP,
    );

    // 4. Blending & Color target mask.
    let mask = (pipe.color_mask as u32) & 0xF;
    pm4.set_context_reg(CB_TARGET_MASK, mask);
    pm4.set_context_reg(CB_COLOR_CONTROL, CB_MODE_NORMAL | CB_ROP3_COPY);
    let blend_cntl = if pipe.blend_enabled {
        let src_rgb = blend_factor(pipe.src_factor);
        let dst_rgb = blend_factor(pipe.dst_factor);
        BLEND_ENABLE
            | f(src_rgb, BLEND_COLOR_SRCBLEND_SHIFT, 0x1F)
            | f(dst_rgb, BLEND_COLOR_DESTBLEND_SHIFT, 0x1F)
            | f(src_rgb, BLEND_ALPHA_SRCBLEND_SHIFT, 0x1F)
            | f(dst_rgb, BLEND_ALPHA_DESTBLEND_SHIFT, 0x1F)
    } else {
        0
    };
    pm4.set_context_reg(CB_BLEND0_CONTROL, blend_cntl);
    pm4.set_context_reg(SX_MRT0_BLEND_OPT, 0); // disable SX blend optimizations
    pm4.set_context_reg(SX_PS_DOWNCONVERT, 0);

    // 5. Scan-conversion / rasterizer modes (single-sample, linear dst).
    pm4.set_context_reg(
        PA_SC_MODE_CNTL_0,
        SC_MODE_VPORT_SCISSOR_ENABLE | SC_MODE_ALTERNATE_RBS_PER_TILE,
    );
    pm4.set_context_reg(
        PA_SC_MODE_CNTL_1,
        SC_MODE1_WALK_SIZE
            | SC_MODE1_WALK_FENCE_SIZE
            | SC_MODE1_SUPERTILE_WALK_ORDER
            | SC_MODE1_TILE_WALK_ORDER
            | SC_MODE1_MULTI_SHADER_ENGINE_PRIM_DISCARD
            | SC_MODE1_FORCE_EOV_CNTDWN
            | SC_MODE1_FORCE_EOV_REZ,
    );
    pm4.set_context_reg(PA_SC_AA_CONFIG, 0); // 1 sample
    // Single-sample target: every sample must be enabled or the coverage mask
    // rejects all fragments. CLEAR_STATE leaves both mask registers zeroed, so
    // nothing would ever rasterize.
    pm4.set_context_reg_seq(PA_SC_AA_MASK_X0Y0_X1Y0, &[0xFFFF_FFFF, 0xFFFF_FFFF]);
    pm4.set_context_reg(DB_EQAA, 0);

    // 6. DB controls: depth testing is deferred to CPU fallback for this pass;
    // ensure the DB does not attempt to read or write unbound depth surfaces.
    // Mesa writes both invalid formats whenever no depth/stencil surface is bound.
    pm4.set_context_reg_seq(DB_Z_INFO, &[0, 0]); // Z_INVALID, STENCIL_INVALID
    pm4.set_context_reg(DB_DEPTH_CONTROL, 0);
    pm4.set_context_reg(DB_RENDER_CONTROL, 0);
    pm4.set_context_reg(DB_COUNT_CONTROL, 0);
    pm4.set_context_reg(DB_RENDER_OVERRIDE2, 0);
    let db_shader = f(DB_EARLY_Z_THEN_LATE_Z, DB_Z_ORDER_SHIFT, 0x3)
        | if uses_kill { DB_KILL_ENABLE } else { 0 }
        | DB_DUAL_QUAD_DISABLE;
    pm4.set_context_reg(DB_SHADER_CONTROL, db_shader);
}

/// Emit shader program pointers and fixed-function interface registers.
pub fn emit_shader_programs(
    pm4: &mut Pm4,
    vs_va: u64,
    vs_vgprs: u32,
    ps_va: u64,
    ps_vgprs: u32,
    ps_input_ena: u32,
    num_interp: u32,
) {
    // --- Vertex Shader ---
    pm4.set_sh_reg(SPI_SHADER_PGM_LO_VS, (vs_va >> 8) as u32);
    pm4.set_sh_reg(SPI_SHADER_PGM_HI_VS, (vs_va >> 40) as u32);
    let vs_vgpr_blocks = (vs_vgprs / 8).saturating_sub(1).min(63);
    let vs_rsrc1 = f(vs_vgpr_blocks, RSRC1_VGPRS_SHIFT, RSRC1_VGPRS_MASK)
        | RSRC1_FLOAT_MODE_DENORM16_64
        | RSRC1_DX10_CLAMP
        | RSRC1_VS_MEM_ORDERED;
    let vs_rsrc2 = f(VS_USER_SGPRS, RSRC2_USER_SGPR_SHIFT, RSRC2_USER_SGPR_MASK);
    pm4.set_sh_reg(SPI_SHADER_PGM_RSRC1_VS, vs_rsrc1);
    pm4.set_sh_reg(SPI_SHADER_PGM_RSRC2_VS, vs_rsrc2);
    pm4.set_sh_reg(
        SPI_SHADER_PGM_RSRC3_VS,
        f(0xFFFF, RSRC3_CU_EN_SHIFT, RSRC3_CU_EN_MASK) | RSRC3_WAVE_LIMIT_VS,
    );
    pm4.set_sh_reg(SPI_SHADER_PGM_RSRC4_VS, f(0xFFFF, 0, 0xFFFF));
    pm4.set_sh_reg(SPI_SHADER_LATE_ALLOC_VS, 0);

    // VS output config: VS exports 3 parameters, POS0 export format = 4COMP.
    let vs_out_config = f(
        VS_PARAM_EXPORTS - 1,
        VS_EXPORT_COUNT_SHIFT,
        VS_EXPORT_COUNT_MASK,
    );
    pm4.set_context_reg(SPI_VS_OUT_CONFIG, vs_out_config);
    pm4.set_context_reg(
        SPI_SHADER_POS_FORMAT,
        f(SPI_SHADER_4COMP, 0, 0xF), // POS0 = 4COMP
    );
    pm4.set_context_reg(
        PA_CL_VTE_CNTL,
        VTE_VPORT_XY_SCALE_ENA | VTE_VPORT_Z | VTE_VTX_W0_FMT,
    );
    pm4.set_context_reg(PA_CL_CLIP_CNTL, DX_CLIP_SPACE_DEF | DX_LINEAR_ATTR_CLIP_ENA);
    pm4.set_context_reg(PA_CL_VS_OUT_CNTL, 0);
    pm4.set_context_reg(VGT_GS_MODE, GS_OFF);
    pm4.set_context_reg(VGT_PRIMITIVEID_EN, 0);

    // GE_PC_ALLOC: 256 lines for parameter cache on Raphael / Navi23.
    pm4.set_uconfig_reg(GE_PC_ALLOC, f(63, 1, 0x3FF));

    // --- Pixel Shader ---
    pm4.set_sh_reg(SPI_SHADER_PGM_LO_PS, (ps_va >> 8) as u32);
    pm4.set_sh_reg(SPI_SHADER_PGM_HI_PS, (ps_va >> 40) as u32);
    let ps_vgpr_blocks = (ps_vgprs / 8).saturating_sub(1).min(63);
    let ps_rsrc1 = f(ps_vgpr_blocks, RSRC1_VGPRS_SHIFT, RSRC1_VGPRS_MASK)
        | RSRC1_FLOAT_MODE_DENORM16_64
        | RSRC1_DX10_CLAMP
        | RSRC1_PS_MEM_ORDERED;
    let ps_rsrc2 = f(PS_USER_SGPRS, RSRC2_USER_SGPR_SHIFT, RSRC2_USER_SGPR_MASK);
    pm4.set_sh_reg(SPI_SHADER_PGM_RSRC1_PS, ps_rsrc1);
    pm4.set_sh_reg(SPI_SHADER_PGM_RSRC2_PS, ps_rsrc2);
    pm4.set_sh_reg(
        SPI_SHADER_PGM_RSRC3_PS,
        f(0xFFFF, RSRC3_CU_EN_SHIFT, RSRC3_CU_EN_MASK) | RSRC3_WAVE_LIMIT_PS,
    );
    pm4.set_sh_reg(SPI_SHADER_PGM_RSRC4_PS, f(0xFFFF, 0, 0xFFFF));

    // Input enables & interpolation: perspective-correct at pixel center.
    pm4.set_context_reg(SPI_PS_INPUT_ENA, ps_input_ena);
    pm4.set_context_reg(SPI_PS_INPUT_ADDR, ps_input_ena);
    pm4.set_context_reg(
        SPI_PS_IN_CONTROL,
        f(num_interp, PS_IN_NUM_INTERP_SHIFT, PS_IN_NUM_INTERP_MASK) | PS_IN_PS_W32_EN,
    );
    pm4.set_context_reg(SPI_BARYC_CNTL, 0); // center evaluation
    pm4.set_context_reg(SPI_INTERP_CONTROL_0, 0);
    // V_INTERP attribute indices address this map, not VS export indices.
    // Color is PARAM0; fog is PARAM2 (PARAM1 contains texture coordinates).
    pm4.set_context_reg(SPI_PS_INPUT_CNTL_0, 0);
    if num_interp > 1 {
        pm4.set_context_reg(SPI_PS_INPUT_CNTL_1, 2);
    }

    // Color export: MRT0 = 32_ABGR, Z export = NONE.
    pm4.set_context_reg(SPI_SHADER_COL_FORMAT, f(SPI_SHADER_32_ABGR, 0, 0xF));
    pm4.set_context_reg(SPI_SHADER_Z_FORMAT, SPI_SHADER_NONE);
    pm4.set_context_reg(CB_SHADER_MASK, CB_SHADER_MASK_RGBA);
}

/// Set VS user SGPRs: 64-bit vertex buffer pointer (`s[0:1]`).
pub fn emit_vs_user_data(pm4: &mut Pm4, vb_va: u64) {
    pm4.set_sh_reg_seq(
        SPI_SHADER_USER_DATA_VS_0,
        &[vb_va as u32, (vb_va >> 32) as u32],
    );
}

/// Set PS user SGPRs: fog / alpha constants (`s[0:6]`).
pub fn emit_ps_user_data(pm4: &mut Pm4, c: &PsConstants) {
    pm4.set_sh_reg_seq(SPI_SHADER_USER_DATA_PS_0, &c.to_sgprs());
}

/// Non-indexed triangle draw (`DRAW_INDEX_AUTO`).
pub fn emit_draw_auto(pm4: &mut Pm4, count: u32) {
    pm4.set_uconfig_reg(VGT_PRIMITIVE_TYPE, DI_PT_TRILIST);
    pm4.num_instances(1);
    pm4.draw_index_auto(count);
}

/// Indexed triangle draw (`DRAW_INDEX_2`).
pub fn emit_draw_indexed(pm4: &mut Pm4, ib_va: u64, index_count: u32, is_u16: bool) {
    pm4.set_uconfig_reg(VGT_PRIMITIVE_TYPE, DI_PT_TRILIST);
    pm4.num_instances(1);
    let (idx_type, max_size) = if is_u16 {
        (VGT_INDEX_16, index_count)
    } else {
        (VGT_INDEX_32, index_count)
    };
    pm4.set_uconfig_reg_idx(VGT_INDEX_TYPE, 2, idx_type);
    pm4.draw_index_2(max_size, ib_va, index_count);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sh_reg(pm4: &Pm4, reg: u32) -> u32 {
        let mut i = 0;
        while i < pm4.dwords.len() {
            let header = pm4.dwords[i];
            let op = (header >> 8) & 0xff;
            let count = ((header >> 16) & 0x3fff) as usize + 1;
            if op == PKT3_SET_SH_REG {
                let start = SI_SH_REG_OFFSET + (pm4.dwords[i + 1] & 0xffff) * 4;
                if reg >= start && (reg - start) / 4 < (count - 1) as u32 {
                    return pm4.dwords[i + 2 + ((reg - start) / 4) as usize];
                }
            }
            i += count + 1;
        }
        panic!("missing SH register {reg:#x}");
    }

    #[test]
    fn gfx10_shader_resources_and_requests() {
        let mut pm4 = Pm4::new();
        emit_preamble(&mut pm4);
        assert_eq!(sh_reg(&pm4, SPI_SHADER_REQ_CTRL_PS), 7);
        assert_eq!(sh_reg(&pm4, SPI_SHADER_USER_ACCUM_PS_0), 0);
        assert_eq!(sh_reg(&pm4, SPI_SHADER_USER_ACCUM_VS_0), 0);

        let mut pm4 = Pm4::new();
        emit_shader_programs(&mut pm4, 0x100000, 16, 0x200000, 8, PERSP_CENTER_ENA, 2);
        let vs = sh_reg(&pm4, SPI_SHADER_PGM_RSRC1_VS);
        let ps = sh_reg(&pm4, SPI_SHADER_PGM_RSRC1_PS);
        assert_eq!(vs & RSRC1_VGPRS_MASK, 1); // 16 VGPRs
        assert_eq!(ps & RSRC1_VGPRS_MASK, 0); // 8 VGPRs
        assert_ne!(vs & RSRC1_VS_MEM_ORDERED, 0);
        assert_ne!(ps & RSRC1_PS_MEM_ORDERED, 0);
        assert_eq!((vs >> 12) & 0xff, 192);
        assert_eq!((ps >> 12) & 0xff, 192);
    }
}
