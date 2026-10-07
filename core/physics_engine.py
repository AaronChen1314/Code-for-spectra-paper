# -*- coding: utf-8 -*-
"""
核心物理引擎模块
"""

import numpy as np
from numba import njit, prange, float64, complex128

# 常量定义
q = 1.60217662e-19       
h = 6.62607004e-34       
c = 299792458.0          
k_B = 1.38064852e-23     
T_STC = 298.15           
REAL_EG_TOP = 1.80  
REAL_EG_BOT = 1.25  
IQE_TOP = 1.00; IQE_BOT = 1.00
SHADING_LOSS = 0.00

@njit(fastmath=True)
def snell_vec_jit(n_1, n_2, th_1):
    sin_th2 = n_1 * np.sin(th_1) / n_2
    return np.arcsin(sin_th2)

@njit(fastmath=True)
def interface_r_vec_jit(pol, n_i, n_f, th_i, th_f):
    if pol == 0: # s
        num = n_i * np.cos(th_i) - n_f * np.cos(th_f)
        den = n_i * np.cos(th_i) + n_f * np.cos(th_f)
        return num / den
    else: # p
        num = n_i * np.cos(th_f) - n_f * np.cos(th_i)
        den = n_i * np.cos(th_f) + n_f * np.cos(th_i)
        return num / den

@njit(fastmath=True)
def interface_t_vec_jit(pol, n_i, n_f, th_i, th_f):
    if pol == 0: # s
        return 2 * n_i * np.cos(th_i) / (n_i * np.cos(th_i) + n_f * np.cos(th_f))
    else: # p
        return 2 * n_i * np.cos(th_f) / (n_i * np.cos(th_f) + n_f * np.cos(th_i))

@njit(fastmath=True)
def calc_absorption_for_angle(ang_deg, wls, n_entry, d_list_full, n_air, n_glass, pol_code):
    num_layers = n_entry.shape[0]
    num_wls = len(wls)
    theta_air = np.full(num_wls, np.radians(ang_deg))
    
    n1_real = np.real(n_air); n2_real = np.real(n_glass)
    sin_th2 = (n1_real * np.sin(theta_air)) / n2_real
    
    sin_th2_safe = np.zeros_like(sin_th2)
    for i in range(num_wls):
        val = sin_th2[i]
        if val > 1.0: val = 1.0
        elif val < -1.0: val = -1.0
        sin_th2_safe[i] = val
        
    theta_glass = np.arcsin(sin_th2_safe)
    
    rs = interface_r_vec_jit(0, n_air, n_glass, theta_air, theta_glass)
    rp = interface_r_vec_jit(1, n_air, n_glass, theta_air, theta_glass)
    
    if pol_code == 0: R_ag = np.abs(rs)**2
    else: R_ag = np.abs(rp)**2
    
    th_list = np.zeros((num_layers, num_wls), dtype=complex128)
    th_list[0] = theta_glass
    
    for i in range(1, num_layers):
        sin_th_next = n_entry[i-1] * np.sin(th_list[i-1]) / n_entry[i]
        th_list[i] = np.arcsin(sin_th_next)
        
    kz_list = 2 * np.pi * n_entry * np.cos(th_list) / wls
    delta = np.zeros((num_layers, num_wls), dtype=complex128)
    for i in range(1, num_layers-1):
        delta[i] = kz_list[i] * d_list_full[i]
        
    r0 = interface_r_vec_jit(pol_code, n_entry[0], n_entry[1], th_list[0], th_list[1])
    t0 = interface_t_vec_jit(pol_code, n_entry[0], n_entry[1], th_list[0], th_list[1])
    
    M00 = 1/t0; M01 = r0/t0; M10 = r0/t0; M11 = 1/t0
    
    I_mats_00 = np.zeros((num_layers-1, num_wls), dtype=complex128)
    I_mats_01 = np.zeros((num_layers-1, num_wls), dtype=complex128)
    I_mats_10 = np.zeros((num_layers-1, num_wls), dtype=complex128)
    I_mats_11 = np.zeros((num_layers-1, num_wls), dtype=complex128)
    
    I_mats_00[0] = 1/t0; I_mats_01[0] = r0/t0
    I_mats_10[0] = r0/t0; I_mats_11[0] = 1/t0
    
    for i in range(1, num_layers-1):
        exp_mdp = np.exp(-1j * delta[i])
        exp_pdp = np.exp(1j * delta[i])
        M00 = M00 * exp_mdp; M10 = M10 * exp_mdp
        M01 = M01 * exp_pdp; M11 = M11 * exp_pdp
        
        r = interface_r_vec_jit(pol_code, n_entry[i], n_entry[i+1], th_list[i], th_list[i+1])
        t = interface_t_vec_jit(pol_code, n_entry[i], n_entry[i+1], th_list[i], th_list[i+1])
        i00 = 1/t; i01 = r/t; i10 = r/t; i11 = 1/t
        
        I_mats_00[i] = i00; I_mats_01[i] = i01
        I_mats_10[i] = i10; I_mats_11[i] = i11
        
        m00_new = M00 * i00 + M01 * i10
        m01_new = M00 * i01 + M01 * i11
        m10_new = M10 * i00 + M11 * i10
        m11_new = M10 * i01 + M11 * i11
        M00, M01, M10, M11 = m00_new, m01_new, m10_new, m11_new
        
    r_total = M10 / M00
    
    v_curr = np.ones(num_wls, dtype=complex128)
    w_curr = r_total
    
    Sz_in = np.zeros((num_layers, num_wls))
    Sz_out = np.zeros((num_layers, num_wls))
    
    n0 = n_entry[0]; cos0 = np.cos(th_list[0])
    if pol_code == 0:
        Y0 = n0 * cos0
    else:
        Y0 = n0 / cos0
    Sz_in[0] = np.real((v_curr + w_curr) * np.conj(Y0 * (v_curr - w_curr)))
    Sz_out[0] = Sz_in[0] 
    Sz_incident = np.real(Y0) 
    
    for i in range(num_layers - 1):
        if i > 0:
            phase = delta[i]
            v_end = v_curr * np.exp(1j * phase)
            w_end = w_curr * np.exp(-1j * phase)
            n_i = n_entry[i]; cos_i = np.cos(th_list[i])
            if pol_code == 0:
                Y_i = n_i * cos_i
            else:
                Y_i = n_i / cos_i
            E_end = v_end + w_end
            H_end = Y_i * (v_end - w_end)
            Sz_out[i] = np.real(E_end * np.conj(H_end))
            v_curr = v_end; w_curr = w_end
        
        i00 = I_mats_00[i]; i01 = I_mats_01[i]
        i10 = I_mats_10[i]; i11 = I_mats_11[i]
        det = i00*i11 - i01*i10
        v_next = (i11 * v_curr - i01 * w_curr) / det
        w_next = (-i10 * v_curr + i00 * w_curr) / det
        v_curr = v_next; w_curr = w_next
        
        n_next = n_entry[i+1]; cos_next = np.cos(th_list[i+1])
        if pol_code == 0:
            Y_next = n_next * cos_next
        else:
            Y_next = n_next / cos_next
        E_start = v_curr + w_curr
        H_start = Y_next * (v_curr - w_curr)
        Sz_in[i+1] = np.real(E_start * np.conj(H_start))

    Sz_out[num_layers-1] = Sz_in[num_layers-1] 
    
    absorp = np.zeros((num_layers, num_wls))
    for i in range(1, num_layers): 
        for w in range(num_wls):
            inc = Sz_incident[w]
            if np.abs(inc) < 1e-20: inc = 1.0
            val = (Sz_in[i, w] - Sz_out[i, w]) / inc
            if val < 0: val = 0
            absorp[i, w] = val
            
    R_stack = np.abs(r_total)**2
    t_total = 1.0 / M00
    factor = (1.0 - R_ag) / (1.0 - R_ag * R_stack)
    # Floating-point field differencing can create small negative layer losses;
    # clipping those independently can make their sum exceed the actual stack
    # loss at oblique incidence. Preserve the relative positive layer losses,
    # but normalize their sum to the energy-conserving total 1-R-T calculated
    # from the same fields. The front air/glass incoherent factor is unchanged.
    for w in range(num_wls):
        inc = Sz_incident[w]
        if np.abs(inc) < 1e-20:
            inc = 1.0
        if pol_code == 0:
            transmission = (np.abs(t_total[w])**2
                            * np.real(n_entry[num_layers - 1, w]
                                      * np.cos(th_list[num_layers - 1, w]))
                            / np.real(n_entry[0, w] * np.cos(th_list[0, w])))
        else:
            transmission = (np.abs(t_total[w])**2
                            * np.real(n_entry[num_layers - 1, w]
                                      * np.conj(np.cos(th_list[num_layers - 1, w])))
                            / np.real(n_entry[0, w]
                                      * np.conj(np.cos(th_list[0, w]))))
        total_stack_absorption = 1.0 - R_stack[w] - transmission
        if total_stack_absorption < 0.0 and total_stack_absorption > -1e-10:
            total_stack_absorption = 0.0
        elif total_stack_absorption < 0.0:
            total_stack_absorption = np.nan
        target = total_stack_absorption * factor[w]
        positive_sum = 0.0
        for i in range(1, num_layers - 1):
            positive_sum += absorp[i, w]
        if positive_sum > 1e-30:
            scale = target / positive_sum
            for i in range(1, num_layers - 1):
                absorp[i, w] *= scale
        else:
            for i in range(1, num_layers - 1):
                absorp[i, w] = 0.0
    return absorp


@njit(fastmath=True)
def calc_total_absorption_for_angle(
        ang_deg, wls, n_entry, d_list_full, n_air, n_glass, pol_code):
    """Return the same energy-conserving total absorption without layer fields."""
    num_layers = n_entry.shape[0]
    num_wls = len(wls)
    theta_air = np.full(num_wls, np.radians(ang_deg))
    sin_th2 = (np.real(n_air) * np.sin(theta_air)) / np.real(n_glass)
    for w in range(num_wls):
        if sin_th2[w] > 1.0:
            sin_th2[w] = 1.0
        elif sin_th2[w] < -1.0:
            sin_th2[w] = -1.0
    theta_glass = np.arcsin(sin_th2)
    rs = interface_r_vec_jit(0, n_air, n_glass, theta_air, theta_glass)
    rp = interface_r_vec_jit(1, n_air, n_glass, theta_air, theta_glass)
    R_ag = np.abs(rs)**2 if pol_code == 0 else np.abs(rp)**2

    th_list = np.zeros((num_layers, num_wls), dtype=complex128)
    th_list[0] = theta_glass
    for layer in range(1, num_layers):
        th_list[layer] = np.arcsin(
            n_entry[layer - 1] * np.sin(th_list[layer - 1]) / n_entry[layer])

    r0 = interface_r_vec_jit(
        pol_code, n_entry[0], n_entry[1], th_list[0], th_list[1])
    t0 = interface_t_vec_jit(
        pol_code, n_entry[0], n_entry[1], th_list[0], th_list[1])
    M00 = 1.0 / t0
    M01 = r0 / t0
    M10 = r0 / t0
    M11 = 1.0 / t0
    for layer in range(1, num_layers - 1):
        delta = (2.0 * np.pi * n_entry[layer]
                 * np.cos(th_list[layer]) / wls * d_list_full[layer])
        M00 = M00 * np.exp(-1j * delta)
        M10 = M10 * np.exp(-1j * delta)
        M01 = M01 * np.exp(1j * delta)
        M11 = M11 * np.exp(1j * delta)
        r = interface_r_vec_jit(
            pol_code, n_entry[layer], n_entry[layer + 1],
            th_list[layer], th_list[layer + 1])
        t = interface_t_vec_jit(
            pol_code, n_entry[layer], n_entry[layer + 1],
            th_list[layer], th_list[layer + 1])
        i00 = 1.0 / t
        i01 = r / t
        m00_new = M00 * i00 + M01 * i01
        m01_new = M00 * i01 + M01 * i00
        m10_new = M10 * i00 + M11 * i01
        m11_new = M10 * i01 + M11 * i00
        M00, M01, M10, M11 = m00_new, m01_new, m10_new, m11_new

    r_total = M10 / M00
    t_total = 1.0 / M00
    R_stack = np.abs(r_total)**2
    total = np.empty(num_wls)
    for w in range(num_wls):
        if pol_code == 0:
            transmission = (np.abs(t_total[w])**2
                            * np.real(n_entry[num_layers - 1, w]
                                      * np.cos(th_list[num_layers - 1, w]))
                            / np.real(n_entry[0, w] * np.cos(th_list[0, w])))
        else:
            transmission = (np.abs(t_total[w])**2
                            * np.real(n_entry[num_layers - 1, w]
                                      * np.conj(np.cos(th_list[num_layers - 1, w])))
                            / np.real(n_entry[0, w]
                                      * np.conj(np.cos(th_list[0, w]))))
        stack_absorption = 1.0 - R_stack[w] - transmission
        if stack_absorption < 0.0 and stack_absorption > -1e-10:
            stack_absorption = 0.0
        elif stack_absorption < 0.0:
            stack_absorption = np.nan
        total[w] = (stack_absorption * (1.0 - R_ag[w])
                    / (1.0 - R_ag[w] * R_stack[w]))
    return total


@njit(parallel=True, cache=True)
def generate_total_absorptance_lut_numba(
        lut_angles, wls, n_entry, d_list_full, n_air, n_glass):
    """Unpolarized total-device absorptance LUT without layer-resolved fields."""
    lut = np.empty((len(lut_angles), len(wls)))
    for index in prange(len(lut_angles)):
        abs_s = calc_total_absorption_for_angle(
            lut_angles[index], wls, n_entry, d_list_full, n_air, n_glass, 0)
        abs_p = calc_total_absorption_for_angle(
            lut_angles[index], wls, n_entry, d_list_full, n_air, n_glass, 1)
        lut[index] = 0.5 * (abs_s + abs_p)
    return lut

@njit(parallel=True, cache=True)
def generate_lut_numba(lut_angles, wls, n_entry, d_list_full, n_air, n_glass):
    num_angles = len(lut_angles)
    num_layers = len(d_list_full)
    num_wls = len(wls)
    lut = np.zeros((num_angles, num_layers, num_wls))
    for i in prange(num_angles):
        ang = lut_angles[i]
        abs_s = calc_absorption_for_angle(ang, wls, n_entry, d_list_full, n_air, n_glass, 0)
        abs_p = calc_absorption_for_angle(ang, wls, n_entry, d_list_full, n_air, n_glass, 1)
        for L in range(num_layers):
            for W in range(num_wls):
                lut[i, L, W] = 0.5 * (abs_s[L, W] + abs_p[L, W])
    return lut

@njit(fastmath=True, cache=True)
def get_V_and_derivatives(J_target, J_ph, J01, J02, n1, n2, Rs, Rsh, Vt):
    if J_target >= J_ph: 
        Vd = - Rsh * (J_target - J_ph)
    else:
        if J_ph > 1e-9: 
            Vd = n1 * Vt * np.log(max(J_ph - J_target, 1e-12) / J01 + 1)
        else: 
            Vd = 0.6
            
    for _ in range(20):
        term1 = np.exp(Vd / (n1*Vt))
        term2 = np.exp(Vd / (n2*Vt))
        f_val = J_ph - J_target - J01*(term1 - 1) - J02*(term2 - 1) - Vd/Rsh
        df_val = - (J01/(n1*Vt))*term1 - (J02/(n2*Vt))*term2 - 1.0/Rsh
        if abs(df_val) < 1e-12: break
        diff = f_val / df_val
        Vd = Vd - diff
        if abs(diff) < 1e-5: break

    term1 = np.exp(Vd / (n1*Vt))
    term2 = np.exp(Vd / (n2*Vt))
    dJ_dVd = - (J01 / (n1 * Vt)) * term1 - (J02 / (n2 * Vt)) * term2 - 1.0 / Rsh
    d2J_dVd2 = - (J01 / ((n1 * Vt)**2)) * term1 - (J02 / ((n2 * Vt)**2)) * term2
    
    dV_dJ = 1.0 / dJ_dVd - Rs
    d2V_dJ2 = - (dJ_dVd ** -3) * d2J_dVd2
    V = Vd - J_target * Rs
    return V, dV_dJ, d2V_dJ2

@njit(fastmath=True, cache=True)
def solve_single_IV_point_fast(J_ph_top, J_ph_bot, T, top_params, bot_params, Rs_tj=0.0):
    Vt = k_B * T / q
    J01_t, J02_t, n1_t, n2_t, Rs_t, Rsh_t = top_params
    J01_b, J02_b, n1_b, n2_b, Rs_b, Rsh_b = bot_params
    J_lim = min(J_ph_top, J_ph_bot)
    if J_lim <= 1e-9: return 0.0

    # Use a safeguarded Newton solve inside the physical current interval.
    # The former unbracketed update could jump between roots as temperature
    # changed by only millikelvins, creating discontinuous Pmax and preventing
    # the electro-thermal fixed point from converging.
    J_low = 0.0
    J_high = J_lim * (1.0 - 1e-12)
    J_curr = 0.9 * J_lim

    for _ in range(50):
        v_t, dVt_dJ, d2Vt_dJ2 = get_V_and_derivatives(J_curr, J_ph_top, J01_t, J02_t, n1_t, n2_t, Rs_t, Rsh_t, Vt)
        v_b, dVb_dJ, d2Vb_dJ2 = get_V_and_derivatives(J_curr, J_ph_bot, J01_b, J02_b, n1_b, n2_b, Rs_b, Rsh_b, Vt)
        
        # 功率一阶与二阶导数
        dP_dJ = v_t + v_b - 2.0 * J_curr * Rs_tj + J_curr * (dVt_dJ + dVb_dJ)
        d2P_dJ2 = 2.0 * (dVt_dJ + dVb_dJ - Rs_tj) + J_curr * (d2Vt_dJ2 + d2Vb_dJ2)
        
        if dP_dJ > 0.0:
            J_low = J_curr
        else:
            J_high = J_curr
        if J_high - J_low < max(1e-13, J_lim * 1e-10):
            break

        if np.isfinite(d2P_dJ2) and abs(d2P_dJ2) >= 1e-12:
            J_next = J_curr - dP_dJ / d2P_dJ2
        else:
            J_next = 0.5 * (J_low + J_high)
        if (not np.isfinite(J_next) or J_next <= J_low or J_next >= J_high):
            J_next = 0.5 * (J_low + J_high)
        J_curr = J_next
        
    v_t, _, _ = get_V_and_derivatives(J_curr, J_ph_top, J01_t, J02_t, n1_t, n2_t, Rs_t, Rsh_t, Vt)
    v_b, _, _ = get_V_and_derivatives(J_curr, J_ph_bot, J01_b, J02_b, n1_b, n2_b, Rs_b, Rsh_b, Vt)
    return J_curr * (v_t + v_b - J_curr * Rs_tj)


@njit(parallel=True, fastmath=True, cache=True)
def calculate_annual_energy_numba(
        J_top_arr, J_bot_arr, T_cell_arr, top_p, bot_p, alpha_coeffs,
        eg_top=1.80, eg_bot=1.25, rs_tj=0.0,
        eg_temp_coeff_nbg=0.72e-3,
        temp_flags=(True, True, True)):
    """
    计算全年累积发电量（Wh/cm²）。
    
    温度物理修正:
    - NBG 带隙使用线性正温度系数公式 Eg(T) = Eg(T_STC) + 0.72e-3 × (T - T_STC)
    - WBG带隙采用+0.31 meV/K线性温度系数
    - 移除光生电流敏感参数估算，直接依赖由TMM光学层计算的 Jsc(T)
    """
    N = len(J_top_arr)
    total_energy = 0.0
    A1 = alpha_coeffs[0]; A2 = alpha_coeffs[1]; A3 = alpha_coeffs[2]; A4 = alpha_coeffs[3]; A5 = alpha_coeffs[4]

    # 计算标称态(STC 25℃)下暗饱和电流随带隙偏移的指数缩放因子
    Vt_stc = k_B * T_STC / q

    # 顶电池：WBG 基准带隙（线性公式，无额外 STC 偏移）
    n1_t_ref = top_p[2]; n2_t_ref = top_p[3]
    delta_eg_top = eg_top - 1.80
    top_j01_scale = np.exp(-delta_eg_top / (n1_t_ref * Vt_stc))
    top_j02_scale = np.exp(-delta_eg_top / (n2_t_ref * Vt_stc))
    # Rsh 随带隙缩放（经验关系）
    top_rsh_scale = np.exp(-delta_eg_top * 1.5)

    # 底电池：NBG 基准带隙（线性温漂，无需预计算 STC 偏移）
    n1_b_ref = bot_p[2]; n2_b_ref = bot_p[3]
    delta_eg_bot = eg_bot - 1.25
    bot_j01_scale = np.exp(-delta_eg_bot / (n1_b_ref * Vt_stc))
    bot_j02_scale = np.exp(-delta_eg_bot / (n2_b_ref * Vt_stc))
    bot_rsh_scale = np.exp(-delta_eg_bot * 1.5)

    for i in prange(N):
        T = T_cell_arr[i]
        T_eg = T if temp_flags[0] else T_STC
        T_j0 = T if temp_flags[1] else T_STC
        T_iv = T if temp_flags[2] else T_STC

        ratio = T_j0 / T_STC
        Vt_cell_j0 = k_B * T_j0 / q

        # ── 需求1: WBG 带隙温漂（线性公式，正温度系数 +0.31 meV/K） ──
        Eg_t = eg_top + 3.1e-4 * (T_eg - T_STC)

        # ── 需求1: NBG 带隙温漂（线性公式，正温度系数 +0.72 meV/K） ──
        Eg_b = eg_bot + eg_temp_coeff_nbg * (T_eg - T_STC)

        # ── TMM 光学模型已天然含温漂效应，此处直接读取光电流 ──
        J_t = J_top_arr[i] * 1e-3
        J_b = J_bot_arr[i] * 1e-3

        # ── 顶电池暗电流温度缩放 (阻尼温度收缩二阶理想因子，w=0.43) ──
        n1_t = top_p[2]; n2_t_ref = top_p[3]
        n2_t = 0.43 * n2_t_ref + 0.57 * (1.0 + (n2_t_ref - 1.0) * (T_STC / T_j0))
        term1 = ratio**(A2/n1_t)
        term2 = np.exp( eg_top / (n1_t * Vt_stc) - Eg_t / (n1_t * Vt_cell_j0) )
        j01_t = (top_p[0] * top_j01_scale) * term1 * term2

        term1b = ratio**(A3/n2_t)
        term2b = np.exp( eg_top / (n2_t_ref * Vt_stc) - Eg_t / (n2_t * Vt_cell_j0) )
        j02_t = (top_p[1] * top_j02_scale) * term1b * term2b
        rs_t = top_p[4] * (ratio**A4)
        rsh_t = top_p[5] * top_rsh_scale * (ratio**A5)
        p_t_curr = (j01_t, j02_t, n1_t, n2_t, rs_t, rsh_t)

        # ── 底电池暗电流温度缩放 (阻尼温度收缩二阶理想因子，w=0.43) ──
        n1_b = bot_p[2]; n2_b_ref = bot_p[3]
        n2_b = 0.43 * n2_b_ref + 0.57 * (1.0 + (n2_b_ref - 1.0) * (T_STC / T_j0))
        term1 = ratio**(A2/n1_b)
        term2 = np.exp( eg_bot / (n1_b * Vt_stc) - Eg_b / (n1_b * Vt_cell_j0) )
        j01_b = (bot_p[0] * bot_j01_scale) * term1 * term2

        term1b = ratio**(A3/n2_b)
        term2b = np.exp( eg_bot / (n2_b_ref * Vt_stc) - Eg_b / (n2_b * Vt_cell_j0) )
        j02_b = (bot_p[1] * bot_j02_scale) * term1b * term2b
        rs_b = bot_p[4] * (ratio**A4)
        rsh_b = bot_p[5] * bot_rsh_scale * (ratio**A5)
        p_b_curr = (j01_b, j02_b, n1_b, n2_b, rs_b, rsh_b)

        p_max = solve_single_IV_point_fast(J_t * IQE_TOP, J_b * IQE_BOT, T_iv, p_t_curr, p_b_curr, rs_tj)
        
        # 外层整机 Pmax 温度修正（A1 当前为 0.0，预留接口）
        if temp_flags[0] and temp_flags[1] and temp_flags[2]:
            p_max *= (1.0 + A1 * (T - T_STC))
            
        total_energy += p_max
    return total_energy

@njit(parallel=True, fastmath=True, cache=True)
def calculate_hourly_power_numba(
        J_top_arr, J_bot_arr, T_cell_arr, top_p, bot_p, alpha_coeffs,
        eg_top=1.80, eg_bot=1.25, rs_tj=0.0,
        eg_temp_coeff_nbg=0.72e-3,
        temp_flags=(True, True, True)):
    """
    计算逐小时最大功率密度数组（W/cm²）。
    与 calculate_annual_energy_numba 同步的物理修正。
    """
    N = len(J_top_arr)
    p_max_arr = np.zeros(N)
    A1 = alpha_coeffs[0]; A2 = alpha_coeffs[1]; A3 = alpha_coeffs[2]; A4 = alpha_coeffs[3]; A5 = alpha_coeffs[4]

    # 标称态带隙缩放因子
    Vt_stc = k_B * T_STC / q
    
    n1_t_ref = top_p[2]; n2_t_ref = top_p[3]
    delta_eg_top = eg_top - 1.80
    top_j01_scale = np.exp(-delta_eg_top / (n1_t_ref * Vt_stc))
    top_j02_scale = np.exp(-delta_eg_top / (n2_t_ref * Vt_stc))
    top_rsh_scale = np.exp(-delta_eg_top * 1.5)

    n1_b_ref = bot_p[2]; n2_b_ref = bot_p[3]
    delta_eg_bot = eg_bot - 1.25
    bot_j01_scale = np.exp(-delta_eg_bot / (n1_b_ref * Vt_stc))
    bot_j02_scale = np.exp(-delta_eg_bot / (n2_b_ref * Vt_stc))
    bot_rsh_scale = np.exp(-delta_eg_bot * 1.5)

    for i in prange(N):
        T = T_cell_arr[i]
        T_eg = T if temp_flags[0] else T_STC
        T_j0 = T if temp_flags[1] else T_STC
        T_iv = T if temp_flags[2] else T_STC

        ratio = T_j0 / T_STC
        Vt_cell_j0 = k_B * T_j0 / q

        # WBG 带隙温漂（线性公式，+0.31 meV/K）
        Eg_t = eg_top + 3.1e-4 * (T_eg - T_STC)
        # NBG 带隙温漂（线性公式，+0.72 meV/K）
        Eg_b = eg_bot + eg_temp_coeff_nbg * (T_eg - T_STC)

        # 光学层 TMM 已经包含带隙温漂带来的 Jsc 变化，直接读取
        J_t = J_top_arr[i] * 1e-3
        J_b = J_bot_arr[i] * 1e-3

        # ── 顶电池暗电流温度缩放 (阻尼温度收缩二阶理想因子，w=0.43) ──
        n1_t = top_p[2]; n2_t_ref = top_p[3]
        n2_t = 0.43 * n2_t_ref + 0.57 * (1.0 + (n2_t_ref - 1.0) * (T_STC / T_j0))
        term1 = ratio**(A2/n1_t)
        term2 = np.exp( eg_top / (n1_t * Vt_stc) - Eg_t / (n1_t * Vt_cell_j0) )
        j01_t = (top_p[0] * top_j01_scale) * term1 * term2

        term1b = ratio**(A3/n2_t)
        term2b = np.exp( eg_top / (n2_t_ref * Vt_stc) - Eg_t / (n2_t * Vt_cell_j0) )
        j02_t = (top_p[1] * top_j02_scale) * term1b * term2b
        rs_t = top_p[4] * (ratio**A4)
        rsh_t = top_p[5] * top_rsh_scale * (ratio**A5)
        p_t_curr = (j01_t, j02_t, n1_t, n2_t, rs_t, rsh_t)

        # ── 底电池暗电流温度缩放 (阻尼温度收缩二阶理想因子，w=0.43) ──
        n1_b = bot_p[2]; n2_b_ref = bot_p[3]
        n2_b = 0.43 * n2_b_ref + 0.57 * (1.0 + (n2_b_ref - 1.0) * (T_STC / T_j0))
        term1 = ratio**(A2/n1_b)
        term2 = np.exp( eg_bot / (n1_b * Vt_stc) - Eg_b / (n1_b * Vt_cell_j0) )
        j01_b = (bot_p[0] * bot_j01_scale) * term1 * term2

        term1b = ratio**(A3/n2_b)
        term2b = np.exp( eg_bot / (n2_b_ref * Vt_stc) - Eg_b / (n2_b * Vt_cell_j0) )
        j02_b = (bot_p[1] * bot_j02_scale) * term1b * term2b
        rs_b = bot_p[4] * (ratio**A4)
        rsh_b = bot_p[5] * bot_rsh_scale * (ratio**A5)
        p_b_curr = (j01_b, j02_b, n1_b, n2_b, rs_b, rsh_b)

        p_max = solve_single_IV_point_fast(J_t * IQE_TOP, J_b * IQE_BOT, T_iv, p_t_curr, p_b_curr, rs_tj)
        
        # 外层整机 Pmax 温度修正（A1 当前为 0.0，预留接口）
        if temp_flags[0] and temp_flags[1] and temp_flags[2]:
            p_max *= (1.0 + A1 * (T - T_STC))
            
        p_max_arr[i] = p_max
    return p_max_arr
