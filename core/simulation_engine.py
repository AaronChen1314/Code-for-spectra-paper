# -*- coding: utf-8 -*-
"""
仿真引擎模块
"""

import numpy as np
try:
    from scipy.integrate import trapezoid
except Exception:
    trapezoid = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

import os
import glob
from data_io.file_io import FileIO
from core.physics_engine import (
    generate_lut_numba, generate_total_absorptance_lut_numba,
    calculate_annual_energy_numba,
    calculate_hourly_power_numba,
    interface_r_vec_jit, interface_t_vec_jit,
    q, h, c, T_STC
)
from utils.physics_utils import Joule_Physics, Model_Utils
from config.config_manager import get_config
from simulation_logging.log_manager import get_logger

# Try to import cache modules, if not available, disable caching
try:
    from cache.cache_manager import CacheManager, LUTCache, SimulationCache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    CacheManager = None
    LUTCache = None
    SimulationCache = None

logger = get_logger()
config = get_config()

class SimulationEngine:
    """仿真引擎类"""
    
    def __init__(self, data_folder='data'):
        """初始化仿真引擎"""
        self.data_dir = data_folder
        self.wls = np.array(config.SIM_WLS, dtype=float)
        self.materials = {}
        self.glass_file = 'nk_glass.txt'
        self.current_thick_wbg = config.DEFAULT_THICKNESS_WBG
        self.current_thick_nbg = config.DEFAULT_THICKNESS_NBG
        self.current_bandgap_wbg = config.REAL_EG_TOP  # 当前WBG带隙
        self.current_bandgap_nbg = config.REAL_EG_BOT  # 当前NBG带隙
        self._nbg_temperature_nk_cache = {}
        
        # 初始化缓存
        if config.ENABLE_CACHE and CACHE_AVAILABLE:
            self.cache_manager = CacheManager(
                cache_dir=config.CACHE_DIR,
                max_size_gb=config.CACHE_MAX_SIZE_GB
            )
            self.lut_cache = LUTCache(self.cache_manager)
            self.sim_cache = SimulationCache(self.cache_manager)
        else:
            self.cache_manager = None
            self.lut_cache = None
            self.sim_cache = None
        
        self.load_materials()
        self.lut_angles = np.linspace(0, 90, 46) 
        self._load_am15g_spectrum()
        
    def _load_am15g_spectrum(self):
        """加载真实的 AM1.5G 标准光谱数据用于热物理模型加权"""
        try:
            # 数据目录通常位于项目根目录下的 data 目录
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            std_spec_folder = os.path.join(BASE_DIR, 'data', 'nrel_meteosat_标准光谱适配模型版(25度)_2019_one_axis')
            
            candidates = glob.glob(os.path.join(std_spec_folder, "*.csv"))
            target_csv = candidates[0]
            for cc in candidates:
                bn = os.path.basename(cc).lower()
                if not any(k in bn for k in ['summary', 'matrix', 'result', 'compare', 'jsub']):
                    target_csv = cc
                    break
                    
            df_std, wls_std, spec_std, col_std = FileIO.load_nrel_data_robust(target_csv)
            if spec_std is not None and spec_std.ndim == 1:
                spec_std = spec_std[None, :]
                
            # 插值到仿真波长网格
            std_spectrum_raw = np.interp(self.wls, wls_std, spec_std[0], left=0, right=0)
            
            # 归一化到标准 1000 W/m²
            std_spec_total_power = trapezoid(spec_std[0], wls_std)
            scale_to_1000 = 1000.0 / std_spec_total_power if std_spec_total_power > 0 else 1.0
            
            self.am15g_spectrum = std_spectrum_raw * scale_to_1000
            logger.info("Successfully loaded AM1.5G spectrum for thermal absorption calculation.")
        except Exception as e:
            logger.warning(f"Failed to load AM1.5G spectrum, falling back to uniform: {e}")
            self.am15g_spectrum = None
        
    def reload_stack_definition(self):
        """重新加载堆叠定义"""
        names  = ['ITO', 'NiO', 'WBG', 'C60_1', 'SnO2', 'Au', 'PEDOT', 'NBG', 'C60_2', 'BCP', 'Ag']
        # Ag is a finite 150 nm layer. Callers append a semi-infinite Air exit
        # medium so that this thickness and the Ag parasitic absorption are active.
        thicks = [150, 7, self.current_thick_wbg, 20, 20, 1, 18, self.current_thick_nbg, 20, 4, 150]
        n_stack = []; d_stack = []
        for i, name in enumerate(names):
            mat_key = name.split('_')[0]
            n_stack.append(self.materials[mat_key])
            d_stack.append(thicks[i])
        return np.array(n_stack), np.array(d_stack, dtype=float)

    def set_absorber_thickness(self, t_wbg, t_nbg):
        """设置吸收层厚度"""
        self.current_thick_wbg = t_wbg
        self.current_thick_nbg = t_nbg

    def _stack_on_wavelength_grid(
            self, wavelengths, temperature_k, shifted_absorbers=None):
        """Build the existing device stack on an arbitrary wavelength grid.

        Outside each material's measured nk interval the explicitly selected
        boundary condition is vacuum-like n=1, k=0. Glass remains the incident
        medium; a semi-infinite Air exit makes the 150 nm Ag layer finite.
        """
        wavelengths = np.asarray(wavelengths, dtype=float)
        material_keys = ['ITO', 'NiO', 'WBG', 'C60', 'SnO2', 'Au',
                         'PEDOT', 'NBG', 'C60', 'BCP', 'Ag']
        thicknesses = np.array([150, 7, self.current_thick_wbg, 20, 20, 1,
                                18, self.current_thick_nbg, 20, 4, 150], dtype=float)
        values = {}
        for key in set(material_keys + ['Glass']):
            wl, n_raw, k_raw = self.raw_material_data[key]
            if shifted_absorbers is not None and key in shifted_absorbers:
                n_raw, k_raw = shifted_absorbers[key]
            elif key == 'WBG':
                eg = self.current_bandgap_wbg + 3.1e-4 * (temperature_k - T_STC)
                n_raw, k_raw = Model_Utils.shift_nk_by_bandgap(
                    wl, n_raw, k_raw, config.NK_SOURCE_EG_TOP, eg)
            elif key == 'NBG':
                eg = self.current_bandgap_nbg + config.EG_TEMP_COEFF_NBG * (temperature_k - T_STC)
                n_raw, k_raw = Model_Utils.shift_nk_by_bandgap(
                    wl, n_raw, k_raw, config.NK_SOURCE_EG_BOT, eg)
            values[key] = (np.interp(wavelengths, wl, n_raw, left=1.0, right=1.0)
                           + 1j*np.interp(wavelengths, wl, k_raw, left=0.0, right=0.0))
        n_layers = np.vstack([values[key] for key in material_keys])
        n_entry = np.vstack([values['Glass'][None, :], n_layers,
                             np.ones((1, len(wavelengths)), dtype=complex)])
        d_list = np.concatenate(([0.0], thicknesses, [0.0]))
        return n_entry, d_list, values['Glass']

    def _thermal_absorptance_luts(
            self, wavelengths, temperature_grid, shifted_absorbers=None):
        """Return total-device absorptance LUTs for the full thermal spectrum."""
        direct = []
        diffuse = []
        theta_rad = np.radians(self.lut_angles)
        weights = np.sin(2 * theta_rad)
        norm = trapezoid(weights, theta_rad)
        air = np.ones(len(wavelengths), dtype=complex)
        if shifted_absorbers is not None and len(shifted_absorbers) != len(temperature_grid):
            raise ValueError("shifted_absorbers must align with temperature_grid")
        for temperature_index, temperature_k in enumerate(temperature_grid):
            precomputed = (None if shifted_absorbers is None
                           else shifted_absorbers[temperature_index])
            n_entry, d_list, glass = self._stack_on_wavelength_grid(
                wavelengths, temperature_k, shifted_absorbers=precomputed)
            # The exact 90-degree transfer-matrix limit is singular when
            # adjacent out-of-range media both use the n=1, k=0 fallback.
            # Evaluate only that endpoint just below grazing incidence while
            # retaining the nominal 0..90-degree interpolation grid.
            safe_angles = np.minimum(self.lut_angles, 89.999)
            total_lut = generate_total_absorptance_lut_numba(
                safe_angles, wavelengths, n_entry, d_list, air, glass)
            direct.append(total_lut)
            diffuse.append(trapezoid(total_lut * weights[:, None], theta_rad, axis=0) / norm)
        return np.asarray(direct), np.asarray(diffuse)
    
    def set_bandgap(self, eg_wbg=None, eg_nbg=None):
        """设置带隙值并重新加载材料
        
        Args:
            eg_wbg: 宽带隙材料的带隙值（eV），None表示不改变
            eg_nbg: 窄带隙材料的带隙值（eV），None表示不改变
        """
        if eg_wbg is not None:
            self.current_bandgap_wbg = eg_wbg
        if eg_nbg is not None:
            self.current_bandgap_nbg = eg_nbg
        # Only the two absorber nk arrays depend on bandgap.  Re-reading and
        # reparsing every material file at every scan point is unnecessary and
        # adds tens of minutes of aggregate I/O to the 45,024-point run.
        # Rebuild the same STC absorber arrays directly from the immutable raw
        # data; the numerical path is identical to load_materials().
        for key, source_eg, target_eg in (
                ('WBG', config.NK_SOURCE_EG_TOP, self.current_bandgap_wbg),
                ('NBG', config.NK_SOURCE_EG_BOT, self.current_bandgap_nbg)):
            wl_raw, n_raw, k_raw = self.raw_material_data[key]
            n_new, k_new = Model_Utils.shift_nk_by_bandgap(
                wl_raw, n_raw, k_raw, source_eg, target_eg)
            self.materials[key] = (
                np.interp(self.wls, wl_raw, n_new, left=1.0, right=1.0)
                + 1j * np.interp(self.wls, wl_raw, k_new, left=0.0, right=0.0)
            )

    def load_materials(self):
        """加载材料光学常数"""
        # 显式记录各材料源文件的波长单位，避免依赖数值范围猜测。
        # 取値范围: 'um' = 微米，'nm' = 纳米。
        # 此表可根据实际数据来源进行扩充。
        MATERIAL_UNIT_MAP = {
            'nk_au.txt':   'um',   # Au nk数据：波长范围 0.31~1.70 微米
            'nk_ag.txt':   'um',   # Ag nk数据：波长范围 0.19~1.94 微米
            'nk_glass.txt': 'nm',  # 玻璃：217~6700 nm
            'nk_ito.txt':  'nm',   # ITO：193~1689 nm
            'nk_nio.txt':  'nm',   # NiO
            'nk_wbg_perovskite.txt': 'nm',  # WBG 钙钛矿: 301~1333 nm
            'nk_nbg_perovskite.txt': 'nm',  # NBG 钙钛矿: 301~1058 nm
            'nk_c60.txt':  'nm',   # C60
            'nk_sno2.txt': 'nm',   # SnO2
            'nk_pedot.txt': 'nm',  # PEDOT
            'nk_bcp.txt':  'nm',   # BCP
        }

        self.materials['Air'] = np.ones_like(self.wls) + 0j
        self.raw_material_data = {}
        path_glass = os.path.join(self.data_dir, self.glass_file)
        if not os.path.exists(path_glass):
            logger.warning(f"Glass file not found: {path_glass}")
            return

        gw, gn, gk = Model_Utils.parse_nk_data(path_glass)
        if gw is None:
            logger.warning("Failed to parse glass data")
            return

        self.raw_material_data['Glass'] = (gw.copy(), gn.copy(), gk.copy())
        self.materials['Glass'] = (np.interp(self.wls, gw, gn, left=1.0, right=1.0)
                                   + 1j*np.interp(self.wls, gw, gk, left=0.0, right=0.0))

        file_map = {
            'ITO': 'nk_ito.txt', 'NiO': 'nk_nio.txt', 'WBG': 'nk_wbg_perovskite.txt',
            'C60': 'nk_c60.txt', 'SnO2': 'nk_sno2.txt', 'Au': 'nk_au.txt',
            'PEDOT': 'nk_pedot.txt', 'NBG': 'nk_nbg_perovskite.txt',
            'BCP': 'nk_bcp.txt', 'Ag': 'nk_ag.txt'
        }

        for key, fname in file_map.items():
            path = os.path.join(self.data_dir, fname)
            if not os.path.exists(path):
                logger.warning(f"Material file not found: {path}")
                continue

            wl_raw, n_raw, k_raw = Model_Utils.parse_nk_data(path)
            if wl_raw is None:
                logger.warning(f"Failed to parse material data: {key}")
                continue

            # 优先使用显式单位；仅对未登记文件回退到数值范围判断。
            unit_hint = MATERIAL_UNIT_MAP.get(fname, 'auto')
            if unit_hint == 'um' or (unit_hint == 'auto' and np.max(wl_raw) < 100.0):
                wl_raw *= 1000.0  # 微米 → 纳米

            self.raw_material_data[key] = (wl_raw.copy(), n_raw.copy(), k_raw.copy())

            if key == 'WBG':
                self.raw_wbg_nk = (wl_raw, n_raw, k_raw)
                n_raw, k_raw = Model_Utils.shift_nk_by_bandgap(
                    wl_raw, n_raw, k_raw,
                    config.NK_SOURCE_EG_TOP, self.current_bandgap_wbg
                )
            elif key == 'NBG':
                self.raw_nbg_nk = (wl_raw, n_raw, k_raw)
                n_raw, k_raw = Model_Utils.shift_nk_by_bandgap(
                    wl_raw, n_raw, k_raw,
                    config.NK_SOURCE_EG_BOT, self.current_bandgap_nbg
                )

            self.materials[key] = (np.interp(self.wls, wl_raw, n_raw, left=1.0, right=1.0)
                                   + 1j*np.interp(self.wls, wl_raw, k_raw, left=0.0, right=0.0))

    def run_simulation_step(self, env_data, city_name=None, return_details=False, temp_flags=(True, True, True), initial_t_cell=None, return_temperature=False):
        """运行单步仿真"""

        # 缓存键同时包含厚度与带隙；特殊温度拆解计算不使用缓存。
        # 若带有特殊拆解温度标志，则直接绕过缓存（不读也不写）
        if not return_details and self.sim_cache and city_name and temp_flags == (True, True, True):
            cache_key = (
                f"{city_name}_{self.current_thick_wbg}_{self.current_thick_nbg}"
                f"_{self.current_bandgap_wbg:.4f}_{self.current_bandgap_nbg:.4f}"
            )
            cached_result = self.sim_cache.get_simulation(
                city_name, self.current_thick_wbg, self.current_thick_nbg,
                bandgap_wbg=self.current_bandgap_wbg,
                bandgap_nbg=self.current_bandgap_nbg
            )
            if cached_result is not None:
                logger.info(
                    f"Using cached result for {city_name} "
                    f"{self.current_thick_wbg}/{self.current_thick_nbg} "
                    f"Eg={self.current_bandgap_wbg:.3f}/{self.current_bandgap_nbg:.3f} eV"
                )
                return cached_result
        
        T_eg_active = temp_flags[0]
        T_STC = 298.15
        T_hourly = env_data['t_cell'] if T_eg_active else np.full_like(env_data['t_cell'], T_STC)
        
        # 建立插值网格以加速 TMM 计算
        if not T_eg_active or (np.max(T_hourly) - np.min(T_hourly) < 0.1):
            T_grid = np.array([T_hourly[0]])
        else:
            T_min = np.min(T_hourly) - 2.0
            T_max = np.max(T_hourly) + 2.0
            T_grid = np.linspace(T_min, T_max, 13)
            
        lut_T_list = []
        diffuse_abs_T_list = []
        shifted_absorbers_T = []
        temp_shift_stc_wbg = 0.0
        
        for T_k in T_grid:
            Eg_top_T = self.current_bandgap_wbg + 3.1e-4 * (T_k - T_STC)
            Eg_bot_T = self.current_bandgap_nbg + config.EG_TEMP_COEFF_NBG * (T_k - T_STC)
            
            # 动态平移带隙
            wbg_wl, wbg_n, wbg_k = self.raw_wbg_nk
            nbg_wl, nbg_n, nbg_k = self.raw_nbg_nk
            
            n_wbg_new, k_wbg_new = Model_Utils.shift_nk_by_bandgap(
                wbg_wl, wbg_n, wbg_k, config.NK_SOURCE_EG_TOP, Eg_top_T
            )
            nbg_cache_key = (float(self.current_bandgap_nbg), float(T_k))
            cached_nbg = self._nbg_temperature_nk_cache.get(nbg_cache_key)
            if cached_nbg is None:
                cached_nbg = Model_Utils.shift_nk_by_bandgap(
                    nbg_wl, nbg_n, nbg_k, config.NK_SOURCE_EG_BOT, Eg_bot_T)
                self._nbg_temperature_nk_cache[nbg_cache_key] = cached_nbg
            n_nbg_new, k_nbg_new = cached_nbg
            shifted_absorbers_T.append({
                'WBG': (n_wbg_new, k_wbg_new),
                'NBG': (n_nbg_new, k_nbg_new),
            })
            
            self.materials['WBG'] = (
                np.interp(self.wls, wbg_wl, n_wbg_new, left=1.0, right=1.0)
                + 1j * np.interp(self.wls, wbg_wl, k_wbg_new, left=0.0, right=0.0)
            )
            self.materials['NBG'] = (
                np.interp(self.wls, nbg_wl, n_nbg_new, left=1.0, right=1.0)
                + 1j * np.interp(self.wls, nbg_wl, k_nbg_new, left=0.0, right=0.0)
            )
            
            n_stack, d_stack = self.reload_stack_definition()
            n_entry = np.vstack([self.materials['Glass'][None, :], n_stack,
                                 self.materials['Air'][None, :]])
            d_list_full = np.concatenate(([0.0], d_stack, [0.0]))
            
            # 由于温度网格极小(13次计算，不到1秒)，可直接调用 numba 加速版求解而无需走外部缓存
            # Avoid the exact grazing-incidence singularity without changing
            # the physical boundary model or nominal interpolation grid.
            safe_angles = np.minimum(self.lut_angles, 89.999)
            lut = generate_lut_numba(
                safe_angles, self.wls, n_entry, d_list_full,
                self.materials['Air'], self.materials['Glass']
            )

            # The annual electrical calculation consumes only absorber-layer
            # currents.  Discard the other 11 layer arrays immediately after
            # the unchanged TMM solve, before temperature/AOI interpolation and
            # hourly broadcasting.  WBG/NBG are indices 3/8 in the full stack.
            # This preserves the exact absorber values while reducing the large
            # annual intermediate arrays from 13 layers to 2.
            lut = lut[:, (3, 8), :]
            
            theta_rad = np.radians(self.lut_angles)
            weights = np.sin(2 * theta_rad)
            norm = trapezoid(weights, theta_rad)
            diffuse_abs = trapezoid(lut * weights[:, None, None], theta_rad, axis=0) / norm
            
            lut_T_list.append(lut)
            diffuse_abs_T_list.append(diffuse_abs)
            
        lut_T = np.array(lut_T_list)          # (N_T, angles, layers, wls)
        diffuse_abs_T = np.array(diffuse_abs_T_list) # (N_T, layers, wls)

        thermal_lut_T = thermal_diffuse_T = None
        if 'thermal_irradiance' in env_data and 'thermal_wls' in env_data:
            thermal_lut_T, thermal_diffuse_T = self._thermal_absorptance_luts(
                env_data['thermal_wls'], T_grid,
                shifted_absorbers=shifted_absorbers_T)
        
        # ── 方案一：电热自洽迭代温度模型 ──
        # 当允许带隙温漂 (temp_flags[0] 为 True)，且传入了 T_air 和风速数据时，开启自洽迭代
        enable_iteration = temp_flags[0] and 't_air' in env_data and 'wind_speed' in env_data
        
        if initial_t_cell is not None:
            T_hourly = np.array(initial_t_cell, dtype=float).copy()
        else:
            T_hourly = env_data['t_cell'].copy()  # 初始化温度数组 (绝对温度，开尔文)
        max_iters = 50 if enable_iteration else 1
        tolerance = 0.05  # 温度收敛容差 (K)
        relaxation = 0.5
        final_evaluation = False
        converged = not enable_iteration
        convergence_iterations = 0
        
        total_energy = 0.0
        p_hourly = np.zeros(len(T_hourly))
        J_wbg = np.zeros(len(T_hourly))
        J_nbg = np.zeros(len(T_hourly))
        
        for iteration in range(max_iters + 1):
            T_hourly_prev = T_hourly.copy()
            
            # 1. 沿角度进行逐小时插值
            aoi_safe = np.clip(env_data['aoi'], 0, 90)
            step = 90.0 / (len(self.lut_angles) - 1)
            idx_float = aoi_safe / step
            idx_low = np.floor(idx_float).astype(int)
            idx_low = np.clip(idx_low, 0, len(self.lut_angles)-2)
            w_high = idx_float - idx_low
            w_low = 1.0 - w_high
            
            # 2. 沿温度轴进行逐小时线性插值，利用当前小时估算温度 T_hourly
            if len(T_grid) == 1:
                lut_low = lut_T[0, idx_low]
                lut_high = lut_T[0, idx_low + 1]
                diffuse_abs_hourly = np.repeat(diffuse_abs_T, len(T_hourly), axis=0)
            else:
                T_grid_np = np.array(T_grid)
                idx = np.searchsorted(T_grid_np, T_hourly, side='right') - 1
                idx = np.clip(idx, 0, len(T_grid_np) - 2)
                t0 = T_grid_np[idx]
                t1 = T_grid_np[idx + 1]
                w = (T_hourly - t0) / (t1 - t0 + 1e-12)
                
                # 直接抽取所需的 lut，避免生成完整的 lut_hourly (节省 ~4GB 内存)
                lut_T_idx_low = lut_T[idx, idx_low]
                lut_T_idx_next_low = lut_T[idx + 1, idx_low]
                lut_low = (1 - w)[:, None, None] * lut_T_idx_low + w[:, None, None] * lut_T_idx_next_low
                
                lut_T_idx_high = lut_T[idx, idx_low + 1]
                lut_T_idx_next_high = lut_T[idx + 1, idx_low + 1]
                lut_high = (1 - w)[:, None, None] * lut_T_idx_high + w[:, None, None] * lut_T_idx_next_high
                
                # 线性插值 diffuse_abs
                diffuse_abs_hourly = (1 - w)[:, None, None] * diffuse_abs_T[idx] + w[:, None, None] * diffuse_abs_T[idx + 1]
                
            abs_direct = w_low[:, None, None] * lut_low + w_high[:, None, None] * lut_high
            
            term_direct = abs_direct * env_data['direct_flux'][:, None, :]
            term_diffuse = diffuse_abs_hourly * env_data['diffuse_flux'][:, None, :]
            total_flux = term_direct + term_diffuse
            
            j_layers = q * trapezoid(total_flux, x=self.wls, axis=2) * 0.1 * (1.0 - config.SHADING_LOSS)
            J_wbg = j_layers[:, 0]
            J_nbg = j_layers[:, 1]
            
            # 3. 求解当前迭代下的逐小时最大电学输出功率 p_hourly
            p_hourly = calculate_hourly_power_numba(
                J_wbg, J_nbg, T_hourly,
                np.array(config.TOP_PARAMS), np.array(config.BOT_PARAMS),
                np.array(config.ALPHA_COEFFS),
                eg_top=self.current_bandgap_wbg, eg_bot=self.current_bandgap_nbg,
                rs_tj=config.RS_TJ,
                eg_temp_coeff_nbg=config.EG_TEMP_COEFF_NBG,
                temp_flags=temp_flags
            )
            total_energy = np.sum(p_hourly)

            if not enable_iteration:
                break
                
            # 4. 根据实时输出功率与输入辐照计算工作效率
            poa_hourly = np.asarray(env_data['poa'], dtype=float)
            if np.any(poa_hourly < 0.0):
                raise RuntimeError("POA irradiance cannot be negative")
            positive_poa = poa_hourly > 0.0
            eff_hourly = np.zeros_like(poa_hourly)
            # p_hourly and POA are both evaluated for every positive-irradiance
            # hour; do not replace sub-5 W/m2 irradiance with an artificial floor.
            eff_hourly[positive_poa] = (
                p_hourly[positive_poa] / (poa_hourly[positive_poa] * 1e-4))
            eff_hourly = np.clip(eff_hourly, 0.0, 0.40)  # 效率硬裁剪至合理区间 (0~40%)
            
            # 5. 计算既定裸器件薄膜堆叠的动态光热吸收率，并区分直射AOI与半球散射。
            if thermal_lut_T is not None:
                incident = env_data['thermal_irradiance']
                direct_fraction_all = env_data['thermal_direct_fraction']
                absorbed_power_hourly = np.empty(len(T_hourly), dtype=float)
                # Preserve the identical row-wise interpolation/integration
                # operations while bounding temporary arrays.  A full-year
                # 2002-wavelength thermal spectrum otherwise creates several
                # >70 MB arrays per iteration in each of four worker processes.
                thermal_chunk_hours = 256
                for start in range(0, len(T_hourly), thermal_chunk_hours):
                    stop = min(start + thermal_chunk_hours, len(T_hourly))
                    sl = slice(start, stop)
                    if len(T_grid) == 1:
                        thermal_direct = (
                            w_low[sl, None] * thermal_lut_T[0, idx_low[sl]]
                            + w_high[sl, None] * thermal_lut_T[0, idx_low[sl] + 1])
                        thermal_diffuse = np.repeat(
                            thermal_diffuse_T[0][None, :], stop - start, axis=0)
                    else:
                        td_low = (
                            (1 - w[sl])[:, None] * thermal_lut_T[idx[sl], idx_low[sl]]
                            + w[sl, None] * thermal_lut_T[idx[sl] + 1, idx_low[sl]])
                        td_high = (
                            (1 - w[sl])[:, None] * thermal_lut_T[idx[sl], idx_low[sl] + 1]
                            + w[sl, None] * thermal_lut_T[idx[sl] + 1, idx_low[sl] + 1])
                        thermal_direct = (
                            w_low[sl, None] * td_low + w_high[sl, None] * td_high)
                        thermal_diffuse = (
                            (1 - w[sl])[:, None] * thermal_diffuse_T[idx[sl]]
                            + w[sl, None] * thermal_diffuse_T[idx[sl] + 1])
                    direct_fraction = direct_fraction_all[sl, None]
                    absorbed_power_hourly[sl] = trapezoid(
                        incident[sl] * (
                            direct_fraction * thermal_direct
                            + (1.0 - direct_fraction) * thermal_diffuse),
                        x=env_data['thermal_wls'], axis=1)
                abs_factor = np.zeros_like(poa_hourly)
                abs_factor[positive_poa] = (
                    absorbed_power_hourly[positive_poa] / poa_hourly[positive_poa])
            else:
                photon_E = h * c / (self.wls * 1e-9)
                absorbed_flux = np.sum(total_flux, axis=1)
                absorbed_power_hourly = trapezoid(absorbed_flux * photon_E,
                                                   x=self.wls, axis=1)
                abs_factor = np.zeros_like(poa_hourly)
                abs_factor[positive_poa] = (
                    absorbed_power_hourly[positive_poa] / poa_hourly[positive_poa])

            eps = 1e-9
            if (not np.all(np.isfinite(abs_factor)) or
                    np.any(abs_factor < -eps) or np.any(abs_factor > 1.0 + eps)):
                raise RuntimeError(
                    f"Non-physical dynamic absorptance: min={np.nanmin(abs_factor)}, "
                    f"max={np.nanmax(abs_factor)}")
            abs_factor = np.clip(abs_factor, 0.0, 1.0)

            # The accepted temperature has now been used to evaluate optics,
            # electrical output, efficiency and absorptance consistently.
            if final_evaluation:
                break

            # 调用物理公式重新计算工作温度 (向量化极速处理)
            # abs_factor 现在可能是一个形如 (N_hours,) 的数组
            T_c_raw_K = Joule_Physics.calc_cell_temperature(
                env_data['t_air'], env_data['poa'], env_data['wind_speed'],
                absorption_factor=abs_factor, efficiency=eff_hourly,
                method='pvsyst', U0=config.THERMAL_U0, U1=config.THERMAL_U1
            ) + 273.15

            # Under-relaxation stabilizes hours near sharp optical transitions.
            # Convergence is tested on the unrelaxed fixed-point residual, so
            # damping cannot make an unconverged state appear converged.
            max_diff = np.max(np.abs(T_c_raw_K - T_hourly_prev))
            # 6. 若当前状态满足固定点容差则直接接受；否则才欠松弛更新，
            # 避免更新后的温度未经复核便被标记为收敛。
            convergence_iterations = iteration + 1
            if max_diff < tolerance:
                logger.info(f"Electro-thermal iteration converged at step {iteration + 1}. Max difference: {max_diff:.4f} K")
                converged = True
                final_evaluation = True
            elif convergence_iterations >= max_iters:
                worst = int(np.argmax(np.abs(T_c_raw_K - T_hourly_prev)))
                source_hour = int(env_data.get('indices', np.arange(len(T_hourly)))[worst])
                raise RuntimeError(
                    f"Electro-thermal iteration did not converge after {max_iters} updates; "
                    f"max temperature difference={max_diff:.6f} K, "
                    f"active_hour={worst}, source_row={source_hour}, "
                    f"Tcell={T_hourly_prev[worst] - 273.15:.6f} C, "
                    f"PVsyst_target={T_c_raw_K[worst] - 273.15:.6f} C, "
                    f"POA={env_data['poa'][worst]:.6f} W/m2, "
                    f"AOI={env_data['aoi'][worst]:.6f} deg")
            else:
                T_hourly = T_hourly_prev + relaxation * (T_c_raw_K - T_hourly_prev)

        # 缓存结果（含带隙维度），仅当 temp_flags 为全 True 且 city_name 有效时存入全局缓存
        if self.sim_cache and city_name and temp_flags == (True, True, True):
            self.sim_cache.set_simulation(
                city_name, self.current_thick_wbg, self.current_thick_nbg,
                total_energy,
                bandgap_wbg=self.current_bandgap_wbg,
                bandgap_nbg=self.current_bandgap_nbg
            )
        
        if return_details:
            return {
                'total_energy': total_energy,
                'J_wbg': J_wbg,
                'J_nbg': J_nbg,
                'p_hourly': p_hourly,
                't_cell_final': T_hourly - 273.15,
                'dynamic_absorptance': abs_factor if enable_iteration else np.full(len(T_hourly), np.nan),
                'converged': converged,
                'convergence_iterations': convergence_iterations,
                'convergence_max_diff_K': max_diff if enable_iteration else 0.0
            }

        if return_temperature:
            return total_energy, T_hourly

        return total_energy

    def calculate_total_absorption_factor(self, I_sun=None):
        """
        基于当前材料nk与层厚，计算既定裸器件薄膜堆叠（含有限Ag层寄生吸收）的光热吸收率。
        """
        # 1. 重新加载当前的堆栈定义
        n_stack, d_stack = self.reload_stack_definition()
        n_entry = np.vstack([self.materials['Glass'][None, :], n_stack,
                             self.materials['Air'][None, :]])
        d_list_full = np.concatenate(([0.0], d_stack, [0.0]))
        
        # 2. 计算法向入射 (AOI=0°) 下所有波长的 TMM 吸收和反射率
        num_wls = len(self.wls)
        theta_air = np.zeros(num_wls)
        theta_glass = np.zeros(num_wls)
        
        # 计算空气-玻璃界面反射率 R_ag
        n_air = self.materials['Air']
        n_glass = self.materials['Glass']
        rs = interface_r_vec_jit(0, n_air, n_glass, theta_air, theta_glass)
        R_ag = np.abs(rs)**2
        
        # 运行 TMM 得到 R_stack
        th_list = np.zeros((len(n_entry), num_wls), dtype=np.complex128) # 0° 下折射角全为0
        kz_list = 2 * np.pi * n_entry * np.cos(th_list) / self.wls
        delta = np.zeros((len(n_entry), num_wls), dtype=np.complex128)
        for i in range(1, len(n_entry)-1):
            delta[i] = kz_list[i] * d_list_full[i]
            
        r0 = interface_r_vec_jit(0, n_entry[0], n_entry[1], th_list[0], th_list[1])
        t0 = interface_t_vec_jit(0, n_entry[0], n_entry[1], th_list[0], th_list[1])
        
        M00 = 1.0/t0; M10 = r0/t0
        M01 = r0/t0; M11 = 1.0/t0
        
        M00_arr = M00 + 0j
        M10_arr = M10 + 0j
        M01_arr = M01 + 0j
        M11_arr = M11 + 0j
        
        for i in range(1, len(n_entry)-1):
            exp_mdp = np.exp(-1j * delta[i])
            exp_pdp = np.exp(1j * delta[i])
            M00_arr = M00_arr * exp_mdp; M10_arr = M10_arr * exp_mdp
            M01_arr = M01_arr * exp_pdp; M11_arr = M11_arr * exp_pdp
            
            r = interface_r_vec_jit(0, n_entry[i], n_entry[i+1], th_list[i], th_list[i+1])
            t = interface_t_vec_jit(0, n_entry[i], n_entry[i+1], th_list[i], th_list[i+1])
            i00 = 1.0/t; i01 = r/t; i10 = r/t; i11 = 1.0/t
            
            m00_new = M00_arr * i00 + M01_arr * i10
            m10_new = M10_arr * i00 + M11_arr * i10
            m01_new = M00_arr * i01 + M01_arr * i11
            m11_new = M10_arr * i01 + M11_arr * i11
            M00_arr, M01_arr, M10_arr, M11_arr = m00_new, m01_new, m10_new, m11_new
            
        r_total = M10_arr / M00_arr
        R_stack = np.abs(r_total)**2
        
        # 器件总反射率 R_total
        R_total = R_ag + ((1.0 - R_ag)**2 * R_stack) / (1.0 - R_ag * R_stack)
        
        # 器件总吸收率 A_total = 1 - R_total
        A_wls = 1.0 - R_total
        
        # 3. 光谱加权积分
        if I_sun is not None:
            # 此时 I_sun 为传入的真实光谱（可能是一维也可能是 [N_hours, num_wls] 的二维数组）
            # 使用 scipy.integrate.trapezoid 沿最后一个维度积分
            total_power = trapezoid(I_sun, x=self.wls, axis=-1)
            absorbed_power = trapezoid(A_wls * I_sun, x=self.wls, axis=-1)
            
            # 由于存在极弱光或无光时刻，需要避免除以 0
            if np.isscalar(total_power):
                return float(absorbed_power / total_power) if total_power > 0 else 0.943
            else:
                abs_factor = np.full_like(total_power, 0.943, dtype=float)
                mask = total_power > 1e-5
                abs_factor[mask] = absorbed_power[mask] / total_power[mask]
                return abs_factor
        elif hasattr(self, 'am15g_spectrum') and self.am15g_spectrum is not None:
            I_sun = self.am15g_spectrum
        else:
            h_const = 6.62607004e-34
            c_const = 299792458.0
            kB_const = 1.38064852e-23
            T_sun = 5778.0 # 太阳表面有效温度
            wl_m = self.wls * 1e-9
            I_sun = (2.0 * h_const * c_const**2 / wl_m**5) / (np.exp(h_const * c_const / (wl_m * kB_const * T_sun)) - 1.0)
        
        # 加权吸收率积分
        total_power = trapezoid(I_sun, x=self.wls)
        absorbed_power = trapezoid(A_wls * I_sun, x=self.wls)
        
        absorption_factor = absorbed_power / total_power if total_power > 0 else 0.943
        return float(absorption_factor)
