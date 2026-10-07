# -*- coding: utf-8 -*-
"""
物理辅助工具模块，包含组件温度模型、Perez天空散射模型、以及光谱数据解析与带隙平移算法。
"""

import numpy as np
from scipy.interpolate import interp1d

class Joule_Physics:
    """光伏组件热物理计算类"""
    
    @staticmethod
    def calc_cell_temperature(T_air, P_in_POA_W_m2, wind_speed, absorption_factor=0.943, efficiency=0.20, method='pvsyst', **kwargs):
        """
        计算光伏电池的实际表面工作温度。支持热平衡模型（考虑转换效率电能抵消）、PVsyst模型与经典的 Faiman 半经验模型。
        
        参数:
            T_air: 环境空气温度 (°C)
            P_in_POA_W_m2: 倾斜面上的总入射太阳辐射强度 (W/m²)
            wind_speed: 风速 (m/s)
            absorption_factor: 组件的总光热吸收率 (无量纲)
            efficiency: 组件的工作转换效率 (无量纲)
            method: 温度模型选择, 'heat_balance', 'pvsyst' 或 'faiman'
            **kwargs: 其他模型参数，如：
                      U0: PVsyst 模型常数热损失项
                      U1: PVsyst 模型风速相关热损失项
                      U_L0: Faiman 模型常数热损失项 (默认 37.93 W/m²K)
                      U_L1: Faiman 模型风速冷却系数 (默认 10.47 W/s m³K)
            
        返回:
            T_c_celsius: 电池组件的估算工作温度 (°C)
        """
        # 风速不能为负数 (使用 np.maximum 以完美支持向量化数组计算)
        w_speed = np.maximum(0.0, wind_speed)
        
        if method == 'pvsyst':
            # PVsyst 模型 (考虑了吸收率和效率)
            U0 = kwargs.get('U0', 14.58)
            U1 = kwargs.get('U1', 4.37)
            eff = np.clip(efficiency, 0.0, 0.40)
            T_c_celsius = T_air + (P_in_POA_W_m2 * absorption_factor * (1.0 - eff)) / (U0 + U1 * w_speed)
            return T_c_celsius
            
        elif method == 'faiman':
            # Faiman 半经验模型
            U_L0 = kwargs.get('U_L0', 37.93)
            U_L1 = kwargs.get('U_L1', 10.47)
            T_c_celsius = T_air + P_in_POA_W_m2 / (U_L0 + U_L1 * w_speed)
            return T_c_celsius
            
        elif method == 'heat_balance':
            # 基于能量守恒的物理热平衡模型 (学术推荐，考虑了光电转换输出对产热的扣减)
            u_c = kwargs.get('u_c', 4.3)
            u_v = kwargs.get('u_v', 1.528)
            
            # 1. 计算对流换热系数 h_c (Faiman 模型的经验风速关系式)
            h_c = u_c + u_v * w_speed
            
            # 2. 计算辐射换热系数 h_r (常温下对绝对温度辐射换热项 of 线性化近似)
            h_r = 0.028 * (T_air + 273.15)
            
            # 3. 总散热系数 U = 对流项 + 辐射项
            U = h_c + h_r
            
            # 4. 考虑组件的总光热吸收率与当前最大功率输出效率 (扣除电能对发热的抵消，使用 np.clip 以支持向量化)
            eff = np.clip(efficiency, 0.0, 0.40)
            T_c_celsius = T_air + (P_in_POA_W_m2 * (absorption_factor - eff)) / U
            return T_c_celsius
            
        else:
            raise ValueError(f"Unknown temperature calculation method: {method}. Use 'pvsyst', 'faiman' or 'heat_balance'.")


class Perez_Sky_Model:
    """Perez 天空各向异性散射模型类"""
    
    # Perez 晴朗度区间与对应的各向异性参数表
    # 每行格式: [晴朗度上限, f11, f12, f13, f21, f22, f23]
    PEREZ_COEFFS = np.array([
        [1.065, -0.008, 0.588, -0.062, -0.060, 0.072, -0.022],
        [1.230,  0.130, 0.683, -0.151, -0.019, 0.066, -0.029],
        [1.500,  0.330, 0.487, -0.221,  0.055, -0.066, -0.026],
        [1.950,  0.568, 0.187, -0.295,  0.109, -0.152, -0.014],
        [2.800,  0.873, -0.392, -0.362, 0.226, -0.462,  0.001],
        [4.500,  1.132, -1.237, -0.412, 0.288, -0.823,  0.056],
        [6.200,  1.060, -1.600, -0.359, 0.264, -1.127,  0.131],
        [999.9,  0.678, -0.327, -0.250, 0.156, -1.377,  0.251]
    ])
    
    @staticmethod
    def get_air_mass(zenith_deg):
        """
        利用经典的 Kasten-Young 经验模型计算大气质量 (Air Mass, AM)。
        
        参数:
            zenith_deg: 太阳天顶角 (度)
            
        返回:
            am: 修正后的大气质量
        """
        # 限制天顶角在合理物理范围内，防止数学计算溢出
        z_safe = np.clip(zenith_deg, 0.0, 89.9)
        cos_z = np.cos(np.radians(z_safe))
        
        # 大气质量公式
        am = 1.0 / (cos_z + 0.50572 * np.power(96.07995 - z_safe, -1.6364))
        return am
        
    @classmethod
    def calc_perez_factors(cls, DNI, DHI, zenith_deg, AM, dni_extra=1367.0):
        """
        根据 Perez 天空散射辐射模型，计算天空各向异性指数 F1 和 F2。
        
        参数:
            DNI: 太阳法向直射辐射 (W/m²)
            DHI: 水平散射辐射 (W/m²)
            zenith_deg: 太阳天顶角 (度)
            AM: 大气质量
            
        返回:
            F1: 环日各向异性指数
            F2: 地平线增亮指数
        """
        z_rad = np.radians(zenith_deg)
        term = 1.041 * (z_rad ** 3)
        
        # 防止除以 0
        dhi_safe = max(0.0001, DHI)
        
        # 1. 计算天空晴朗度 epsilon
        epsilon = ((DHI + DNI) / dhi_safe + term) / (1.0 + term)
        
        # 2. 计算天空亮度 delta。逐小时调用方应传入日地距离修正后的地外法向辐照度。
        dni_extra_safe = max(1.0, float(dni_extra))
        delta = DHI * AM / dni_extra_safe
        
        # 3. 确定晴朗度 epsilon 所在的 Perez 区间
        idx = 7
        for i, coeff in enumerate(cls.PEREZ_COEFFS):
            if epsilon <= coeff[0]:
                idx = i
                break
                
        c = cls.PEREZ_COEFFS[idx]
        
        # 4. 根据查表系数计算 F1 和 F2 因子
        F1 = max(0.0, c[1] + c[2] * delta + c[3] * z_rad)
        F2 = c[4] + c[5] * delta + c[6] * z_rad
        
        return F1, F2


class Model_Utils:
    """光伏仿真模型的光学常数及带隙计算辅助类"""
    
    @staticmethod
    def parse_nk_data(content):
        """
        解析波长相关的折射率 n 和消光系数 k 的文本数据。
        
        参数:
            content: 文件路径或类文件对象
            
        返回:
            wls: 波长数组 (nm)
            n: 折射率实部数组
            k: 消光系数(折射率虚部)数组
        """
        try:
            # 1. 尝试以逗号分隔格式读取，并跳过第一行表头
            data = np.genfromtxt(content, delimiter=',', skip_header=1)
            
            # 2. 若数据维度不符，尝试以空白字符分隔读取（含跳过表头）
            if data is None or data.ndim != 2 or data.shape[1] < 3:
                data = np.genfromtxt(content, skip_header=1)
                
            # 3. 若依然不符，尝试从第 0 行开始解析
            if data is None or data.ndim != 2 or data.shape[1] < 3:
                data = np.genfromtxt(content, skip_header=0)
                
            if data is None or data.ndim != 2 or data.shape[1] < 3:
                return None, None, None
                
            wls = data[:, 0]
            n = data[:, 1]
            k = data[:, 2]
            
            # 4. 确保波长按升序排列
            idx = np.argsort(wls)
            wls = wls[idx]
            n = n[idx]
            k = k[idx]
            
            return wls, n, k
        except Exception:
            return None, None, None
            
    @staticmethod
    def shift_nk_by_bandgap(wls_nm, n_old, k_old, old_eg, new_eg):
        """
        根据钙钛矿半导体的带隙调整，在光子能量尺度 (eV) 上对消光系数 k 进行光谱平移。
        
        参数:
            wls_nm: 光谱波长数组 (nm)
            n_old: 原始折射率 n 数组
            k_old: 原始消光系数 k 数组
            old_eg: 原带隙能量值 (eV)
            new_eg: 新带隙能量值 (eV)
            
        返回:
            n_new: 调整后的折射率数组 (通常保持实部不变)
            k_new: 调整后的消光系数数组
        """
        if abs(new_eg - old_eg) < 0.0001:
            return n_old, k_old
            
        # 1. 将光谱波长 (nm) 转换为光子能量 (eV) (hc ≈ 1240.0 eV·nm)
        energy_ev = 1240.0 / wls_nm
        
        # 2. 升序排列光子能量以便进行一维插值
        idx = np.argsort(energy_ev)
        E_sorted = energy_ev[idx]
        k_sorted = k_old[idx]
        
        # 3. 在能量空间进行平移：E_shifted = E - (Eg_new - Eg_old)
        delta_eg = new_eg - old_eg
        E_shifted = energy_ev - delta_eg
        
        # 4. 插值生成新的消光系数 k (越界处 k 的物理填充值设为 0.0)
        f_k = interp1d(E_sorted, k_sorted, bounds_error=False, fill_value=0.0)
        k_new = f_k(E_shifted)
        
        # 确保消光系数不小于 0
        k_new = np.maximum(0.0, k_new)
        
        # 计算消光系数的变化量 delta_k
        delta_k = k_new - k_old
        
        # 将数据按能量升序排列以进行 KK 积分
        idx_sort = np.argsort(energy_ev)
        E_sort = energy_ev[idx_sort]
        dk_sort = delta_k[idx_sort]
        dn_sort = np.zeros_like(E_sort)
        
        # 采用离散 Kramers-Kronig 积分求解折射率的变化量 delta_n
        # \Delta n(E) = (2/\pi) * \mathcal{P} \int \frac{E' \Delta k(E') - E \Delta k(E)}{E'^2 - E^2} dE'
        for i, e_val in enumerate(E_sort):
            numerator = E_sort * dk_sort - e_val * dk_sort[i]
            denominator = E_sort**2 - e_val**2
            
            integrand = np.zeros_like(E_sort)
            mask = np.abs(E_sort - e_val) > 1e-6
            integrand[mask] = numerator[mask] / denominator[mask]
            
            # 洛必达法则处理奇异点 (E' -> E)
            if i > 0 and i < len(E_sort) - 1:
                dk_prime = (dk_sort[i+1] - dk_sort[i-1]) / (E_sort[i+1] - E_sort[i-1])
            elif i == 0:
                dk_prime = (dk_sort[1] - dk_sort[0]) / (E_sort[1] - E_sort[0])
            else:
                dk_prime = (dk_sort[-1] - dk_sort[-2]) / (E_sort[-1] - E_sort[-2])
                
            integrand[~mask] = (dk_sort[i] + e_val * dk_prime) / (2 * e_val) if e_val > 1e-6 else 0.0
            
            # 使用 np.trapezoid 避免 np.trapz 在新版 numpy 中被废弃的问题
            dn_sort[i] = (2.0 / np.pi) * np.trapezoid(integrand, E_sort)
            
        # 将计算得到的 dn_sort 恢复到原始未排序的波长/能量序列
        delta_n = np.zeros_like(energy_ev)
        delta_n[idx_sort] = dn_sort
        
        # 修正原折射率 n
        n_new = n_old + delta_n
        
        # 增加安全限幅截断，确保折射率始终在物理合理范围 [1.0, 5.0] 内
        n_new = np.clip(n_new, 1.0, 5.0)
        
        return n_new, k_new
