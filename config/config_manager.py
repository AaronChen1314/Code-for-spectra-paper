# -*- coding: utf-8 -*-
"""
配置管理模块
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any

def generate_adaptive_wavelength_grid() -> List[float]:
    wls = []
    # 300 - 600 nm 步长 10
    wls.extend(range(300, 600, 10))
    # 600 - 780 nm 步长 2
    wls.extend(range(600, 780, 2))
    # 780 - 1020 nm 步长 2
    wls.extend(range(780, 1020, 2))
    # 1020 - 1100 nm 步长 10
    wls.extend(range(1020, 1101, 10))
    return [float(x) for x in wls]

@dataclass
class SimulationConfig:
    """仿真配置类"""
    
    # 材料参数
    DEFAULT_THICKNESS_WBG: float = 450.0
    DEFAULT_THICKNESS_NBG: float = 1200.0
    REAL_EG_TOP: float = 1.80
    REAL_EG_BOT: float = 1.25
    NK_SOURCE_EG_TOP: float = 1.80
    NK_SOURCE_EG_BOT: float = 1.25
    
    # 光学参数
    IQE_TOP: float = 1.00
    IQE_BOT: float = 1.00
    SHADING_LOSS: float = 0.00
    SIM_WLS: List[float] = field(default_factory=generate_adaptive_wavelength_grid)
    
    # 电学参数
    TOP_PARAMS: List[float] = field(default_factory=lambda: [2.8884e-22, 1.6934e-14, 1.1605, 1.9316, 2.2566, 9.2477e+03])
    BOT_PARAMS: List[float] = field(default_factory=lambda: [9.6741e-15, 1.7069e-12, 1.2555, 1.5381, 0.99859, 23524.0])
    RS_TJ: float = 0.0
    # ALPHA_COEFFS 说明:
    #   [0] = A1: Pmax 整机功率温度系数 (K^-1)，典型钙钛矿: -0.001~-0.003，负值表示升温降功率
    #         该项补偿本征物理模型未完全覆盖的 FF 额外热衰减
    #   [1] = A2: J01 功率律温度指数，物理值约 3.0（对应 ni² ∝ T³）
    #   [2] = A3: J02 功率律温度指数，与 SI 采用的二极管支路温度指数 β2=2.5 一致
    #         A2=β1=3.0，A3=β2=2.5
    #   [3] = A4: Rs 温度指数
    #   [4] = A5: Rsh 温度指数
    ALPHA_COEFFS: List[float] = field(default_factory=lambda: [0.0, 3.0, 2.5, 1.5, -2.0])
    
    # 带隙温度系数 —— 线性模型
    # WBG (宽带隙钙钛矿): 使用线性模型，physics_engine.py中系数为+0.31 meV/K
    # NBG (窄带隙 Sn-Pb 混合钙钛矿): 使用线性模型
    #   Eg(T) = Eg(T_STC) + EG_TEMP_COEFF_NBG × (T - T_STC)
    #   dEg/dT = +0.72 meV/K (正温度系数，升温带隙增大)
    EG_TEMP_COEFF_NBG: float = 0.72e-3  # NBG 带隙线性温度系数 (eV/K)

    # PVsyst 电热耦合模型的热损失参数（当前全年逐时仿真的实际调用值）
    THERMAL_U0: float = 14.58  # W m^-2 K^-1
    THERMAL_U1: float = 4.37  # W s m^-3 K^-1
    
    # Jsc温度依赖由温度相关nk与TMM重新积分得到，不使用额外经验灵敏度参数。
    # 扫描配置
    SCAN_ENABLE: bool = True
    SCAN_WBG_RANGE: List[float] = field(default_factory=lambda: list(range(200, 501, 10)))
    SCAN_NBG_RANGE: List[float] = field(default_factory=lambda: list(range(800, 1201, 10)))
    
    # 带隙扫描配置
    SCAN_BANDGAP_ENABLE: bool = False  # 是否启用带隙扫描
    SCAN_BANDGAP_WBG_RANGE: List[float] = field(default_factory=lambda: [round(x * 0.001, 3) for x in range(1700, 2001)])  # 宽带隙范围 1.70-2.00 eV, 步长0.001
    FIXED_THICKNESS_WBG_FOR_BG_SCAN: float = 450.0  # 标准光谱电流匹配得到的固定WBG厚度
    FIXED_THICKNESS_NBG_FOR_BG_SCAN: float = 1200.0  # 用户指定的固定NBG厚度
    
    # 并行配置
    MAX_WORKERS: int = 1
    ENABLE_NESTED_PROGRESS: bool = True
    
    # 缓存配置
    ENABLE_CACHE: bool = True
    CACHE_DIR: str = 'cache'
    CACHE_MAX_SIZE_GB: float = 10.0
    
    # I/O配置
    ENABLE_ASYNC_IO: bool = True
    BATCH_IO_SIZE: int = 1000
    
    # 日志配置
    LOG_LEVEL: str = 'INFO'
    ENABLE_STRUCTURED_LOGGING: bool = True
    
    def from_dict(self, config_dict: Dict[str, Any]):
        """从字典更新配置"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            key: getattr(self, key)
            for key in dir(self)
            if not key.startswith('_') and not callable(getattr(self, key))
        }

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file='config/config.yaml'):
        """初始化配置管理器"""
        self.config_file = config_file
        self.config = SimulationConfig()
        self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
                    if config_dict:
                        self.config.from_dict(config_dict)
            except Exception as e:
                print(f"Error loading config: {e}")
    
    def save_config(self):
        """保存配置文件"""
        config_dir = os.path.dirname(self.config_file)
        os.makedirs(config_dir, exist_ok=True)
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config.to_dict(), f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get_config(self) -> SimulationConfig:
        """获取配置实例"""
        return self.config
    
    def update_config(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def validate_config(self):
        """验证配置有效性"""
        # 验证厚度范围
        if self.config.DEFAULT_THICKNESS_WBG <= 0:
            self.config.DEFAULT_THICKNESS_WBG = 450.0
        
        if self.config.DEFAULT_THICKNESS_NBG <= 0:
            self.config.DEFAULT_THICKNESS_NBG = 1200.0
        
        # 验证扫描范围
        if not self.config.SCAN_WBG_RANGE:
            self.config.SCAN_WBG_RANGE = list(range(200, 501, 50))
        
        if not self.config.SCAN_NBG_RANGE:
            self.config.SCAN_NBG_RANGE = list(range(800, 1201, 50))
        
        # 验证工作线程数
        import multiprocessing
        max_possible = multiprocessing.cpu_count() * 2
        if self.config.MAX_WORKERS > max_possible:
            self.config.MAX_WORKERS = max_possible

# 全局配置实例
config_manager = ConfigManager()

def get_config() -> SimulationConfig:
    """获取全局配置实例"""
    return config_manager.get_config()

def update_config(**kwargs):
    """更新全局配置"""
    config_manager.update_config(**kwargs)


def print_physics_parameters():
    """
    仿真前物理参数全量打印函数。
    在每次仿真启动前调用，详细罗列所有底层物理参数及其物理意义，
    防止"黑盒计算"。
    """
    cfg = get_config()

    # 物理常数
    q   = 1.60217662e-19   # 元电荷 (C)
    k_B = 1.38064852e-23   # 玻尔兹曼常数 (J/K)
    T_STC = 298.15         # 标准测试条件温度 (K)

    sep  = '═' * 76
    sep2 = '─' * 76

    print(f"\n{sep}")
    print(f"   ⚙️  仿真底层物理参数一览（仿真前自检）")
    print(f"{sep}")

    # ── 基本物理常数 ──
    print(f"\n  ▸ 基本物理常数")
    print(f"  {sep2}")
    print(f"    元电荷 q           = {q:.6e} C")
    print(f"    玻尔兹曼常数 k_B   = {k_B:.6e} J/K")
    print(f"    标准温度 T_STC     = {T_STC} K  ({T_STC - 273.15:.1f} °C)")

    # ── 材料与器件结构参数 ──
    print(f"\n  ▸ 材料与器件结构参数")
    print(f"  {sep2}")
    print(f"    WBG 吸收层厚度     = {cfg.DEFAULT_THICKNESS_WBG} nm")
    print(f"    NBG 吸收层厚度     = {cfg.DEFAULT_THICKNESS_NBG} nm")
    print(f"    WBG 实际带隙       = {cfg.REAL_EG_TOP} eV    (宽带隙钙钛矿)")
    print(f"    NBG 实际带隙       = {cfg.REAL_EG_BOT} eV    (窄带隙 Sn-Pb 钙钛矿)")
    print(f"    WBG NK源文件带隙   = {cfg.NK_SOURCE_EG_TOP} eV")
    print(f"    NBG NK源文件带隙   = {cfg.NK_SOURCE_EG_BOT} eV")

    # ── 光学参数 ──
    print(f"\n  ▸ 光学参数")
    print(f"  {sep2}")
    print(f"    IQE_TOP (顶电池内量子效率)   = {cfg.IQE_TOP}")
    print(f"    IQE_BOT (底电池内量子效率)   = {cfg.IQE_BOT}")
    print(f"    遮蔽损失 SHADING_LOSS        = {cfg.SHADING_LOSS}")

    # ── 带隙温度系数 ──
    print(f"\n  ▸ 带隙温度系数")
    print(f"  {sep2}")
    print(f"    WBG (宽带隙, 正温度系数 — 升温带隙轻微增大):")
    print(f"      模型: 线性温度系数 (定义于 physics_engine.py)")
    print(f"      dEg/dT = +0.31 meV/K")
    print(f"    NBG (窄带隙 Sn-Pb, 正温度系数 — 升温带隙增大):")
    print(f"      模型: 线性 Eg(T) = Eg(T₀) + coeff × (T - T₀)")
    print(f"      coeff = {cfg.EG_TEMP_COEFF_NBG:.4e} eV/K  (+{cfg.EG_TEMP_COEFF_NBG*1e3:.2f} meV/K)")
    
    # ── 光生电流（Jsc）温度耦合机制 ──
    print(f"\n  ▸ 光学电流温度耦合机制")
    print(f"  {sep2}")
    print(f"    模型: 动态 TMM 光学传输矩阵自洽求解")
    print(f"    原理: 带隙随温度漂移 → n(λ,T)/k(λ,T) 实时移动 → 重构各温度层级的光学矩阵 (LUT-T)")
    print(f"          → 消除经验参数，直接精准还原 Jsc 的物理温变行为")

    # ── 电学参数 ──
    top_p = cfg.TOP_PARAMS
    bot_p = cfg.BOT_PARAMS
    print(f"\n  ▸ 电学参数 (二极管等效电路模型)")
    print(f"  {sep2}")
    print(f"    ┌────────────────┬──────────────────┬──────────────────┐")
    print(f"    │ 参数           │ 顶电池 (WBG)     │ 底电池 (NBG)     │")
    print(f"    ├────────────────┼──────────────────┼──────────────────┤")
    print(f"    │ J01 (A/cm²)    │ {top_p[0]:<16.4e} │ {bot_p[0]:<16.4e} │  暗饱和电流(扩散)")
    print(f"    │ J02 (A/cm²)    │ {top_p[1]:<16.4e} │ {bot_p[1]:<16.4e} │  暗饱和电流(复合)")
    print(f"    │ n1  (理想因子)  │ {top_p[2]:<16.4f} │ {bot_p[2]:<16.4f} │  一阶理想因子")
    print(f"    │ n2  (理想因子)  │ {top_p[3]:<16.4f} │ {bot_p[3]:<16.4f} │  二阶理想因子")
    print(f"    │ Rs  (Ω·cm²)    │ {top_p[4]:<16.4f} │ {bot_p[4]:<16.4f} │  串联电阻")
    print(f"    │ Rsh (Ω·cm²)    │ {top_p[5]:<16.4e} │ {bot_p[5]:<16.4e} │  并联电阻")
    print(f"    └────────────────┴──────────────────┴──────────────────┘")
    print(f"    Rs_TJ (隧穿结串阻) = {cfg.RS_TJ} Ω·cm²")

    # ── 温度系数 ALPHA_COEFFS ──
    ac = cfg.ALPHA_COEFFS
    print(f"\n  ▸ 温度系数 ALPHA_COEFFS")
    print(f"  {sep2}")
    print(f"    A1 (Pmax 功率温度系数)  = {ac[0]:+.6f} K⁻¹   {'← 升温降功率' if ac[0] < 0 else '← 当前为零/正'}")
    print(f"    A2 (J01 温度指数)       = {ac[1]:.4f}         ← J01 ∝ T^(A2/n1)")
    print(f"    A3 (J02 温度指数)       = {ac[2]:.4f}         ← J02 ∝ T^(A3/n2)")
    print(f"    A4 (Rs 温度指数)        = {ac[3]:.4f}         ← Rs ∝ T^A4")
    print(f"    A5 (Rsh 温度指数)       = {ac[4]:.4f}         ← Rsh ∝ T^A5")

    print(f"\n{sep}\n")
