# -*- coding: utf-8 -*-
"""
日志管理模块，提供普通控制台/文件日志和结构化性能/仿真日志记录
"""

import os
import json
import logging

def get_logger():
    """获取标准 Logger 实例，支持输出到终端和 logs/simulation.log 文件"""
    logger = logging.getLogger("simulation_engine")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 控制台输出
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        
        # 确保 logs 目录存在并输出到文件
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler("logs/simulation.log", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

class StructuredLogger:
    """结构化日志记录器，记录仿真数据和性能指标"""
    
    def __init__(self):
        self.logger = get_logger()
        os.makedirs("logs", exist_ok=True)
        
    def log_simulation_result(self, city, wbg_thick, nbg_thick, yield_kwh, eff, total_insol=0.0, scan_points=1):
        """记录仿真计算结果，写入 JSONL 格式的日志文件"""
        data = {
            "city": city,
            "wbg_thick": float(wbg_thick),
            "nbg_thick": float(nbg_thick),
            "yield_kwh_m2": float(yield_kwh),
            "eff_pct": float(eff),
            "total_insol": float(total_insol),
            "scan_points": int(scan_points)
        }
        
        os.makedirs("logs", exist_ok=True)
        try:
            with open("logs/simulation_results.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.error(f"写入结构化仿真日志失败: {e}")
            
    def log_performance(self, metric_name, value, **kwargs):
        """记录系统运行性能指标，写入 JSONL 格式的日志文件"""
        data = {
            "metric": metric_name,
            "value": float(value),
            **kwargs
        }
        
        os.makedirs("logs", exist_ok=True)
        try:
            with open("logs/performance.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.error(f"写入结构化性能日志失败: {e}")

_structured_logger = StructuredLogger()

def get_structured_logger():
    """获取结构化 Logger 实例"""
    return _structured_logger
