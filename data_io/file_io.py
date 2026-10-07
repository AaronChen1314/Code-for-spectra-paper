# -*- coding: utf-8 -*-
"""
I/O操作模块
"""

import os
import pandas as pd
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor

class FileIO:
    @staticmethod
    def read_csv_async(filepath, **kwargs):
        """异步读取CSV文件"""
        return pd.read_csv(filepath, **kwargs)
    
    @staticmethod
    def write_csv_async(filepath, data, **kwargs):
        """异步写入CSV文件"""
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, **kwargs)
        else:
            pd.DataFrame(data).to_csv(filepath, **kwargs)
    
    @staticmethod
    def read_binary(filepath):
        """读取二进制文件"""
        with open(filepath, 'rb') as f:
            return np.load(f)
    
    @staticmethod
    def write_binary(filepath, data):
        """写入二进制文件"""
        with open(filepath, 'wb') as f:
            np.save(f, data)
    
    @staticmethod
    def load_nrel_data_robust(filepath):
        """鲁棒加载NREL气象数据"""
        try:
            df = pd.read_csv(filepath, header=0, skiprows=2, na_values='.')
            spec_cols = []; wls_vals = []; col_map = {} 
            for c in df.columns:
                cs = str(c).strip(); lower_c = cs.lower()
                if 'temperature' in lower_c: col_map['T_air'] = c
                elif 'wind speed' in lower_c: col_map['Wind'] = c
                elif 'zenith' in lower_c and 'solar' in lower_c: col_map['Zenith'] = c
                elif 'albedo' in lower_c: col_map['Albedo'] = c
                # 修复: 精确匹配太阳方位角 (Solar Azimuth Angle) 与面板朝向方位角 (Panel Azimuth Angle)
                elif 'solar azimuth' in lower_c: col_map['Solar_Azimuth'] = c
                elif 'panel azimuth' in lower_c: col_map['Panel_Azimuth'] = c
                elif 'dni' in lower_c or 'direct normal' in lower_c: col_map['DNI'] = c
                elif 'dhi' in lower_c or 'diffuse horizontal' in lower_c: col_map['DHI'] = c
                elif lower_c == 'ghi' or 'global horizontal' in lower_c: col_map['GHI'] = c
                elif 'panel tilt' in lower_c: col_map['Panel_Tilt'] = c
                if re.match(r'^[\d\.]+(\s*um)?$', cs) or ('um' in cs and re.search(r'[\d\.]+', cs)):
                    try:
                        v_str = re.findall(r'[\d\.]+', cs)[0]; v = float(v_str)
                        if v < 100: v *= 1000.0
                        if 250 <= v <= 4500: spec_cols.append(c); wls_vals.append(v)
                    except: pass
            
            if len(df.columns) > 31:
                if 'DHI' not in col_map: col_map['DHI'] = df.columns[11]           
                if 'DNI' not in col_map: col_map['DNI'] = df.columns[12]           
                if 'Zenith' not in col_map: col_map['Zenith'] = df.columns[8] 
            sorted_indices = np.argsort(wls_vals)
            sorted_wls = np.array(wls_vals)[sorted_indices]
            sorted_cols = [spec_cols[i] for i in sorted_indices]
            spectra_matrix = df[sorted_cols].fillna(0.0).values 
            spectra_matrix = np.maximum(0.0, spectra_matrix)
            first_wl_raw = float(re.findall(r'[\d\.]+', str(sorted_cols[0]))[0])
            if first_wl_raw < 100: spectra_matrix *= 0.001 
            return df, sorted_wls, spectra_matrix, col_map
        except Exception as e:
            print(f"Error loading NREL data: {e}")
            return None, None, None, None
    
    @staticmethod
    def batch_read_files(filepaths, reader_func, max_workers=4):
        """批量读取文件"""
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(reader_func, fp): fp for fp in filepaths}
            for future in future_to_file:
                filepath = future_to_file[future]
                try:
                    results[filepath] = future.result()
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    results[filepath] = None
        return results
    
    @staticmethod
    def batch_write_files(filepath_data_pairs, writer_func, max_workers=4):
        """批量写入文件"""
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(writer_func, fp, data): fp for fp, data in filepath_data_pairs.items()}
            for future in future_to_file:
                filepath = future_to_file[future]
                try:
                    results[filepath] = future.result()
                except Exception as e:
                    print(f"Error writing {filepath}: {e}")
                    results[filepath] = False
        return results
