# multi_conductor_exporter.py
import csv
import numpy as np
from multi_conductor_calculator import MultiConductorCalculator

  
class MultiConductorExporter:
    def __init__(self, calculator: MultiConductorCalculator):
        self.calculator = calculator
   
    def export_capacitance_matrix(self, filename: str, unit_prefix: str = None, length: float = None) -> None:
        self._export_matrix(
            matrix=self.calculator.calculate_capacitance_matrix(),
            filename=filename,
            unit_prefix=unit_prefix,
            length=length,
            matrix_type='Capacitance',
            unit_base='F'
        )
    
    def export_inductance_matrix(self, filename: str, unit_prefix: str = None, length: float = None) -> None:
        self._export_matrix(
            matrix=self.calculator.calculate_inductance_matrix(),
            filename=filename,
            unit_prefix=unit_prefix,
            length=length,
            matrix_type='Inductance',
            unit_base='H'
        )
    
    def export_characteristic_impedance_matrix(self, filename: str, unit_prefix: str = None) -> None:
        self._export_matrix(
            matrix=self.calculator.calculate_characteristic_impedance_matrix(),
            filename=filename,
            unit_prefix=unit_prefix,
            length=None,
            matrix_type='Characteristic Impedance',
            unit_base='Ω'
        )
    
    def _export_matrix(self, matrix: np.ndarray, filename: str, unit_prefix: str, length: float, matrix_type: str, unit_base: str) -> None:
        unit_factors = {
            'm': 1e3,    # milli
            'u': 1e6,    # micro
            'n': 1e9,    # nano
            'p': 1e12,   # pico
            'f': 1e15,   # femto
            'a': 1e18    # atto
        }
        
        unit_strings = {
            'm': f'm{unit_base}',
            'u': f'μ{unit_base}',
            'n': f'n{unit_base}',
            'p': f'p{unit_base}',
            'f': f'f{unit_base}',
            'a': f'a{unit_base}',
            None: unit_base
        }
        
        # 長さを考慮（L行列やC行列の場合）
        if length is not None:
            matrix = matrix * length
        
        # 単位変換を適用
        if unit_prefix in unit_factors:
            matrix = matrix * unit_factors[unit_prefix]
        
        # ヘッダー行の作成
        n_conductors = len(self.calculator.conductors)
        header = ['Conductor']
        for i in range(n_conductors):
            unit_label = unit_strings[unit_prefix]
            header.append(f'{matrix_type}{i+1} [{unit_label}]')

        # CSVファイルに書き出し
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            # 各行のデータを書き出し
            for i in range(n_conductors):
                row = [f'Conductor {i+1}']
                row.extend([f'{val:.6g}' for val in matrix[i]])
                writer.writerow(row)
        
        print(f"{matrix_type} matrix has been saved to {filename}")
