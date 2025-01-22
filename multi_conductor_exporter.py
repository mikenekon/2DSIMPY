# multi_conductor_exporter.py
import csv
import numpy as np
from multi_conductor_calculator import MultiConductorCalculator

class MultiConductorExporter:
    def __init__(self, calculator: MultiConductorCalculator):
        self.calculator = calculator
   
    def export_capacitance_matrix(self, filename: str, unit_prefix: str = None, length: float = None) -> None:
        """
        容量行列をCSVファイルに出力する
        
        Parameters:
        -----------
        filename : str
            出力するCSVファイルの名前
        unit_prefix : str, optional
            容量値の補助単位 ('m', 'u', 'n', 'p', 'a')
            指定しない場合は補助単位なし
        length : float, optional
            長さの指定（メートル単位）。指定しない場合は単位長さ（1m）あたりの容量
        """
        # 単位変換の係数を定義
        unit_factors = {
            'm': 1e3,    # milli
            'u': 1e6,    # micro
            'n': 1e9,    # nano
            'p': 1e12,   # pico
            'f': 1e15,   # femto
            'a': 1e18    # atto
        }
        
        # 単位プレフィックスの文字列を定義
        unit_strings = {
            'm': 'mF',
            'u': 'μF',
            'n': 'nF',
            'p': 'pF',
            'f': 'fF',
            'a': 'aF',
            None: 'F'
        }
        
        # 容量行列を計算
        C = self.calculator.calculate_capacitance_matrix()
        
        # 長さを掛ける
        if length is not None:
            C = C * length
        
        # 単位変換を適用
        if unit_prefix in unit_factors:
            C = C * unit_factors[unit_prefix]
        
        # ヘッダー行の作成
        n_conductors = len(self.calculator.conductors)
        header = ['Conductor']
        for i in range(n_conductors):
            header.append(f'C{i+1} [{unit_strings[unit_prefix]}/m]' if length is None else f'C{i+1} [{unit_strings[unit_prefix]}/{length}]')

        # CSVファイルに書き出し
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            # 各行のデータを書き出し
            for i in range(n_conductors):
                row = [f'Conductor {i+1}']
                row.extend([f'{val:.6g}' for val in C[i]])
                writer.writerow(row)
            
            # 合計容量の計算と出力
            writer.writerow([])  # 空行
            
            # 自己容量の合計
            self_cap = [f'{C[i,i]:.6g}' for i in range(n_conductors)]
            writer.writerow(['Self Capacitance'] + self_cap)
            
            # 相互容量の合計
            mutual_cap = []
            for i in range(n_conductors):
                total_mutual = sum(C[i,j] for j in range(n_conductors) if i != j)
                mutual_cap.append(f'{total_mutual:.6g}')
            writer.writerow(['Total Mutual Capacitance'] + mutual_cap)
            
            # 各導体の総容量
            total_cap = [f'{sum(C[i,:]):.6g}' for i in range(n_conductors)]
            writer.writerow(['Total Capacitance'] + total_cap)

        print(f"Capacitance matrix has been saved to {filename}")
        