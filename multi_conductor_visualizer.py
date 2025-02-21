# multi_conductor_visualizer.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from multi_conductor_calculator import MultiConductorCalculator

class MultiConductorVisualizer:
    def __init__(self, calculator: MultiConductorCalculator):
        self.calculator = calculator


    def plot_chage_distribution(self, charge_density: np.ndarray):
        scale = 1e9
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # FREE タイプで無限遠ノードがある場合のフラグ
        has_infinity_gnd = self.calculator.has_infinity_gnd
        
        start_idx = 0
        for i, conductor in enumerate(self.calculator.conductors):
            end_idx = start_idx + conductor['N_points']
            
            # 無限遠ノード対応: charge_density配列の境界チェック
            if has_infinity_gnd and end_idx > len(charge_density) - 1:
                end_idx = len(charge_density) - 1
                
            conductor_charge_density = charge_density[start_idx:end_idx]
            
            print(f"Conductor {i+1} charge density range:",
                np.min(conductor_charge_density),
                np.max(conductor_charge_density))

            dl = conductor['dl']
            conductor_charge = conductor_charge_density * dl

            print(f"Conductor {i+1} total charge:",
                np.sum(conductor_charge))

            theta = np.linspace(0, 2*np.pi, conductor['N_points'], endpoint=False)
            ax1.plot(theta, conductor_charge, label=f'Conductor {i+1}', marker='o')

            points = conductor['points']

            # 導体形状描画
            ax2.plot(points[:,0] * scale, points[:,1] * scale, 'k-', alpha=0.5)
            # 最後の点から最初の点に線を追加して閉じる
            ax2.plot([points[-1, 0] * scale, points[0, 0] * scale], 
                    [points[-1, 1] * scale, points[0, 1] * scale], 'k-', alpha=0.5)


            sizes = np.abs(conductor_charge)
            size = 50 + (sizes / (np.max(np.abs(charge_density)) * dl)) * 500

            scatter = ax2.scatter(points[:,0]*1e9, points[:,1]*1e9,
                                s=size,
                                c=conductor_charge,
                                cmap='RdBu_r',
                                norm=plt.Normalize(vmin=-np.max(np.abs(conductor_charge)),
                                                vmax=np.max(np.abs(conductor_charge))),
                                alpha=0.6)
            start_idx = end_idx
        
        # 無限遠ノードの電荷も表示（オプション）
        if has_infinity_gnd:
            inf_charge = charge_density[-1]
            print(f"Infinity GND node charge: {inf_charge}")
        
        ax1.set_xlabel('Angle [rad]')
        ax1.set_ylabel('Charge [C]')
        ax1.grid(True)
        ax1.legend()

        ax2.set_xlabel('x [nm]')  # 単位を変更
        ax2.set_ylabel('y [nm]')  # 単位を変更
        ax2.grid(True)

        # 変更後:
        colorbar = plt.colorbar(scatter, ax=ax2, label='Charge [C]')
        # FREEタイプの場合はGND線を描画しない
        if not hasattr(self.calculator, 'type') or self.calculator.type != "FREE":
            ax2.axhline(y=0, color='k', linestyle='--', label='GND')
        ax2.legend()

        ax2.axis('equal')

        plt.tight_layout()
        plt.show()

    def plot_potential(self, charge_density: np.ndarray):
        scale = 1e9

        # 導体のサイズに基づいてマージンを計算
        conductors_width = max([max(c['points'][:,0]) for c in self.calculator.conductors]) - min([min(c['points'][:,0]) for c in self.calculator.conductors])
        margin_x = max(conductors_width * 0.5, 200e-9)  # 導体幅の50%か200nmの大きい方
        
        # 表示範囲の計算
        x_min = min([min(c['points'][:,0]) for c in self.calculator.conductors]) - margin_x
        x_max = max([max(c['points'][:,0]) for c in self.calculator.conductors]) + margin_x
        y_min = 0

        # 最大高さの取得（矩形導体対応）
        max_height = 0
        for c in self.calculator.conductors:
            if c['type'] == 'circle':
                max_height = max(max_height, c['height'] + c['radius'])
            else:  # rectangle
                max_height = max(max_height, c['base_height'] + c['height'])
        
        y_max = max_height * 2

        y_min = -y_max

        # ポテンシャル計算
        X, Y, potential = self.calculate_potential(
            charge_density,
            (x_min, x_max),
            (y_min, y_max)
        )
        
        # プロット
        fig, ax = plt.subplots(figsize=(10, 8))
        
        contour = ax.contourf(X*scale, Y*scale, potential, levels=20, cmap='RdBu_r')
        plt.colorbar(contour, label='Potential [V]')
        
        # 導体の描画
        for conductor in self.calculator.conductors:
            points = conductor['points']
            ax.plot(points[:,0]*scale, points[:,1]*scale, 'k-', linewidth=2)
            # 最後の点から最初の点に線を追加して閉じる
            ax.plot([points[-1, 0] * scale, points[0, 0] * scale], 
                    [points[-1, 1] * scale, points[0, 1] * scale], 'k-', linewidth=2)

        # FREEタイプの場合はGND線を描画しない
        if not hasattr(self.calculator, 'type') or self.calculator.type != "FREE":
            ax.axhline(y=0, color='k', linestyle='--', label='GND')
    
        ax.set_xlabel('x [nm]')
        ax.set_ylabel('y [nm]')
        ax.grid(True)
        ax.axis('equal')
        plt.title('Potential Distribution')
        plt.show()
    
    ###########  電気力線表示 ##############
    # 開発中。　円ではなく点の集合でポリゴンにして内部かどうかを判定する必要があると思われる
    def is_inside_conductor(self, x: float, y: float, margin: float = 1e-9) -> bool:
        """Check if point (x,y) is inside any conductor"""
        for conductor in self.calculator.conductors:
            if conductor['type'] == 'circle':
                center_x = conductor['points'][0,0]  # 円の中心のx座標
                center_y = conductor['height']       # 円の中心のy座標
                if np.sqrt((x - center_x)**2 + (y - center_y)**2) <= conductor['radius'] + margin:
                    return True
            else:  # rectangle
                points = conductor['points']
                x_min = np.min(points[:,0]) - margin
                x_max = np.max(points[:,0]) + margin
                y_min = np.min(points[:,1]) - margin
                y_max = np.max(points[:,1]) + margin
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    return True
        return False

    def _generate_start_points(self, n_lines: int, charge_density: np.ndarray) -> np.ndarray:
        """Generate starting points for field lines based on charge distribution"""
        start_points = []
        offset = 2e-9  # 導体表面からのオフセットを大きめに

        # 無限遠GNDノードがある場合は対応
        has_infinity_gnd = hasattr(self.calculator, 'has_infinity_gnd') and self.calculator.has_infinity_gnd
        
        # 実際の導体ポイント数に対応する電荷密度を準備
        if has_infinity_gnd and len(charge_density) > sum(c['N_points'] for c in self.calculator.conductors):
            # 無限遠ノードを除いた電荷密度を使用
            actual_charge_density = charge_density[:-1]
        else:
            actual_charge_density = charge_density
        
        # 導体のdlの連結配列を作成
        all_dls = np.concatenate([c['dl'] for c in self.calculator.conductors])
        
        # 必要に応じて長さを調整
        if len(actual_charge_density) > len(all_dls):
            actual_charge_density = actual_charge_density[:len(all_dls)]
        elif len(actual_charge_density) < len(all_dls):
            all_dls = all_dls[:len(actual_charge_density)]
        
        # 合計重みを計算
        total_abs_charge_weights = np.sum(np.abs(actual_charge_density * all_dls))

        start_idx = 0
        for conductor in self.calculator.conductors:
            end_idx = start_idx + conductor['N_points']
            
            # 無限遠ノード対応: charge_density配列の境界チェック
            if end_idx > len(actual_charge_density):
                end_idx = len(actual_charge_density)
                
            conductor_charge = actual_charge_density[start_idx:end_idx]
            points = conductor['points']
            
            if end_idx > start_idx:  # ポイントがある場合のみ処理
                # 導体の電荷の符号を確認
                total_charge = np.sum(conductor_charge * conductor['dl'][:len(conductor_charge)])
                if abs(total_charge) < 1e-20:  # 電荷がほぼゼロの場合はスキップ
                    start_idx = end_idx
                    continue
                    
                # 電荷の符号に応じて法線ベクトルの向きを決定
                charge_sign = np.sign(total_charge)
                
                # 電荷密度の絶対値に比例した数の開始点を生成
                charge_weights = np.abs(conductor_charge * conductor['dl'][:len(conductor_charge)])
                total_weight = np.sum(charge_weights)
                if total_weight > 0:
                    n_points = max(1, int(n_lines * total_weight / total_abs_charge_weights))
                    
                    # 電荷密度の大きい場所により多くの開始点を配置
                    probabilities = charge_weights / total_weight
                    indices = np.random.choice(len(conductor_charge), size=n_points, p=probabilities)
                    
                    for idx in indices:
                        # 法線ベクトルの計算
                        next_idx = (idx + 1) % len(points)
                        prev_idx = idx - 1 if idx > 0 else len(points) - 1
                        
                        # 前後の点から接線ベクトルを計算
                        dx = points[next_idx,0] - points[prev_idx,0]
                        dy = points[next_idx,1] - points[prev_idx,1]
                        length = np.sqrt(dx*dx + dy*dy)
                        
                        if length > 0:
                            # 法線ベクトル (接線ベクトルを90度回転)
                            normal = np.array([-dy/length, dx/length])
                            # 電荷の符号に応じて法線の向きを調整
                            normal *= charge_sign
                            
                            # 開始点の位置を計算
                            start_point = points[idx] + normal * offset
                            
                            # 開始点が他の導体の内部に入っていないことを確認
                            if not self.is_inside_conductor(start_point[0], start_point[1]):
                                start_points.append(start_point)
                
            start_idx = end_idx
        
        return np.array(start_points) if start_points else np.array([])

    def plot_electric_field_lines(self, charge_density: np.ndarray, n_lines: int = 20):
        """Plot electric field lines using streamplot"""
        scale = 1e9  # Convert to nanometers for display
        
        # 導体のサイズに基づいてマージンを計算
        conductors_width = max([max(c['points'][:,0]) for c in self.calculator.conductors]) - min([min(c['points'][:,0]) for c in self.calculator.conductors])
        margin_x = max(conductors_width * 0.5, 200e-9)  # 導体幅の50%か200nmの大きい方

        x_min = min([min(c['points'][:,0]) for c in self.calculator.conductors]) - margin_x
        x_max = max([max(c['points'][:,0]) for c in self.calculator.conductors]) + margin_x
        
        # FREEタイプの場合は上下対称の表示領域にする
        if hasattr(self.calculator, 'type') and self.calculator.type == "FREE":
            max_height = 0
            for c in self.calculator.conductors:
                if c['type'] == 'circle':
                    max_height = max(max_height, c['height'] + c['radius'])
                else:  # rectangle
                    max_height = max(max_height, c['base_height'] + c['height'])
            
            y_margin = max_height * 1.5
            y_min = -y_margin
            y_max = y_margin
        else:
            # MSやSPタイプの場合は従来通り
            y_min = 0
            max_height = 0
            for c in self.calculator.conductors:
                if c['type'] == 'circle':
                    max_height = max(max_height, c['height'] + c['radius'])
                else:  # rectangle
                    max_height = max(max_height, c['base_height'] + c['height'])
            
            y_max = max_height * 2
        
        # Create grid for field calculation
        nx, ny = 50, 50
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)
        
        # Calculate electric field components
        Ex = np.zeros_like(X)
        Ey = np.zeros_like(Y)
        
        for i in range(nx):
            for j in range(ny):
                Ex[j,i], Ey[j,i] = self.calculate_electric_field(X[j,i], Y[j,i], charge_density)
        
        # Normalize field for better visualization
        E_magnitude = np.sqrt(Ex**2 + Ey**2)
        max_magnitude = np.percentile(E_magnitude[E_magnitude > 0], 95)
        if max_magnitude > 0:
            Ex = np.clip(Ex, -max_magnitude, max_magnitude)
            Ey = np.clip(Ey, -max_magnitude, max_magnitude)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot potential contours
        _, _, potential = self.calculate_potential(charge_density, (x_min, x_max), (y_min, y_max), n_points=nx)
        contour = ax.contourf(X*scale, Y*scale, potential, levels=20, cmap='RdBu_r', alpha=0.3)
        plt.colorbar(contour, label='Potential [V]')
        
        # Generate start points
        start_points = self._generate_start_points(n_lines, charge_density)
        if len(start_points) > 0:
            # Plot electric field lines
            streamplot = ax.streamplot(X*scale, Y*scale, Ex, Ey, 
                                    density=0.5,
                                    color='black',
                                    linewidth=1,
                                    arrowsize=1.5,
                                    start_points=start_points*scale,
                                    integration_direction='both')
        
        # Draw conductors
        for conductor in self.calculator.conductors:
            points = conductor['points']
            ax.plot(points[:,0]*scale, points[:,1]*scale, 'k-', linewidth=2)
            ax.plot([points[-1, 0]*scale, points[0, 0]*scale],
                    [points[-1, 1]*scale, points[0, 1]*scale], 'k-', linewidth=2)
        
        # FREEタイプ以外の場合のみGND面を描画
        if not hasattr(self.calculator, 'type') or self.calculator.type != "FREE":
            ax.axhline(y=0, color='k', linestyle='--', label='GND')
        
        ax.set_xlabel('x [nm]')
        ax.set_ylabel('y [nm]')
        ax.grid(True)
        ax.axis('equal')
        
        # タイトルにモデルタイプを含める
        if hasattr(self.calculator, 'type'):
            title = f'Electric Field Lines and Potential Distribution ({self.calculator.type} model)'
        else:
            title = 'Electric Field Lines and Potential Distribution'
        plt.title(title)
        
        plt.show()
    

    def calculate_potential(self, charge_density: np.ndarray, x_range: tuple, y_range: tuple, n_points: int = 100) -> np.ndarray:
      x = np.linspace(x_range[0], x_range[1], n_points)
      y = np.linspace(y_range[0], y_range[1], n_points)
      X, Y = np.meshgrid(x, y)
      potential = np.zeros_like(X)

      start_idx = 0
      for conductor in self.calculator.conductors:
          end_idx = start_idx + conductor['N_points']
          conductor_charge = charge_density[start_idx:end_idx]

          for i, (xi, yi, qi) in enumerate(zip(conductor['points'][:, 0], 
                                                  conductor['points'][:, 1], 
                                                  conductor_charge)):
              potential += qi * self.calculator.green_function(X, Y, xi, yi)

          start_idx = end_idx

      return X, Y, potential

    def calculate_electric_field(self, x: float, y: float, charge_density: np.ndarray) -> Tuple[float, float]:
        """Calculate electric field at point (x,y)"""
        # 導体内部の場合は電界=0を返す
        if self.is_inside_conductor(x, y):
            return 0.0, 0.0

        Ex = 0.0
        Ey = 0.0
        epsilon = self.calculator.epsilon_0 * self.calculator.epsilon_r
        
        # 無限遠GNDノードがある場合は対応
        has_infinity_gnd = hasattr(self.calculator, 'has_infinity_gnd') and self.calculator.has_infinity_gnd
        
        # 実際の導体ポイント数に対応する電荷密度を準備
        if has_infinity_gnd and len(charge_density) > sum(c['N_points'] for c in self.calculator.conductors):
            # 無限遠ノードを除いた電荷密度を使用
            actual_charge_density = charge_density[:-1]
        else:
            actual_charge_density = charge_density
        
        start_idx = 0
        for conductor in self.calculator.conductors:
            end_idx = start_idx + conductor['N_points']
            
            # 配列長の確認
            if end_idx > len(actual_charge_density):
                end_idx = len(actual_charge_density)
                
            conductor_charge = actual_charge_density[start_idx:end_idx]
            
            for i, (xi, yi, qi) in enumerate(zip(conductor['points'][:end_idx-start_idx, 0], 
                                            conductor['points'][:end_idx-start_idx, 1],
                                            conductor_charge)):
                dli = conductor['dl'][i] if i < len(conductor['dl']) else 0
                
                dx = x - xi
                dy = y - yi
                r2 = dx*dx + dy*dy
                if r2 < 1e-30:
                    continue
                    
                E = qi * dli / (2 * np.pi * epsilon)
                Ex += E * dx / r2
                Ey += E * dy / r2
                
                # MSタイプの場合のみイメージ法の影響を追加
                if hasattr(self.calculator, 'type') and self.calculator.type == "MS":
                    dy_image = y + yi
                    r2_image = dx*dx + dy_image*dy_image
                    Ex -= E * dx / r2_image
                    Ey -= E * dy_image / r2_image
                
            start_idx = end_idx
            
        # FREEタイプで無限遠ノードがある場合の影響を追加（オプション）
        if has_infinity_gnd and len(charge_density) > len(actual_charge_density):
            inf_charge = charge_density[-1]
            # ここに無限遠ノードからの電界の影響を計算するコードを追加
            # 例: 無限遠からの電界は (x,y) に向かう方向に弱く影響
            
        return Ex, Ey

