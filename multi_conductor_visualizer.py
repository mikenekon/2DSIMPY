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
    def is_inside_conductor(self, x: float, y: float, margin: float = 1e-9) -> bool:
        """Check if point (x,y) is inside any conductor with improved accuracy"""
        for conductor in self.calculator.conductors:
            if conductor['type'] == 'circle':
                # 円形導体の場合は中心からの距離で判定
                center_x = conductor['points'][0,0]  # 円の中心のx座標ではなく
                center_y = conductor['height']       # 実際の中心座標を計算
                radius = conductor['radius']
                
                # 中心からの距離を計算
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if distance <= radius + margin:
                    return True
            else:  # rectangle または多角形
                # Point-in-Polygon アルゴリズムを使用
                points = conductor['points']
                n = len(points)
                inside = False
                
                # Ray casting algorithm
                p1x, p1y = points[0]
                for i in range(n + 1):
                    p2x, p2y = points[i % n]
                    if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
                    p1x, p1y = p2x, p2y
                
                if inside:
                    return True
                
                # マージン分を考慮（簡易的）
                if min(np.linalg.norm(np.array([x, y]) - points[i]) for i in range(n)) <= margin:
                    return True
        
        return False


    def _generate_start_points(self, n_lines: int, charge_density: np.ndarray) -> np.ndarray:
        """Generate starting points for field lines with uniform distribution"""
        start_points = []
        offset = 3e-9  # 導体表面からのオフセット
        
        # 導体の数を取得
        n_conductors = len(self.calculator.conductors)
        
        # 各導体あたりの電気力線数を計算（均等に割り当て）
        lines_per_conductor = max(8, n_lines // n_conductors)
        
        # 無限遠GNDノードがある場合は対応
        has_infinity_gnd = hasattr(self.calculator, 'has_infinity_gnd') and self.calculator.has_infinity_gnd
        
        # 実際の導体ポイント数に対応する電荷密度を準備
        if has_infinity_gnd and len(charge_density) > sum(c['N_points'] for c in self.calculator.conductors):
            charge_density_adjusted = charge_density[:-1]
        else:
            charge_density_adjusted = charge_density
        
        start_idx = 0
        for conductor_idx, conductor in enumerate(self.calculator.conductors):
            end_idx = start_idx + conductor['N_points']
            
            # 配列長の確認
            if end_idx > len(charge_density_adjusted):
                end_idx = len(charge_density_adjusted)
                
            conductor_charge = charge_density_adjusted[start_idx:end_idx]
            points = conductor['points']
            
            if end_idx > start_idx:  # ポイントがある場合のみ処理
                # 導体の電荷の符号を確認
                dl_segment = conductor['dl'][:len(conductor_charge)]
                total_charge = np.sum(conductor_charge * dl_segment)
                if abs(total_charge) < 1e-20:  # 電荷がほぼゼロの場合はスキップ
                    start_idx = end_idx
                    continue
                    
                # 電荷の符号に応じて法線ベクトルの向きを決定
                charge_sign = np.sign(total_charge)
                
                # 導体形状ごとに均等に分布
                if conductor['type'] == 'circle':
                    # 円の場合は角度に基づいて均等に分布
                    theta = np.linspace(0, 2*np.pi, lines_per_conductor, endpoint=False)
                    for angle in theta:
                        # 円周上の点を計算
                        radius = conductor['radius']
                        center_x = conductor['points'][0,0]  # 中心x座標
                        center_y = conductor['height']       # 中心y座標
                        
                        pt_x = center_x + radius * np.cos(angle)
                        pt_y = center_y + radius * np.sin(angle)
                        
                        # 法線ベクトル（中心から外向き）
                        normal_x = np.cos(angle)
                        normal_y = np.sin(angle)
                        
                        # 電荷の符号に応じて調整
                        normal = np.array([normal_x, normal_y]) * charge_sign
                        
                        # 開始点の位置を計算
                        start_point = np.array([pt_x, pt_y]) + normal * offset
                        
                        # 他の導体と重ならないことを確認
                        if not self.is_inside_conductor(start_point[0], start_point[1]):
                            start_points.append(start_point)
                else:
                    # 矩形/多角形の場合は均等に点を配置
                    n_points = len(points)
                    # 辺の数に関わらず均等な間隔で点を選択
                    indices = np.linspace(0, n_points - 1, lines_per_conductor, dtype=int)
                    
                    for idx in indices:
                        # 法線ベクトル計算
                        next_idx = (idx + 1) % n_points
                        prev_idx = (idx - 1 + n_points) % n_points
                        
                        # 前後の点から接線ベクトルを計算
                        tangent_x = points[next_idx,0] - points[prev_idx,0]
                        tangent_y = points[next_idx,1] - points[prev_idx,1]
                        length = np.sqrt(tangent_x**2 + tangent_y**2)
                        
                        if length > 0:
                            # 法線ベクトル（接線ベクトルを90度回転）
                            normal_x = -tangent_y / length
                            normal_y = tangent_x / length
                            
                            # 外向き判定：導体の中心から見て外側を向いているか
                            center_x = np.mean(points[:,0])
                            center_y = np.mean(points[:,1])
                            point_to_center_x = center_x - points[idx,0]
                            point_to_center_y = center_y - points[idx,1]
                            
                            # 法線と中心へのベクトルの内積が正なら内向き
                            dot_product = normal_x * point_to_center_x + normal_y * point_to_center_y
                            if dot_product > 0:  # 内向きなら反転
                                normal_x = -normal_x
                                normal_y = -normal_y
                            
                            # 電荷の符号に応じて法線の向きを調整
                            normal_x *= charge_sign
                            normal_y *= charge_sign
                            
                            # 開始点の位置を計算
                            start_x = points[idx,0] + normal_x * offset
                            start_y = points[idx,1] + normal_y * offset
                            
                            # 開始点が他の導体の内部に入っていないことを確認
                            if not self.is_inside_conductor(start_x, start_y):
                                start_points.append(np.array([start_x, start_y]))
                
            start_idx = end_idx
        
        return np.array(start_points) if start_points else np.array([])

    def plot_electric_field_lines(self, charge_density: np.ndarray, n_lines: int = 40):
        """Plot electric field lines using streamplot with improved visualization"""
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
        
        # Create grid for field calculation - より細かいグリッドを使用
        nx, ny = 80, 80  # グリッド解像度を上げる
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
        
        # Generate start points with improved algorithm
        start_points = self._generate_start_points(n_lines, charge_density)
        if len(start_points) > 0:
            # Plot electric field lines with improved parameters
            streamplot = ax.streamplot(X*scale, Y*scale, Ex, Ey, 
                                    density=2.0,  # 密度を上げる
                                    color='black',
                                    linewidth=1,
                                    arrowsize=1.2,
                                    start_points=start_points*scale,
                                    integration_direction='both')  # 両方向に線を引く
        
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
        """Calculate electric field at point (x,y) with improved handling"""
        # 導体内部判定を改善したメソッドを使用
        if self.is_inside_conductor(x, y):
            return 0.0, 0.0  # 導体内部は電界ゼロ

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

