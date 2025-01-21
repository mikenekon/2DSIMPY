import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from typing import Tuple, List , Dict
import concurrent.futures
import csv

class MultiConductorCalculator:
  def __init__(self, epsilon_r: float = 1.0 , epsilon_0: float = 8.854e-12, height_top: float = None):
    self.conductors = []
    self.epsilon_0 = epsilon_0
    self.epsilon_r = epsilon_r
    self.height_top = height_top  # StripLineの上GNDの高さ

    self.n_gauss = 10
    self.gauss_points, self.gauss_weights = np.polynomial.legendre.leggauss(self.n_gauss)
    self.gauss_points = (self.gauss_points + 1) / 2
    self.gauss_weights = self.gauss_weights / 2

    # 高速化のために定数の計算をクラス内で保持
    self.epsilon = self.epsilon_0 * self.epsilon_r
    self.constant = 1 / (4 * np.pi * self.epsilon)
    self.cache_influence_matrix = None  # 行列Aを再利用
    self.cache_inverse_matrix = None    # 逆行列Aのキャッシュ

  def add_conductor(self, radius: float, height: float, N_points: int, x_offset: float = 0.0):
      """円形導体追加メソッド"""
      points = self._create_circle_points(radius, height, N_points, x_offset)
      
      conductor = {
          'type': 'circle',
          'radius': radius,
          'height': height,
          'N_points': N_points,
          'points': points,
          'dl': self._calculate_dl_for_points(points)  # 統一された dl 計算
      }
      self.conductors.append(conductor)
      self.cache_influence_matrix = None  # キャッシュされてる行列Aをクリア
      return len(self.conductors) - 1

  def add_rectangular_conductor(self, width: float, height: float, base_height: float, 
                              N_points: int, x_offset: float = 0.0):
      """矩形導体追加メソッド"""
      points = self._create_rectangle_points(width, height, base_height, N_points, x_offset)
      
      conductor = {
          'type': 'rectangle',
          'width': width,
          'height': height,
          'base_height': base_height,
          'N_points': len(points),
          'points': points,
          'dl': self._calculate_dl_for_points(points)  # 統一された dl 計算
      }
      
      self.conductors.append(conductor)
      self.cache_influence_matrix = None  # キャッシュされてる行列Aをクリア
      return len(self.conductors) - 1

  def _calculate_dl_for_points(self, points: np.ndarray) -> np.ndarray:
      """点列に対する dl を計算（形状によらず共通）"""
      N = len(points)
      dl = np.zeros(N)
      
      for i in range(N):
          next_i = (i + 1) % N
          dx = points[next_i, 0] - points[i, 0]
          dy = points[next_i, 1] - points[i, 1]
          dl[i] = np.sqrt(dx*dx + dy*dy)
      
      return dl

  def _create_circle_points(self, radius: float, height: float, N_points: int, x_offset: float = 0.0) -> np.ndarray:
      """円形の点列を生成"""
      theta = np.linspace(0, 2*np.pi, N_points, endpoint=False)
      x = radius * np.cos(theta) + x_offset
      y = height + radius * np.sin(theta)
      return np.column_stack((x, y))

  def _create_rectangle_points(self, width: float, height: float, base_height: float,
                            N_points: int, x_offset: float = 0.0) -> np.ndarray:
      """矩形の点列を生成"""
      perimeter = 2 * (width + height)
      points_per_unit = N_points / perimeter
      
      # 各辺の点数を計算
      points_width = int(np.round(width * points_per_unit))
      points_height = int(np.round(height * points_per_unit))
      
      points_list = []
      
      # 下辺
      x_bottom = np.linspace(-width/2, width/2, points_width)
      y_bottom = np.full_like(x_bottom, base_height)
      points_list.extend(zip(x_bottom[:-1], y_bottom[:-1]))
      
      # 右辺
      y_right = np.linspace(base_height, base_height+height, points_height)
      x_right = np.full_like(y_right, width/2)
      points_list.extend(zip(x_right[:-1], y_right[:-1]))
      
      # 上辺
      x_top = np.linspace(width/2, -width/2, points_width)
      y_top = np.full_like(x_top, base_height+height)
      points_list.extend(zip(x_top[:-1], y_top[:-1]))
      
      # 左辺
      y_left = np.linspace(base_height+height, base_height, points_height)
      x_left = np.full_like(y_left, -width/2)
      points_list.extend(zip(x_left[:-1], y_left[:-1]))
      
      points = np.array(points_list)
      points[:, 0] += x_offset
      
      return points

  # Green関数
  def green_function(self, x: np.ndarray, y:np.ndarray,
                     x_prime: np.ndarray, y_prime: np.ndarray) -> np.ndarray:
    if self.height_top is not None:
       return self._green_function_sp(x,y,x_prime,y_prime)
    else:
       return self._green_function_ms(x,y,x_prime,y_prime)


  # MS用Green関数
  def _green_function_ms(self, x: np.ndarray, y:np.ndarray,
                     x_prime: np.ndarray, y_prime: np.ndarray) -> np.ndarray:
    # 高速化のために定数の計算をクラス内で保持
    epsilon = self.epsilon     # self.epsilon_0 * self.epsilon_r
    constant = self.constant   #  1 / (4 * np.pi * self.epsilon)

    x2 = (x - x_prime)**2

    # 本体
    r_squared = x2 + (y - y_prime)**2
    r_squared = np.maximum(r_squared, 1e-30)
    G_direct = -constant * np.log(r_squared)

    # 影像
    r_image_squared = x2 + (y + y_prime)**2
    r_image_squared = np.maximum(r_image_squared, 1e-30)
    G_image = constant * np.log(r_image_squared)

    return G_direct + G_image

  # SP用Green関数
  def _green_function_sp(self, x: np.ndarray, y:np.ndarray,
                     x_prime: np.ndarray, y_prime: np.ndarray) -> np.ndarray:
    # 高速化のために定数の計算をクラス内で保持
    epsilon = self.epsilon     # self.epsilon_0 * self.epsilon_r
    constant = self.constant   #  1 / (4 * np.pi * self.epsilon)
    h = self.height_top

    eps = 1e-30  # 特異点回避のための微小値

    dx = x - x_prime
    dy = y - y_prime
    dy_plus = y + y_prime

    sinh_term = np.sinh(np.pi * dx / (2*h))**2
    sin_term_minus = np.sin(np.pi*dy/(2*h))**2
    sin_term_plus = np.sin(np.pi*dy_plus / (2*h))**2

    numerator = sinh_term + sin_term_plus

    # 0割り対策
    denominator = np.maximum(sinh_term + sin_term_minus , eps)
    g = -constant * np.log(numerator / denominator)
 
    return g
  

  # 線積分の数値解析処理
  def line_integral(self , x1: float, y1: float, x2: float, y2: float,
                  x1_prime: float, y1_prime: float, x2_prime: float, y2_prime: float) -> float:
    result = 0.0

    mid_x1 = (x1 + x2) / 2
    mid_y1 = (y1 + y2) / 2
    mid_x2 = (x1_prime + x2_prime) / 2
    mid_y2 = (y1_prime + y2_prime) / 2

    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x2_prime - x1_prime
    dy2 = y2_prime - y1_prime

    is_self = np.abs(mid_x1 - mid_x2) < 1e-10 and np.abs(mid_y1 - mid_y2) < 1e-10

    n_points = self.n_gauss * 10 if is_self else self.n_gauss
    gauss_points, gauss_weights = np.polynomial.legendre.leggauss(n_points)
    gauss_points = (gauss_points + 1) / 2
    gauss_weights = gauss_weights / 2

    for i, wi in zip(gauss_points, gauss_weights):
      x_i = mid_x1 + (i - 0.5) * dx1
      y_i = mid_y1 + (i - 0.5) * dy1

      for j, wj in zip(gauss_points, gauss_weights):
        x_j = mid_x2 + (j - 0.5) * dx2
        y_j = mid_y2 + (j - 0.5) * dy2

        result += wi * wj * self.green_function(x_i, y_i, x_j , y_j)

    dl1 = np.sqrt(dx1**2 + dy1**2)
    dl2 = np.sqrt(dx2**2 + dy2**2)
    result *= dl1 * dl2

    return result
  
  # 線積分の数値解析処理　ベクトル化バージョン
  def line_integral_vec(self, x1: float, y1: float, x2: float, y2: float,
                      x1_prime: float, y1_prime: float, x2_prime: float, y2_prime: float) -> float:
    mid_x1 = (x1 + x2) / 2
    mid_y1 = (y1 + y2) / 2
    mid_x2 = (x1_prime + x2_prime) / 2
    mid_y2 = (y1_prime + y2_prime) / 2

    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x2_prime - x1_prime
    dy2 = y2_prime - y1_prime

    is_self = np.abs(mid_x1 - mid_x2) < 1e-10 and np.abs(mid_y1 - mid_y2) < 1e-10

    # ガウス点の数を決定
    n_points = self.n_gauss * 10 if is_self else self.n_gauss
    gauss_points, gauss_weights = np.polynomial.legendre.leggauss(n_points)
    gauss_points = (gauss_points + 1) / 2  # [0, 1] の範囲にスケール
    gauss_weights = gauss_weights / 2      # 重みを [0, 1] に合わせて調整

    # ベクトル化された計算のためのメッシュグリッドを作成
    i_mesh, j_mesh = np.meshgrid(gauss_points, gauss_points)
    wi_mesh, wj_mesh = np.meshgrid(gauss_weights, gauss_weights)

    # 最初と2番目のセグメントのすべての積分点を計算
    x_i = mid_x1 + (i_mesh - 0.5) * dx1
    y_i = mid_y1 + (i_mesh - 0.5) * dy1
    x_j = mid_x2 + (j_mesh - 0.5) * dx2
    y_j = mid_y2 + (j_mesh - 0.5) * dy2

    # すべての点ペアに対してグリーン関数を計算
    green_values = self.green_function(x_i, y_i, x_j, y_j)

    # 積分の加重和を実行
    result = np.sum(wi_mesh * wj_mesh * green_values)

    # 線分の長さでスケール
    dl1 = np.sqrt(dx1**2 + dy1**2)
    dl2 = np.sqrt(dx2**2 + dy2**2)
    result *= dl1 * dl2

    return result

  # 行列Aを計算（導体数が変わらなければキャッシュした行列を返す）
  def create_influence_matrix(self) -> np.array:
      # 影響行列の事前計算（ここでは一度計算してキャッシュ）
      if self.cache_influence_matrix is None:
        total_points = sum(c['N_points'] for c in self.conductors)
        A = np.zeros((total_points, total_points))

        current_row = 0
        for i, conductor_i in enumerate(self.conductors):
          current_col = 0
          for j, conductor_j in enumerate(self.conductors):
            sub_matrix = self._calculate_A_matrix(conductor_i, conductor_j)
            A[current_row:current_row+conductor_i['N_points'],
              current_col:current_col+conductor_j['N_points']] = sub_matrix
            current_col += conductor_j['N_points']
          current_row += conductor_i['N_points']

        self.cache_influence_matrix = A
      return self.cache_influence_matrix
      
  # 行列Aの計算
  def _calculate_A_matrix(self, conductor_i: Dict, conductor_j: Dict) -> np.ndarray:
    Ni = conductor_i['N_points']
    Nj = conductor_j['N_points']
    sub_matrix = np.zeros((Ni, Nj))

    for i in range(Ni):
      for j in range(Nj):
        i_prev = i - 1 if i > 0 else Ni - 1
        i_next = (i + 1) % Ni
        j_prev = j - 1 if j > 0 else Nj - 1
        j_next = (j + 1) % Nj

        x1_mid = (conductor_i['points'][i][0] + conductor_i['points'][i_prev][0]) / 2
        y1_mid = (conductor_i['points'][i][1] + conductor_i['points'][i_prev][1]) / 2
        x2_mid = (conductor_i['points'][i][0] + conductor_i['points'][i_next][0]) / 2
        y2_mid = (conductor_i['points'][i][1] + conductor_i['points'][i_next][1]) / 2

        x1_prime_mid = (conductor_j['points'][j][0] + conductor_j['points'][j_prev][0]) / 2
        y1_prime_mid = (conductor_j['points'][j][1] + conductor_j['points'][j_prev][1]) / 2
        x2_prime_mid = (conductor_j['points'][j][0] + conductor_j['points'][j_next][0]) / 2
        y2_prime_mid = (conductor_j['points'][j][1] + conductor_j['points'][j_next][1]) / 2

        len1 = np.sqrt( (x2_mid - x1_mid)**2 + (y2_mid - y1_mid)**2 )
        len2 = np.sqrt( (x2_prime_mid - x1_prime_mid)**2 + (y2_prime_mid - y1_prime_mid)**2 )

        sub_matrix[i, j] = self.line_integral_vec(
            x1_mid, y1_mid, x2_mid, y2_mid,
            x1_prime_mid, y1_prime_mid, x2_prime_mid, y2_prime_mid,
        ) / (len1 * len2)


    return sub_matrix

  # 電荷密度計算
  def solve_charge_density(self, voltages: List[float]) -> np.ndarray:
    A = self.create_influence_matrix()

    v = np.zeros(sum(c['N_points'] for c in self.conductors))
    start_idx = 0
    for i, conductor in enumerate(self.conductors):
      v[start_idx:start_idx + conductor['N_points']] = voltages[i]
      start_idx += conductor['N_points']

    if self.cache_inverse_matrix is None:
       self.cache_inverse_matrix = linalg.inv(A)

    chage_density = self.cache_inverse_matrix @ v

    return chage_density

  # 容量マトリックスを計算
  def calculate_capacitance_matrix(self) -> np.ndarray:
    n_conductors = len(self.conductors)
    C = np.zeros((n_conductors, n_conductors))
    
    cumsum_points = np.cumsum([0] + [c['N_points'] for c in self.conductors])
    conversion_factor = 1
    
    for i in range(n_conductors):
        voltages = [1.0 if j == i else 0.0 for j in range(n_conductors)]
        q = self.solve_charge_density(voltages)
        
        for j in range(n_conductors):
            # Extract charge density for current conductor
            q_conductor = q[cumsum_points[j]:cumsum_points[j+1]]
            dl = self.conductors[j]['dl']
            
            # Calculate total charge on conductor j
            total_charge = 0.0
            for q_i in q_conductor:
                total_charge += q_i
            
            # Store the result
            C[j,i] = total_charge * conversion_factor
    
    return C

  def calculate_potential(self, charge_density: np.ndarray, x_range: tuple, y_range: tuple, n_points: int = 100) -> np.ndarray:
      x = np.linspace(x_range[0], x_range[1], n_points)
      y = np.linspace(y_range[0], y_range[1], n_points)
      X, Y = np.meshgrid(x, y)
      potential = np.zeros_like(X)

      start_idx = 0
      for conductor in self.conductors:
          end_idx = start_idx + conductor['N_points']
          conductor_charge = charge_density[start_idx:end_idx]

          for i, (xi, yi, qi) in enumerate(zip(conductor['points'][:, 0], 
                                                  conductor['points'][:, 1], 
                                                  conductor_charge)):
              potential += qi * self.green_function(X, Y, xi, yi)

          start_idx = end_idx

      return X, Y, potential


  def plot_chage_distribution(self, charge_density: np.ndarray):
      scale = 1e9
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

      start_idx = 0
      for i, conductor in enumerate(self.conductors):
          end_idx = start_idx + conductor['N_points']
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

      ax1.set_xlabel('Angle [rad]')
      ax1.set_ylabel('Charge [C]')
      ax1.grid(True)
      ax1.legend()

      ax2.set_xlabel('x [nm]')  # 単位を変更
      ax2.set_ylabel('y [nm]')  # 単位を変更
      ax2.grid(True)
      colorbar = plt.colorbar(scatter, ax=ax2, label='Charge [C]')
      ax2.axhline(y=0, color='k', linestyle='--', label='GND')
      ax2.legend()
      ax2.axis('equal')

      plt.tight_layout()
      plt.show()

  def plot_potential(self, charge_density: np.ndarray):
      scale = 1e9

      # 導体のサイズに基づいてマージンを計算
      conductors_width = max([max(c['points'][:,0]) for c in self.conductors]) - min([min(c['points'][:,0]) for c in self.conductors])
      margin_x = max(conductors_width * 0.5, 200e-9)  # 導体幅の50%か200nmの大きい方
        
      # 表示範囲の計算
      x_min = min([min(c['points'][:,0]) for c in self.conductors]) - margin_x
      x_max = max([max(c['points'][:,0]) for c in self.conductors]) + margin_x
      y_min = 0

      # 最大高さの取得（矩形導体対応）
      max_height = 0
      for c in self.conductors:
          if c['type'] == 'circle':
              max_height = max(max_height, c['height'] + c['radius'])
          else:  # rectangle
              max_height = max(max_height, c['base_height'] + c['height'])
      
      y_max = max_height * 2
      
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
      for conductor in self.conductors:
          points = conductor['points']
          ax.plot(points[:,0]*scale, points[:,1]*scale, 'k-', linewidth=2)
          # 最後の点から最初の点に線を追加して閉じる
          ax.plot([points[-1, 0] * scale, points[0, 0] * scale], 
                  [points[-1, 1] * scale, points[0, 1] * scale], 'k-', linewidth=2)

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
      for conductor in self.conductors:
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

  def calculate_electric_field(self, x: float, y: float, charge_density: np.ndarray) -> Tuple[float, float]:
      """Calculate electric field at point (x,y)"""
      # 導体内部の場合は電界=0を返す
      if self.is_inside_conductor(x, y):
          return 0.0, 0.0

      Ex = 0.0
      Ey = 0.0
      epsilon = self.epsilon_0 * self.epsilon_r
      
      start_idx = 0
      for conductor in self.conductors:
          end_idx = start_idx + conductor['N_points']
          conductor_charge = charge_density[start_idx:end_idx]
          
          for xi, yi, qi, dli in zip(conductor['points'][:, 0], 
                                    conductor['points'][:, 1],
                                    conductor_charge,
                                    conductor['dl']):
              dx = x - xi
              dy = y - yi
              r2 = dx*dx + dy*dy
              if r2 < 1e-30:
                  continue
                  
              E = qi * dli / (2 * np.pi * epsilon)
              Ex += E * dx / r2
              Ey += E * dy / r2
              
              # Image charge contribution
              dy_image = y + yi
              r2_image = dx*dx + dy_image*dy_image
              Ex -= E * dx / r2_image
              Ey -= E * dy_image / r2_image
              
          start_idx = end_idx
          
      return Ex, Ey

  def _generate_start_points(self, n_lines: int, charge_density: np.ndarray) -> np.ndarray:
      """Generate starting points for field lines based on charge distribution"""
      start_points = []
      offset = 2e-9  # 導体表面からのオフセットを大きめに

      start_idx = 0
      for conductor in self.conductors:
          end_idx = start_idx + conductor['N_points']
          conductor_charge = charge_density[start_idx:end_idx]
          points = conductor['points']
          
          # 導体の電荷の符号を確認
          total_charge = np.sum(conductor_charge * conductor['dl'])
          if abs(total_charge) < 1e-20:  # 電荷がほぼゼロの場合はスキップ
              start_idx = end_idx
              continue
              
          # 電荷の符号に応じて法線ベクトルの向きを決定
          charge_sign = np.sign(total_charge)
          
          # 電荷密度の絶対値に比例した数の開始点を生成
          charge_weights = np.abs(conductor_charge * conductor['dl'])
          total_weight = np.sum(charge_weights)
          if total_weight > 0:
              n_points = max(1, int(n_lines * total_weight / 
                            np.sum(np.abs(charge_density * np.concatenate([c['dl'] for c in self.conductors])))))
              
              # 電荷密度の大きい場所により多くの開始点を配置
              probabilities = charge_weights / total_weight
              indices = np.random.choice(len(points), size=n_points, p=probabilities)
              
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
      conductors_width = max([max(c['points'][:,0]) for c in self.conductors]) - min([min(c['points'][:,0]) for c in self.conductors])
      margin_x = max(conductors_width * 0.5, 200e-9)  # 導体幅の50%か200nmの大きい方
    
      x_min = min([min(c['points'][:,0]) for c in self.conductors]) - margin_x
      x_max = max([max(c['points'][:,0]) for c in self.conductors]) + margin_x
      y_min = 0

      max_height = 0
      for c in self.conductors:
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
                                  density=0.5,  # 密度を下げる
                                  color='black',
                                  linewidth=1,
                                  arrowsize=1.5,
                                  start_points=start_points*scale,
                                  integration_direction='both')  # 両方向に線を引く
      
      # Draw conductors
      for conductor in self.conductors:
          points = conductor['points']
          ax.plot(points[:,0]*scale, points[:,1]*scale, 'k-', linewidth=2)
          ax.plot([points[-1, 0]*scale, points[0, 0]*scale],
                  [points[-1, 1]*scale, points[0, 1]*scale], 'k-', linewidth=2)
      
      ax.axhline(y=0, color='k', linestyle='--', label='GND')
      ax.set_xlabel('x [nm]')
      ax.set_ylabel('y [nm]')
      ax.grid(True)
      ax.axis('equal')
      plt.title('Electric Field Lines and Potential Distribution')
      plt.show()
      
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
    C = self.calculate_capacitance_matrix()
    
    # 長さを掛ける
    if length is not None:
        C = C * length
    
    # 単位変換を適用
    if unit_prefix in unit_factors:
        C = C * unit_factors[unit_prefix]
    
    # ヘッダー行の作成
    n_conductors = len(self.conductors)
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