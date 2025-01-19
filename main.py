import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from typing import Tuple, List , Dict
import concurrent.futures

class MultiConductorCalculator:
  def __init__(self, epsilon_r: float = 1.0 , epsilon_0: float = 8.854e-12):
    self.conductors = []
    self.epsilon_0 = epsilon_0
    self.epsilon_r = epsilon_r

    self.n_gauss = 10
    self.gauss_points, self.gauss_weights = np.polynomial.legendre.leggauss(self.n_gauss)
    self.gauss_points = (self.gauss_points + 1) / 2
    self.gauss_weights = self.gauss_weights / 2

    # 高速化のために定数の計算をクラス内で保持
    self.epsilon = self.epsilon_0 * self.epsilon_r
    self.constant = 1 / (4 * np.pi * self.epsilon)
    self.cache_influence_matrix = None  # 行列Aを再利用

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

  def green_function(self, x: np.ndarray, y:np.ndarray,
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

  def line_integral_2(self, x1, y1, x2, y2, x1_prime, y1_prime, x2_prime, y2_prime):
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

    # 並列化部分
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_results = []
        for i, wi in zip(gauss_points, gauss_weights):
            x_i = mid_x1 + (i - 0.5) * dx1
            y_i = mid_y1 + (i - 0.5) * dy1

            for j, wj in zip(gauss_points, gauss_weights):
                x_j = mid_x2 + (j - 0.5) * dx2
                y_j = mid_y2 + (j - 0.5) * dy2

                future_results.append(executor.submit(self.green_function, x_i, y_i, x_j, y_j))

        # 結果を集計
        for future in concurrent.futures.as_completed(future_results):
            result += future.result()

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
      
  # 行列Aの計算に誤りがあると思われる 1/(Segi*Segj) * G * dl * dl を考慮していない？
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

        #sub_matrix[i, j] = self.line_integral(
        #    x1_mid, y1_mid, x2_mid, y2_mid,
        #    x1_prime_mid, y1_prime_mid, x2_prime_mid, y2_prime_mid,
        #)

        # 1/(長さ１*長さ2)が考慮されていない？
        #  試しに1/長さにしたところポテンシャルのオーダーがあった。
        # つまりどこかで考え違いをしており、A行列の1/SEGが抜けている可能性が高い。
        len1 = np.sqrt( (x2_mid - x1_mid)**2 + (y2_mid - y1_mid)**2 )
        len2 = np.sqrt( (x2_prime_mid - x1_prime_mid)**2 + (y2_prime_mid - y1_prime_mid)**2 )

        sub_matrix[i, j] = self.line_integral(
            x1_mid, y1_mid, x2_mid, y2_mid,
            x1_prime_mid, y1_prime_mid, x2_prime_mid, y2_prime_mid,
        ) / (len1 )


    return sub_matrix


  def solve_charge_density(self, voltages: List[float]) -> np.ndarray:
    A = self.create_influence_matrix()

    v = np.zeros(sum(c['N_points'] for c in self.conductors))
    start_idx = 0
    for i, conductor in enumerate(self.conductors):
      v[start_idx:start_idx + conductor['N_points']] = voltages[i]
      start_idx += conductor['N_points']

    chage_density = linalg.solve(A, v)

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
            for q_i, dl_i in zip(q_conductor, dl):
                total_charge += q_i * dl_i
            
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

          for i, (xi, yi, qi, dli) in enumerate(zip(conductor['points'][:, 0], 
                                                  conductor['points'][:, 1], 
                                                  conductor_charge,
                                                  conductor['dl'])):
              potential += qi * dli * self.green_function(X, Y, xi, yi)

          start_idx = end_idx

      return X, Y, potential


  def plot_chage_distribution(self, charge_density: np.ndarray):
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
          # μm → nm に変更 (*1e9)
          ax2.plot(points[:,0]*1e9, points[:,1]*1e9, 'k-', alpha=0.5)

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
      # 表示範囲の計算
      x_min = min([min(c['points'][:,0]) for c in self.conductors]) - 100e-9
      x_max = max([max(c['points'][:,0]) for c in self.conductors]) + 100e-9
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
      
      contour = ax.contourf(X*1e9, Y*1e9, potential, levels=20, cmap='RdBu_r')
      plt.colorbar(contour, label='Potential [V]')
      
      # 導体の描画
      for conductor in self.conductors:
          points = conductor['points']
          ax.plot(points[:,0]*1e9, points[:,1]*1e9, 'k-', linewidth=2)
      
      ax.axhline(y=0, color='k', linestyle='--', label='GND')
      ax.set_xlabel('x [nm]')
      ax.set_ylabel('y [nm]')
      ax.grid(True)
      ax.axis('equal')
      plt.title('Potential Distribution')
      plt.show()
      

def test1():
  calculator = MultiConductorCalculator(epsilon_r=2.0)

  radius = 100e-9
  height = 300e-9
  spacing = 3 * radius
  n=20

  calculator.add_conductor(radius=radius, height=height, N_points=n, x_offset=0.0)
  calculator.add_conductor(radius=radius, height=height, N_points=n, x_offset=spacing)
  calculator.add_conductor(radius=radius, height=height, N_points=n, x_offset=spacing*2)

  voltages = [1.0,-1.0, 2.0]
  charge_density = calculator.solve_charge_density(voltages)

  C = calculator.calculate_capacitance_matrix()
  print("\n C Matrix [F/m]:")
  print(C)

  calculator.plot_chage_distribution(charge_density)

  calculator.plot_potential(charge_density)

# RECTのテスト
def test2():
    calculator = MultiConductorCalculator(epsilon_r=2.0)
    
    w = 20e-9
    h = 50e-9
    n = 12
    bh1 = 300e-9
    bh2 = 400e-9
    bh3 = 500e-9
    bh4 = 600e-9
    bh5 = 700e-9
    xo1 = 0
    xo2 = 400e-9
    xo3 = 800e-9
    xo4 = 1200e-9
    xo5 = 1600e-9
    
        # Add rectangular conductor
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh1, N_points=n, x_offset=xo1  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh1, N_points=n, x_offset=xo2  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh1, N_points=n, x_offset=xo3  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh1, N_points=n, x_offset=xo4  )
#    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh1, N_points=n, x_offset=xo5  )

    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh2, N_points=n, x_offset=xo1  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh2, N_points=n, x_offset=xo2  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh2, N_points=n, x_offset=xo3  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh2, N_points=n, x_offset=xo4  )
#    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh2, N_points=n, x_offset=xo5  )

    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh3, N_points=n, x_offset=xo1  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh3, N_points=n, x_offset=xo2  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh3, N_points=n, x_offset=xo3  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh3, N_points=n, x_offset=xo4  )
#    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh3, N_points=n, x_offset=xo5  )

    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh4, N_points=n, x_offset=xo1  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh4, N_points=n, x_offset=xo2  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh4, N_points=n, x_offset=xo3  )
#    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh4, N_points=n, x_offset=xo4  )
#    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh4, N_points=n, x_offset=xo5  )

    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh5, N_points=n, x_offset=xo1  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh5, N_points=n, x_offset=xo2  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh5, N_points=n, x_offset=xo3  )
    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh5, N_points=n, x_offset=xo4  )
#    calculator.add_rectangular_conductor( width=w, height=h, base_height=bh5, N_points=n, x_offset=xo5  )

    # Modified voltages array to match the number of conductors
    voltages = [1.0 ,1,0, 1.0, 1.0, 1.0,
                1.0 ,1,0, 1.0, 1.0, 1.0,
                1.0 ,1,0, 1.0, 1.0, 1.0,
                1.0 ,1,0, 1.0, 1.0, 1.0,
                1.0 ,1,0, 1.0, 1.0, 1.0
                ]  # Only one voltage for one conductor
    
    charge_density = calculator.solve_charge_density(voltages)
    calculator.plot_chage_distribution(charge_density)
    calculator.plot_potential(charge_density) 
       
    C = calculator.calculate_capacitance_matrix()
    print("\n C Matrix [F/m]:")
    print(C)
    
    # calculator.plot_chage_distribution(charge_density)
    # calculator.plot_potential(charge_density)



if __name__ == "__main__":
  # test1()
  test2()
