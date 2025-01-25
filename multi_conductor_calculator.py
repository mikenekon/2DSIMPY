# multi_conductor_calculator.py

import numpy as np
from scipy import linalg
from typing import Tuple, List, Dict
import multiprocessing as mp

class MultiConductorCalculator:
    def __init__(self, 
                 type: str = "FREE",   # "MS" , "SP"  , "FREE(GNDなし)"
                 epsilons: List[float] = None,  # [ε1, ε2, ...]
                 heights: List[float] = None):  # [h1, h2, ...]
        """
        Args:
            epsilons: List of relative permittivities [ε1, ε2, ...]
            heights: List of heights from Y=0 [h1, h2, ...]
        """
        self.type = type

        # epsilonと高さをセット。最後の高さを省略した場合は無限大とする
        if epsilons is None:
            self.epsilons = [1.0]
            self.heights = [float('inf')]
        else:
            self.epsilons = epsilons
            if heights is None:
                self.heights = [float('inf')]
            elif len(heights) == len(epsilons) - 1:
                self.heights = heights + [float('inf')]
            elif len(heights) == len(epsilons):
                self.heights = heights
            else:
                raise ValueError("heights must be same length as epsilons or one less")

        self.height_top = self.heights[-1]
        self.epsilon_r = self.epsilons[0]
                
        self.type = type
        self.height_top = self.heights[-1] if heights else float('inf')
        self.conductors = []
        self.epsilon_0 = 8.854e-12    # 真空中の誘電率
        self.epsilon_r = self.epsilons[0] if epsilons else 1.0  # 最初の層の誘電率を設定
        self.mu_0 = 4 * np.pi * 1e-7  # 真空の透磁率 (H/m)
        self.c_squared = 1 / (self.mu_0 * self.epsilon_0)  # 光速の2乗 (m^2/s^2)
        
        self.n_gauss = 10
        self.gauss_points, self.gauss_weights = np.polynomial.legendre.leggauss(self.n_gauss)
        self.gauss_points = (self.gauss_points + 1) / 2
        self.gauss_weights = self.gauss_weights / 2

        # 高速化のために定数の計算をクラス内で保持
        self.epsilon = self.epsilon_0 * self.epsilon_r
        self.constant = 1 / (4 * np.pi * self.epsilon)
        self.cache_influence_matrix = None  # 行列Aを再利用
        self.cache_inverse_matrix = None    # 逆行列Aのキャッシュ



    def add_conductor(self, radius: float, height: float, N_points: int, 
                    x_offset: float = 0.0, is_gnd: bool = False):
        """円形導体追加メソッド"""
        points = self._create_circle_points(radius, height, N_points, x_offset)
        conductor = {
            'type': 'circle',
            'radius': radius,
            'height': height,
            'N_points': N_points,
            'points': points,
            'dl': self._calculate_dl_for_points(points),
            'is_gnd': is_gnd
        }
        self.conductors.append(conductor)
        self.cache_influence_matrix = None
        return len(self.conductors) - 1

    def add_rectangular_conductor(self, width: float, height: float, 
                                base_height: float, N_points: int, 
                                x_offset: float = 0.0, is_gnd: bool = False):
        """矩形導体追加メソッド"""
        points = self._create_rectangle_points(width, height, base_height, N_points, x_offset)
        conductor = {
            'type': 'rectangle',
            'width': width,
            'height': height,
            'base_height': base_height,
            'N_points': len(points),
            'points': points,
            'dl': self._calculate_dl_for_points(points),
            'is_gnd': is_gnd
        }
        self.conductors.append(conductor)
        self.cache_influence_matrix = None
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
    def green_function(self, x: np.ndarray, y: np.ndarray,
                    x_prime: np.ndarray, y_prime: np.ndarray) -> np.ndarray:
        if self.type == "SP":
            return self._green_function_sp(x, y, x_prime, y_prime)
        elif self.type == "MS":
            return self._green_function_ms(x, y, x_prime, y_prime)
        else:  # FREE
            return self._green_function_free(x, y, x_prime, y_prime)
        
    def _green_function_free(self, x: np.ndarray, y: np.ndarray,
                            x_prime: np.ndarray, y_prime: np.ndarray) -> np.ndarray:
        """自己項のみのGreen関数"""
        eps = np.finfo(float).eps
        constant = self.constant
        
        r_squared = (x - x_prime)**2 + (y - y_prime)**2
        r_squared = np.maximum(r_squared, eps)
        return -constant * np.log(r_squared)

    def _green_function_ms(self, x: np.ndarray, y:np.ndarray,
                       x_prime: np.ndarray, y_prime: np.ndarray) -> np.ndarray:
        """MS用Green関数"""
        eps = np.finfo(float).eps  # マシンイプシロンを使用することで安定性を大幅に工場
        epsilon = self.epsilon
        constant = self.constant

        x2 = (x - x_prime)**2

        # 本体
        r_squared = x2 + (y - y_prime)**2
        r_squared = np.maximum(r_squared, eps)
        G_direct = -constant * np.log(r_squared)

        # 影像
        r_image_squared = x2 + (y + y_prime)**2
        r_image_squared = np.maximum(r_image_squared, eps)
        G_image = constant * np.log(r_image_squared)

        return G_direct + G_image

    def _green_function_sp(self, x: np.ndarray, y:np.ndarray,
                       x_prime: np.ndarray, y_prime: np.ndarray) -> np.ndarray:
        """SP用Green関数"""
        epsilon = self.epsilon
        constant = self.constant
        h = self.height_top

        eps = np.finfo(float).eps  # マシンイプシロンを使用することで安定性を大幅に工場

        dx = x - x_prime
        dy = y - y_prime
        dy_plus = y + y_prime

        sinh_term = np.sinh(np.pi * dx / (2*h))**2
        sin_term_minus = np.sin(np.pi*dy/(2*h))**2
        sin_term_plus = np.sin(np.pi*dy_plus / (2*h))**2

        numerator = sinh_term + sin_term_plus
        denominator = np.maximum(sinh_term + sin_term_minus, eps)
        return constant * np.log(numerator / denominator)

    def line_integral_vec(self, x1: float, y1: float, x2: float, y2: float,
                        x1_prime: float, y1_prime: float, x2_prime: float, y2_prime: float) -> float:
        """線積分の数値解析処理 ベクトル化バージョン"""
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

        i_mesh, j_mesh = np.meshgrid(gauss_points, gauss_points)
        wi_mesh, wj_mesh = np.meshgrid(gauss_weights, gauss_weights)

        x_i = mid_x1 + (i_mesh - 0.5) * dx1
        y_i = mid_y1 + (i_mesh - 0.5) * dy1
        x_j = mid_x2 + (j_mesh - 0.5) * dx2
        y_j = mid_y2 + (j_mesh - 0.5) * dy2

        green_values = self.green_function(x_i, y_i, x_j, y_j)
        result = np.sum(wi_mesh * wj_mesh * green_values)

        dl1 = np.sqrt(dx1**2 + dy1**2)
        dl2 = np.sqrt(dx2**2 + dy2**2)
        return result * dl1 * dl2


    def _calculate_submatrix_wrapper(self, args):
        """マルチプロセス計算のためのラッパーメソッド"""
        i, j, N_points_i, N_points_j = args[0], args[1], args[2], args[3]
        conductor_i = self.conductors[i]
        conductor_j = self.conductors[j]
        sub_matrix = self._calculate_A_matrix(conductor_i, conductor_j)
        return (i, j, sub_matrix)

    def create_influence_matrix(self) -> np.array:
        """マルチプロセスを使用した影響係数行列の計算"""
        if self.cache_influence_matrix is None:
            total_points = sum(c['N_points'] for c in self.conductors)
            A = np.zeros((total_points, total_points))

            # 計算するサブマトリックスのリストを準備
            tasks = []
            for i, conductor_i in enumerate(self.conductors):
                for j in range(i, len(self.conductors)):
                    tasks.append((i, j, conductor_i['N_points'], self.conductors[j]['N_points']))

            # マルチプロセス計算
            with mp.get_context('spawn').Pool(processes=mp.cpu_count()) as pool:
                results = pool.map(self._calculate_submatrix_wrapper, tasks)

            # 結果を行列に反映
            current_row = 0
            for i, conductor_i in enumerate(self.conductors):
                current_col = current_row
                for j in range(i, len(self.conductors)):
                    for result in results:
                        if result[0] == i and result[1] == j:
                            sub_matrix = result[2]
                            if i == j:
                                A[current_row:current_row+conductor_i['N_points'],
                                  current_col:current_col+sub_matrix.shape[1]] = sub_matrix
                            else:
                                A[current_row:current_row+conductor_i['N_points'], 
                                  current_col:current_col+sub_matrix.shape[1]] = sub_matrix
                                A[current_col:current_col+sub_matrix.shape[1], 
                                  current_row:current_row+conductor_i['N_points']] = sub_matrix.T
                            current_col += sub_matrix.shape[1]
                            break
                current_row += conductor_i['N_points']

            self.cache_influence_matrix = A
        return self.cache_influence_matrix
    
    def _calculate_A_matrix(self, conductor_i: Dict, conductor_j: Dict) -> np.ndarray:
        """行列Aの計算"""
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

                len1 = np.sqrt((x2_mid - x1_mid)**2 + (y2_mid - y1_mid)**2)
                len2 = np.sqrt((x2_prime_mid - x1_prime_mid)**2 + (y2_prime_mid - y1_prime_mid)**2)

                sub_matrix[i, j] = self.line_integral_vec(
                    x1_mid, y1_mid, x2_mid, y2_mid,
                    x1_prime_mid, y1_prime_mid, x2_prime_mid, y2_prime_mid,
                ) / (len1 * len2)

        return sub_matrix

    # 電荷密度を計算する
    def solve_charge_density(self, voltages: List[float]) -> np.ndarray:
        """電荷密度を計算（GND以外の導体に対する電圧をリストで指定する）"""
        A = self.create_influence_matrix()
        v = np.zeros(sum(c['N_points'] for c in self.conductors))
        
        start_idx = 0
        voltage_idx = 0
        for conductor in self.conductors:
            # GND導体は0V固定、それ以外は指定電圧を使用
            voltage = 0.0 if conductor['is_gnd'] else voltages[voltage_idx]
            v[start_idx:start_idx + conductor['N_points']] = voltage
            
            if not conductor['is_gnd']:
                voltage_idx += 1
                
            start_idx += conductor['N_points']

        if self.cache_inverse_matrix is None:
            self.cache_inverse_matrix = linalg.inv(A)
        
        return self.cache_inverse_matrix @ v

    def calculate_capacitance_matrix_(self) -> np.ndarray:
        """容量行列を計算"""
        n_conductors = len(self.conductors)
        C = np.zeros((n_conductors, n_conductors))
        
        cumsum_points = np.cumsum([0] + [c['N_points'] for c in self.conductors])
        conversion_factor = 1
        
        for i in range(n_conductors):
            voltages = [1.0 if j == i else 0.0 for j in range(n_conductors)]
            q = self.solve_charge_density(voltages)
            
            for j in range(n_conductors):
                q_conductor = q[cumsum_points[j]:cumsum_points[j+1]]
                dl = self.conductors[j]['dl']
                total_charge = sum(q_conductor)
                C[j,i] = total_charge * conversion_factor
        
        return C

    def calculate_capacitance_matrix(self) -> np.ndarray:
        n_conductors = len(self.conductors)
        C_full = np.zeros((n_conductors, n_conductors))
        cumsum_points = np.cumsum([0] + [c['N_points'] for c in self.conductors])
        
        # すべての導体で計算
        for i in range(n_conductors):
            voltages = [1.0 if j == i else 0.0 for j in range(n_conductors)]
            
            # GNDは0V固定なので、solve_charge_densityには信号導体の電圧のみ渡す
            signal_voltages = [v for j, v in enumerate(voltages) 
                            if not self.conductors[j]['is_gnd']]
            q = self.solve_charge_density(signal_voltages)
            
            for j in range(n_conductors):
                q_conductor = q[cumsum_points[j]:cumsum_points[j+1]]
                dl = self.conductors[j]['dl']
                total_charge = np.sum(q_conductor)
                C_full[j,i] = total_charge

        # 信号導体の部分だけを抽出
        signal_indices = [i for i, c in enumerate(self.conductors) if not c['is_gnd']]
        C = C_full[np.ix_(signal_indices, signal_indices)]
        
        return C


    def calculate_inductance_matrix(self) -> np.ndarray:
        """インダクタンス行列を計算"""
        C = self.calculate_capacitance_matrix()
        C_inv = np.linalg.inv(C)  # 容量行列の逆行列
        L = self.epsilon_r * C_inv / self.c_squared
        return L

    def calculate_z_matrix(self) -> np.ndarray:
        """特性インピーダンス行列を計算"""
        C = self.calculate_capacitance_matrix()
        L = self.calculate_inductance_matrix()
        C_inv = np.linalg.inv(C)
        Z0 = np.sqrt(np.dot(L, C_inv))  # 行列の平方根
        return Z0
