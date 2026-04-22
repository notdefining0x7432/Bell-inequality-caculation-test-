"""
贝尔不等式量子最大值计算 - Qutip 直接计算法
============================================

技术路线：
1. 选择量子态 |ψ⟩（GHZ 态、W 态、最大纠缠态等）
2. 定义测量算符 M_x（Pauli 矩阵、旋转测量等）
3. 计算期望值 ⟨ψ| M_{x1} ⊗ M_{x2} ⊗ ... ⊗ M_{xn} |ψ⟩
4. 对所有关联函数加权求和
5. 优化测量设置以找到最大值

优点：
- 直观，易于理解
- 可以找到具体的量子实现
- 计算速度快

缺点：
- 只能找到下界（实际量子值可能更高）
- 需要猜测最优态和测量

作者：kunpengwu
日期：2026-04-22
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from itertools import product
import warnings

# 尝试导入 Qutip
try:
    from qutip import basis, tensor, sigmax, sigmay, sigmaz, expect, Qobj

    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False
    warnings.warn("Qutip 未安装，部分功能不可用。请运行: pip install qutip")


# =============================================================================
# 第一部分：量子态创建
# =============================================================================

def create_ghz_state(n_qubits: int):
    """
    创建 n 量子比特 GHZ 态

    GHZ 态: |GHZ⟩ = (|00...0⟩ + |11...1⟩) / √2

    这是最常用的多方纠缠态，对于 Mermin 类不等式可以达到量子最大值。

    参数:
    - n_qubits: 量子比特数

    返回:
    - Qutip Qobj: GHZ 态
    """
    if not QUTIP_AVAILABLE:
        raise ImportError("Qutip 未安装")

    state_0 = tensor([basis(2, 0) for _ in range(n_qubits)])
    state_1 = tensor([basis(2, 1) for _ in range(n_qubits)])
    return (state_0 + state_1).unit()


def create_w_state(n_qubits: int):
    """
    创建 n 量子比特 W 态

    W 态: |W⟩ = (|100...0⟩ + |010...0⟩ + ... + |00...01⟩) / √n

    参数:
    - n_qubits: 量子比特数

    返回:
    - Qutip Qobj: W 态
    """
    if not QUTIP_AVAILABLE:
        raise ImportError("Qutip 未安装")

    states = []
    for i in range(n_qubits):
        components = [basis(2, 1) if j == i else basis(2, 0)
                      for j in range(n_qubits)]
        states.append(tensor(components))
    return sum(states).unit()


def create_maximally_entangled_state():
    """
    创建两量子比特最大纠缠态 (Bell 态)

    |Φ⁺⟩ = (|00⟩ + |11⟩) / √2

    这是 CHSH 不等式的最优态。

    返回:
    - Qutip Qobj: Bell 态
    """
    if not QUTIP_AVAILABLE:
        raise ImportError("Qutip 未安装")

    return (tensor(basis(2, 0), basis(2, 0)) +
            tensor(basis(2, 1), basis(2, 1))).unit()


# =============================================================================
# 第二部分：测量算符创建
# =============================================================================

def get_pauli_operators():
    """
    获取 Pauli 算符

    返回:
    - 字典: {'sigma_x': σx, 'sigma_y': σy, 'sigma_z': σz}
    """
    if not QUTIP_AVAILABLE:
        raise ImportError("Qutip 未安装")

    return {
        'sigma_x': sigmax(),
        'sigma_y': sigmay(),
        'sigma_z': sigmaz()
    }


def create_bloch_measurement(theta: float, phi: float = 0):
    """
    创建 Bloch 球上的测量算符

    M(θ, φ) = cos(θ)·σx + sin(θ)·cos(φ)·σy + sin(θ)·sin(φ)·σz

    参数:
    - theta: 极角（从 x 轴测量）
    - phi: 方位角（在 yz 平面内）

    返回:
    - Qutip Qobj: 测量算符
    """
    if not QUTIP_AVAILABLE:
        raise ImportError("Qutip 未安装")

    return (np.cos(theta) * sigmax() +
            np.sin(theta) * np.cos(phi) * sigmay() +
            np.sin(theta) * np.sin(phi) * sigmaz())


def create_planar_measurement(theta: float):
    """
    创建平面测量算符（在 xy 平面内）

    M(θ) = cos(θ)·σx + sin(θ)·σy

    这是 CHSH 类不等式的最优测量形式。

    参数:
    - theta: 角度

    返回:
    - Qutip Qobj: 测量算符
    """
    if not QUTIP_AVAILABLE:
        raise ImportError("Qutip 未安装")

    return np.cos(theta) * sigmax() + np.sin(theta) * sigmay()


# =============================================================================
# 第三部分：贝尔不等式表示类
# =============================================================================

class BellInequality:
    """
    贝尔不等式通用表示类

    使用示例:
    >>> ineq = BellInequality(n_parties=2, measurements=[2, 2], name="CHSH")
    >>> ineq.set_coefficients({
    ...     (0, 0): 1, (0, 1): 1,
    ...     (1, 0): 1, (1, 1): -1,
    ... })
    """

    def __init__(self,
                 n_parties: int,
                 measurements: List[int],
                 outputs: List[int] = None,
                 name: str = "Bell Inequality"):
        self.n_parties = n_parties
        self.measurements = measurements
        self.outputs = outputs if outputs else [2] * n_parties
        self.name = name
        self.coefficients = {}

        if len(measurements) != n_parties:
            raise ValueError("measurements 长度必须等于 n_parties")

    def set_coefficients(self, coefficients: Dict[Tuple[int, ...], float]):
        """设置关联函数系数"""
        for settings in coefficients:
            if len(settings) != self.n_parties:
                raise ValueError(f"设置元组长度不等于参与方数")
        self.coefficients = coefficients.copy()

    def get_expression_string(self) -> str:
        """获取不等式的字符串表示"""
        if not self.coefficients:
            return "空不等式"

        party_names = [chr(65 + i) for i in range(self.n_parties)]

        terms = []
        for settings, coef in sorted(self.coefficients.items()):
            sign = '+' if coef >= 0 else '-'
            abs_coef = abs(coef)

            if abs_coef == 1:
                term = f"⟨{'·'.join([f'{party_names[i]}_{settings[i] + 1}' for i in range(self.n_parties)])}⟩"
            else:
                term = f"{abs_coef}·⟨{'·'.join([f'{party_names[i]}_{settings[i] + 1}' for i in range(self.n_parties)])}⟩"

            terms.append(f"{sign}{term}")

        expr = ' '.join(terms)
        if expr.startswith('+'):
            expr = expr[1:]
        return expr.strip()


# =============================================================================
# 第四部分：经典和无信号最大值计算
# =============================================================================

def compute_classical_max(ineq: BellInequality) -> float:
    """
    计算经典最大值

    通过穷举所有确定性策略计算经典最大值。
    对于二元输出，确定性策略对应于每个测量返回固定的 ±1 值。
    """
    max_val = -np.inf

    def generate_strategies(party_idx: int, current_strategy: Dict):
        if party_idx == ineq.n_parties:
            yield current_strategy.copy()
            return

        for outputs_combo in product([0, 1], repeat=ineq.measurements[party_idx]):
            current_strategy[party_idx] = outputs_combo
            yield from generate_strategies(party_idx + 1, current_strategy)

    for strategy in generate_strategies(0, {}):
        value = 0.0
        for settings, coef in ineq.coefficients.items():
            product_val = 1
            for party in range(ineq.n_parties):
                m = settings[party]
                output = strategy[party][m]
                product_val *= (1 - 2 * output)
            value += coef * product_val
        max_val = max(max_val, value)

    return float(max_val)


def compute_nosignal_max(ineq: BellInequality) -> float:
    """
    计算无信号最大值

    对于完整关联不等式，无信号最大值等于系数绝对值之和。
    """
    return sum(abs(coef) for coef in ineq.coefficients.values())


# =============================================================================
# 第五部分：Qutip 求解器
# =============================================================================

class QutipSolver:
    """
    Qutip 求解器

    使用示例:
    >>> solver = QutipSolver(ineq)
    >>> result = solver.solve(state_type='ghz', optimize=True)
    """

    def __init__(self, ineq: BellInequality):
        if not QUTIP_AVAILABLE:
            raise ImportError("Qutip 未安装，请运行: pip install qutip")
        self.ineq = ineq

    def solve(self,
              state_type: str = 'ghz',
              optimize: bool = False,
              n_random: int = 1000) -> Dict:
        """
        求解贝尔不等式

        参数:
        - state_type: 量子态类型 ('ghz', 'w', 'maximally_entangled')
        - optimize: 是否优化测量设置
        - n_random: 随机采样数

        返回:
        - 结果字典
        """
        # 创建量子态
        state = self._create_state(state_type)

        # 计算量子值
        if optimize:
            quantum_max = self._optimize_measurements(state, n_random)
        else:
            quantum_max = self._compute_with_pauli(state)

        return {
            'classical_max': compute_classical_max(self.ineq),
            'quantum_max': quantum_max,
            'nosignal_max': compute_nosignal_max(self.ineq)
        }

    def _create_state(self, state_type: str):
        """创建量子态"""
        if state_type == 'ghz':
            return create_ghz_state(self.ineq.n_parties)
        elif state_type == 'w':
            return create_w_state(self.ineq.n_parties)
        elif state_type == 'maximally_entangled':
            if self.ineq.n_parties != 2:
                raise ValueError("maximally_entangled 仅支持两方")
            return create_maximally_entangled_state()
        else:
            raise ValueError(f"未知态类型: {state_type}")

    def _compute_with_pauli(self, state) -> float:
        """使用 Pauli 测量计算量子值"""
        measurement_sets = {}
        pauli_ops = [sigmax(), sigmay()]

        for party in range(self.ineq.n_parties):
            measurement_sets[party] = {}
            for m in range(self.ineq.measurements[party]):
                measurement_sets[party][m] = pauli_ops[m % 2]

        value = 0.0
        for settings, coef in self.ineq.coefficients.items():
            ops = [measurement_sets[party][settings[party]]
                   for party in range(self.ineq.n_parties)]
            measurement = tensor(ops)
            expectation = expect(measurement, state)
            value += coef * expectation

        return float(value)

    def _optimize_measurements(self, state, n_random: int) -> float:
        """优化测量设置"""
        n_measurements = sum(self.ineq.measurements)
        max_val = -np.inf

        for _ in range(n_random):
            angles = np.random.uniform(0, 2 * np.pi, n_measurements)

            measurement_sets = {}
            idx = 0
            for party in range(self.ineq.n_parties):
                measurement_sets[party] = {}
                for m in range(self.ineq.measurements[party]):
                    measurement_sets[party][m] = create_planar_measurement(angles[idx])
                    idx += 1

            value = 0.0
            for settings, coef in self.ineq.coefficients.items():
                ops = [measurement_sets[party][settings[party]]
                       for party in range(self.ineq.n_parties)]
                measurement = tensor(ops)
                expectation = expect(measurement, state)
                value += coef * expectation

            max_val = max(max_val, value)

        return float(max_val)


# =============================================================================
# 第六部分：预定义不等式
# =============================================================================

def create_chsh_inequality() -> BellInequality:
    """创建 CHSH 不等式"""
    ineq = BellInequality(n_parties=2, measurements=[2, 2], name="CHSH")
    ineq.set_coefficients({
        (0, 0): 1, (0, 1): 1,
        (1, 0): 1, (1, 1): -1,
    })
    return ineq


def create_chain_inequality(n: int) -> BellInequality:
    """创建链不等式 Cn"""
    ineq = BellInequality(n_parties=2, measurements=[n, n], name=f"Chain C{n}")

    coefficients = {}
    for i in range(n - 1):
        coefficients[(i, i)] = 1
        coefficients[(i + 1, i)] = 1
    coefficients[(n - 1, n - 1)] = 1
    coefficients[(0, n - 1)] = -1

    ineq.set_coefficients(coefficients)
    return ineq


def create_mermin_inequality(n: int = 3) -> BellInequality:
    """创建 Mermin 不等式 Mn"""
    ineq = BellInequality(n_parties=n, measurements=[2] * n, name=f"Mermin M{n}")

    if n == 3:
        coefficients = {
            (0, 0, 0): 1,
            (0, 1, 1): -1,
            (1, 0, 1): -1,
            (1, 1, 0): -1,
        }
    else:
        coefficients = {}
        for settings in product([0, 1], repeat=n):
            parity = sum(settings)
            if parity == 0:
                coefficients[settings] = 1
            elif parity == n:
                coefficients[settings] = (-1) ** ((n - 1) // 2)

    ineq.set_coefficients(coefficients)
    return ineq


def create_mermin_user_form() -> BellInequality:
    """创建用户形式的 M3 不等式"""
    ineq = BellInequality(n_parties=3, measurements=[2, 2, 2], name="M3 User Form")
    ineq.set_coefficients({
        (1, 0, 0): 1,
        (0, 1, 0): 1,
        (0, 0, 1): 1,
        (1, 1, 1): -1,
    })
    return ineq


# =============================================================================
# 第七部分：测试
# =============================================================================

def run_tests():
    """运行测试"""
    print("=" * 60)
    print("Qutip 方法测试")
    print("=" * 60)

    # 测试 1: CHSH
    print("\n测试 1: CHSH 不等式")
    print("-" * 40)

    chsh = create_chsh_inequality()
    print(f"不等式: {chsh.get_expression_string()}")

    solver = QutipSolver(chsh)
    result = solver.solve(state_type='maximally_entangled', optimize=True, n_random=2000)

    print(f"经典最大值: {result['classical_max']:.6f} (理论值: 2)")
    print(f"量子最大值: {result['quantum_max']:.6f} (理论值: 2√2 ≈ 2.828)")
    print(f"无信号最大值: {result['nosignal_max']:.6f} (理论值: 4)")

    # 验证最优测量
    print("\n验证 CHSH 最优测量设置:")
    bell = create_maximally_entangled_state()

    # 最优测量: A1=σz, A2=σx, B1=(σz+σx)/√2, B2=(σz-σx)/√2
    A1 = sigmaz()
    A2 = sigmax()
    B1 = (sigmaz() + sigmax()) / np.sqrt(2)
    B2 = (sigmaz() - sigmax()) / np.sqrt(2)

    E11 = expect(tensor(A1, B1), bell)
    E12 = expect(tensor(A1, B2), bell)
    E21 = expect(tensor(A2, B1), bell)
    E22 = expect(tensor(A2, B2), bell)

    chsh_value = E11 + E12 + E21 - E22
    print(f"使用最优测量: CHSH = {chsh_value:.6f} (理论值: 2√2 ≈ {2 * np.sqrt(2):.6f})")

    # 测试 2: Mermin M3
    print("\n测试 2: Mermin M3 不等式")
    print("-" * 40)

    mermin = create_mermin_inequality(3)
    print(f"不等式: {mermin.get_expression_string()}")

    solver = QutipSolver(mermin)
    result = solver.solve(state_type='ghz', optimize=False)

    print(f"经典最大值: {result['classical_max']:.6f} (理论值: 2)")
    print(f"量子最大值: {result['quantum_max']:.6f} (理论值: 4)")
    print(f"无信号最大值: {result['nosignal_max']:.6f} (理论值: 4)")

    # 测试 3: 用户形式 M3
    print("\n测试 3: 用户形式 M3 不等式")
    print("-" * 40)

    m3_user = create_mermin_user_form()
    print(f"不等式: {m3_user.get_expression_string()}")

    solver = QutipSolver(m3_user)
    result = solver.solve(state_type='ghz', optimize=True, n_random=2000)

    print(f"经典最大值: {result['classical_max']:.6f}")
    print(f"量子最大值 (优化后): {result['quantum_max']:.6f}")
    print(f"无信号最大值: {result['nosignal_max']:.6f}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
