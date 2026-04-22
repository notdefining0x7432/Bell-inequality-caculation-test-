"""
贝尔不等式量子最大值计算 - 统一接口
===================================

整合 Qutip 直接计算法和 NPA 层级法，提供统一的计算接口。

推荐使用方法：
1. 两方不等式 → 使用 toqito 方法
2. 多方不等式 → 使用 Qutip 方法

作者：kunpengwu
日期：2026-04-22
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from itertools import product
import warnings

# 尝试导入依赖
try:
    from qutip import basis, tensor, sigmax, sigmay, sigmaz, expect

    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False

try:
    from toqito.state_opt.bell_inequality_max import bell_inequality_max

    TOQITO_AVAILABLE = True
except ImportError:
    TOQITO_AVAILABLE = False


# =============================================================================
# 贝尔不等式表示类
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
        """
        初始化贝尔不等式

        参数:
        - n_parties: 参与方数量
        - measurements: 每方的测量设置数列表
        - outputs: 每方的输出数列表（默认全为 2）
        - name: 不等式名称
        """
        self.n_parties = n_parties
        self.measurements = measurements
        self.outputs = outputs if outputs else [2] * n_parties
        self.name = name
        self.coefficients = {}

        if len(measurements) != n_parties:
            raise ValueError("measurements 长度必须等于 n_parties")

    def set_coefficients(self, coefficients: Dict[Tuple[int, ...], float]):
        """
        设置关联函数系数

        参数:
        - coefficients: 字典，键为测量设置的元组，值为系数

        示例:
        >>> ineq.set_coefficients({
        ...     (0, 0): 1,   # ⟨A1 B1⟩ 的系数
        ...     (0, 1): 1,   # ⟨A1 B2⟩ 的系数
        ...     (1, 0): 1,   # ⟨A2 B1⟩ 的系数
        ...     (1, 1): -1,  # ⟨A2 B2⟩ 的系数
        ... })
        """
        for settings in coefficients:
            if len(settings) != self.n_parties:
                raise ValueError(f"设置元组长度 {len(settings)} 不等于参与方数 {self.n_parties}")
            for i, s in enumerate(settings):
                if s < 0 or s >= self.measurements[i]:
                    raise ValueError(f"设置 {settings} 中第 {i} 方的测量索引 {s} 超出范围")

        self.coefficients = coefficients.copy()

    def get_expression_string(self) -> str:
        """获取不等式的字符串表示"""
        if not self.coefficients:
            return "空不等式"

        party_names = [chr(65 + i) for i in range(self.n_parties)]  # A, B, C, ...

        terms = []
        for settings, coef in sorted(self.coefficients.items()):
            sign = '+' if coef >= 0 else '-'
            abs_coef = abs(coef)

            # 构建关联函数字符串
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
# 经典和无信号最大值计算
# =============================================================================

def compute_classical_max(ineq: BellInequality) -> float:
    """
    计算经典最大值

    通过穷举所有确定性策略计算经典最大值。
    对于二元输出，确定性策略对应于每个测量返回固定的 ±1 值。

    参数:
    - ineq: BellInequality 对象

    返回:
    - 经典最大值
    """
    max_val = -np.inf

    def generate_strategies(party_idx: int, current_strategy: Dict):
        """递归生成所有确定性策略"""
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
                product_val *= (1 - 2 * output)  # 0 -> +1, 1 -> -1
            value += coef * product_val
        max_val = max(max_val, value)

    return float(max_val)


def compute_nosignal_max(ineq: BellInequality) -> float:
    """
    计算无信号最大值

    对于完整关联不等式，无信号最大值等于系数绝对值之和。

    参数:
    - ineq: BellInequality 对象

    返回:
    - 无信号最大值
    """
    return sum(abs(coef) for coef in ineq.coefficients.values())


# =============================================================================
# Qutip 方法
# =============================================================================

def create_ghz_state(n_qubits: int):
    """创建 GHZ 态"""
    if not QUTIP_AVAILABLE:
        raise ImportError("Qutip 未安装")

    state_0 = tensor([basis(2, 0) for _ in range(n_qubits)])
    state_1 = tensor([basis(2, 1) for _ in range(n_qubits)])
    return (state_0 + state_1).unit()


def create_maximally_entangled_state():
    """创建 Bell 态"""
    if not QUTIP_AVAILABLE:
        raise ImportError("Qutip 未安装")

    return (tensor(basis(2, 0), basis(2, 0)) +
            tensor(basis(2, 1), basis(2, 1))).unit()


def create_planar_measurement(theta: float):
    """创建平面测量算符: cos(θ)σx + sin(θ)σy"""
    if not QUTIP_AVAILABLE:
        raise ImportError("Qutip 未安装")

    return np.cos(theta) * sigmax() + np.sin(theta) * sigmay()


def solve_with_qutip(ineq: BellInequality,
                     state_type: str = 'ghz',
                     optimize: bool = False,
                     n_random: int = 1000) -> Dict:
    """
    使用 Qutip 直接计算法求解

    参数:
    - ineq: BellInequality 对象
    - state_type: 量子态类型 ('ghz' 或 'maximally_entangled')
    - optimize: 是否优化测量设置
    - n_random: 随机采样数

    返回:
    - 结果字典
    """
    if not QUTIP_AVAILABLE:
        raise ImportError("Qutip 未安装，请运行: pip install qutip")

    # 创建量子态
    if state_type == 'ghz':
        state = create_ghz_state(ineq.n_parties)
    elif state_type == 'maximally_entangled':
        if ineq.n_parties != 2:
            raise ValueError("maximally_entangled 仅支持两方")
        state = create_maximally_entangled_state()
    else:
        raise ValueError(f"未知态类型: {state_type}")

    if optimize:
        # 优化测量设置
        max_val = -np.inf
        n_measurements = sum(ineq.measurements)

        for _ in range(n_random):
            # 随机生成测量角度
            angles = np.random.uniform(0, 2 * np.pi, n_measurements)

            # 为每方的每个测量分配角度
            measurement_sets = {}
            idx = 0
            for party in range(ineq.n_parties):
                measurement_sets[party] = {}
                for m in range(ineq.measurements[party]):
                    measurement_sets[party][m] = create_planar_measurement(angles[idx])
                    idx += 1

            # 计算不等式值
            value = 0.0
            for settings, coef in ineq.coefficients.items():
                ops = [measurement_sets[party][settings[party]]
                       for party in range(ineq.n_parties)]
                measurement = tensor(ops)
                expectation = expect(measurement, state)
                value += coef * expectation

            max_val = max(max_val, value)

        return {'quantum_max': float(max_val)}
    else:
        # 使用 Pauli 测量
        measurement_sets = {}
        pauli_ops = [sigmax(), sigmay()]

        for party in range(ineq.n_parties):
            measurement_sets[party] = {}
            for m in range(ineq.measurements[party]):
                measurement_sets[party][m] = pauli_ops[m % 2]

        # 计算不等式值
        value = 0.0
        for settings, coef in ineq.coefficients.items():
            ops = [measurement_sets[party][settings[party]]
                   for party in range(ineq.n_parties)]
            measurement = tensor(ops)
            expectation = expect(measurement, state)
            value += coef * expectation

        return {'quantum_max': float(value)}


# =============================================================================
# toqito 方法
# =============================================================================

def solve_with_toqito(ineq: BellInequality, k: Union[int, str] = 1) -> Dict:
    """
    使用 toqito 库求解（仅支持两方）

    参数:
    - ineq: BellInequality 对象
    - k: NPA 层级

    返回:
    - 结果字典
    """
    if not TOQITO_AVAILABLE:
        raise ImportError("toqito 未安装，请运行: pip install toqito==1.1.4")

    if ineq.n_parties != 2:
        raise ValueError("toqito 仅支持两方不等式")

    # 构建 FC 系数矩阵
    ma, mb = ineq.measurements
    M_fc = np.zeros((ma + 1, mb + 1))

    for settings, coef in ineq.coefficients.items():
        i, j = settings
        M_fc[i + 1, j + 1] = coef

    # desc 参数: [oa, ob, ma, mb]
    desc = [2, 2, ma, mb]

    # 计算
    classical = bell_inequality_max(M_fc, desc, 'fc', 'classical')
    quantum = bell_inequality_max(M_fc, desc, 'fc', 'quantum', k=k)
    nosignal = bell_inequality_max(M_fc, desc, 'fc', 'nosignal')

    return {
        'classical_max': classical,
        'quantum_max': quantum,
        'nosignal_max': nosignal
    }


# =============================================================================
# 统一接口
# =============================================================================

def solve(ineq: BellInequality,
          method: str = 'auto',
          state_type: str = 'ghz',
          optimize: bool = False,
          k: Union[int, str] = 1,
          n_random: int = 1000) -> Dict:
    """
    统一求解接口

    参数:
    - ineq: BellInequality 对象
    - method: 计算方法 ('auto', 'toqito', 'qutip')
    - state_type: 量子态类型
    - optimize: 是否优化测量设置
    - k: NPA 层级
    - n_random: 随机采样数

    返回:
    - 结果字典
    """
    result = {
        'classical_max': None,
        'quantum_max': None,
        'nosignal_max': None,
        'method_used': None
    }

    # 计算经典和无信号最大值
    result['classical_max'] = compute_classical_max(ineq)
    result['nosignal_max'] = compute_nosignal_max(ineq)

    # 选择方法
    if method == 'auto':
        if ineq.n_parties == 2 and TOQITO_AVAILABLE:
            method = 'toqito'
        elif QUTIP_AVAILABLE:
            method = 'qutip'
        else:
            raise ImportError("没有可用的计算方法，请安装 qutip 或 toqito")

    result['method_used'] = method

    # 计算
    if method == 'toqito':
        toqito_result = solve_with_toqito(ineq, k)
        result['quantum_max'] = toqito_result['quantum_max']
    elif method == 'qutip':
        qutip_result = solve_with_qutip(ineq, state_type, optimize, n_random)
        result['quantum_max'] = qutip_result['quantum_max']
    else:
        raise ValueError(f"未知方法: {method}")

    return result


# =============================================================================
# 预定义不等式
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
# 测试
# =============================================================================

def run_tests():
    """运行测试"""
    print("=" * 60)
    print("贝尔不等式量子最大值计算 - 测试")
    print("=" * 60)

    # 测试 1: CHSH
    print("\n测试 1: CHSH 不等式")
    print("-" * 40)

    chsh = create_chsh_inequality()
    print(f"不等式: {chsh.get_expression_string()}")

    result = solve(chsh)

    print(f"经典最大值: {result['classical_max']:.6f} (理论值: 2)")
    print(f"量子最大值: {result['quantum_max']:.6f} (理论值: 2√2 ≈ 2.828)")
    print(f"无信号最大值: {result['nosignal_max']:.6f} (理论值: 4)")
    print(f"使用方法: {result['method_used']}")

    # 测试 2: 链不等式 C3
    print("\n测试 2: 链不等式 C3")
    print("-" * 40)

    chain = create_chain_inequality(3)
    print(f"不等式: {chain.get_expression_string()}")

    result = solve(chain)

    print(f"经典最大值: {result['classical_max']:.6f} (理论值: 4)")
    print(f"量子最大值: {result['quantum_max']:.6f} (理论值: 5.196152)")
    print(f"无信号最大值: {result['nosignal_max']:.6f} (理论值: 6)")
    print(f"使用方法: {result['method_used']}")

    # 测试 3: Mermin M3
    print("\n测试 3: Mermin M3 不等式")
    print("-" * 40)

    mermin = create_mermin_inequality(3)
    print(f"不等式: {mermin.get_expression_string()}")

    result = solve(mermin)

    print(f"经典最大值: {result['classical_max']:.6f} (理论值: 2)")
    print(f"量子最大值: {result['quantum_max']:.6f} (理论值: 4)")
    print(f"无信号最大值: {result['nosignal_max']:.6f} (理论值: 4)")
    print(f"使用方法: {result['method_used']}")

    # 测试 4: 用户形式 M3
    print("\n测试 4: 用户形式 M3 不等式")
    print("-" * 40)

    m3_user = create_mermin_user_form()
    print(f"不等式: {m3_user.get_expression_string()}")

    result = solve(m3_user, optimize=True, n_random=2000)

    print(f"经典最大值: {result['classical_max']:.6f}")
    print(f"量子最大值: {result['quantum_max']:.6f}")
    print(f"无信号最大值: {result['nosignal_max']:.6f}")
    print(f"使用方法: {result['method_used']}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


USAGE = """
================================================================================
                    贝尔不等式量子最大值计算 - 使用说明
================================================================================

一、快速开始
------------

from bell_inequality_unified import *

# 创建不等式
ineq = BellInequality(n_parties=2, measurements=[2, 2], name="CHSH")
ineq.set_coefficients({
    (0, 0): 1, (0, 1): 1,
    (1, 0): 1, (1, 1): -1,
})

# 求解
result = solve(ineq)

print(f"经典最大值: {result['classical_max']}")
print(f"量子最大值: {result['quantum_max']}")

二、方法选择
------------

- method='auto': 自动选择（推荐）
- method='toqito': 使用 toqito 库（仅两方）
- method='qutip': 使用 Qutip 直接计算（任意多方）

三、参数说明
------------

- state_type: 量子态类型 ('ghz', 'maximally_entangled')
- optimize: 是否优化测量设置 (True/False)
- k: NPA 层级 (用于 toqito 方法)
- n_random: 随机采样数（用于优化）

四、预定义不等式
----------------

- create_chsh_inequality(): CHSH 不等式
- create_chain_inequality(n): 链不等式 Cn
- create_mermin_inequality(3): Mermin 不等式 M3
- create_mermin_user_form(): 用户形式 M3

================================================================================
"""


def print_usage():
    """打印使用说明"""
    print(USAGE)


if __name__ == "__main__":
    run_tests()
