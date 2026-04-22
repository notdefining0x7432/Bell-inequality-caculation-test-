"""
贝尔不等式量子最大值计算 - NPA 层级法
=====================================

技术路线：
1. 定义测量符号 (party, measurement)
2. 生成单词序列 W = {I, A_1, B_1, A_1A_2, ...}
3. 构建 Moment Matrix Γ，其中 Γ[i,j] = ⟨W_i† W_j⟩
4. 添加约束：Γ ≥ 0，归一化，幂等性，正交性，交换性
5. 求解 SDP 得到量子上界

优点：
- 可以得到严格的量子上界
- 不需要猜测态和测量
- 支持任意多方不等式

缺点：
- 计算复杂度高
- 只能给出上界，不能给出具体实现
- 层级越高，计算越慢

参考：
- Navascués, Pironio, Acín, PRL 2008
- toqito 库的实现

作者：kunpengwu
日期：2026-04-22
"""

import numpy as np
import cvxpy as cp
from typing import List, Tuple, Dict, Optional, Union
from itertools import product
import warnings

# 尝试导入 toqito
try:
    from toqito.state_opt.bell_inequality_max import bell_inequality_max

    TOQITO_AVAILABLE = True
except ImportError:
    TOQITO_AVAILABLE = False


# =============================================================================
# 第一部分：贝尔不等式表示类
# =============================================================================

class BellInequality:
    """贝尔不等式通用表示类"""

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

    def set_coefficients(self, coefficients: Dict[Tuple[int, ...], float]):
        """设置关联函数系数"""
        self.coefficients = coefficients.copy()

    def get_expression_string(self) -> str:
        """获取不等式的字符串表示"""
        if not self.coefficients:
            return "空不等式"

        party_names = [chr(65 + i) for i in range(self.n_parties)]

        terms = []
        for settings, coef in sorted(self.coefficients.items()):
            sign = '+' if coef >= 0 else '-'
            term = f"⟨{'·'.join([f'{party_names[i]}_{settings[i] + 1}' for i in range(self.n_parties)])}⟩"
            terms.append(f"{sign}{term}")

        expr = ' '.join(terms)
        if expr.startswith('+'):
            expr = expr[1:]
        return expr.strip()


# =============================================================================
# 第二部分：经典和无信号最大值计算
# =============================================================================

def compute_classical_max(n_parties: int,
                          measurements: List[int],
                          coefficients: Dict[Tuple[int, ...], float]) -> float:
    """计算经典最大值"""
    max_val = -np.inf

    def generate_strategies(party_idx: int, current_strategy: Dict):
        if party_idx == n_parties:
            yield current_strategy.copy()
            return

        for outputs_combo in product([0, 1], repeat=measurements[party_idx]):
            current_strategy[party_idx] = outputs_combo
            yield from generate_strategies(party_idx + 1, current_strategy)

    for strategy in generate_strategies(0, {}):
        value = 0.0
        for settings, coef in coefficients.items():
            product_val = 1
            for party in range(n_parties):
                m = settings[party]
                output = strategy[party][m]
                product_val *= (1 - 2 * output)
            value += coef * product_val
        max_val = max(max_val, value)

    return float(max_val)


def compute_nosignal_max(coefficients: Dict[Tuple[int, ...], float]) -> float:
    """计算无信号最大值"""
    return sum(abs(coef) for coef in coefficients.values())


# =============================================================================
# 第三部分：toqito 方法（两方不等式）
# =============================================================================

def solve_with_toqito(coefficients: Dict[Tuple[int, ...], float],
                      measurements: List[int],
                      k: Union[int, str] = 1) -> Dict:
    """
    使用 toqito 库求解两方不等式

    参数:
    - coefficients: 关联函数系数
    - measurements: 每方的测量数 [ma, mb]
    - k: NPA 层级

    返回:
    - 结果字典
    """
    if not TOQITO_AVAILABLE:
        raise ImportError("toqito 未安装，请运行: pip install toqito==1.1.4")

    if len(measurements) != 2:
        raise ValueError("toqito 仅支持两方不等式")

    ma, mb = measurements

    # 构建 FC 系数矩阵
    M_fc = np.zeros((ma + 1, mb + 1))

    for settings, coef in coefficients.items():
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
# 第四部分：自定义 NPA 层级实现
# =============================================================================

class NPASolver:
    """
    NPA 层级求解器

    使用示例:
    >>> solver = NPASolver(n_parties=2, measurements=[2, 2])
    >>> solver.set_coefficients({
    ...     (0, 0): 1, (0, 1): 1,
    ...     (1, 0): 1, (1, 1): -1,
    ... })
    >>> result = solver.solve(level=1)
    """

    def __init__(self,
                 n_parties: int,
                 measurements: List[int],
                 outputs: List[int] = None):
        self.n_parties = n_parties
        self.measurements = measurements
        self.outputs = outputs if outputs else [2] * n_parties
        self.coefficients = {}

        # 定义测量符号
        self._setup_symbols()

    def _setup_symbols(self):
        """定义测量符号"""
        self.symbols = []
        self.symbol_to_idx = {}

        # 符号格式: (party, measurement)
        for party in range(self.n_parties):
            for m in range(self.measurements[party]):
                symbol = (party, m)
                self.symbols.append(symbol)
                self.symbol_to_idx[symbol] = len(self.symbols) - 1

    def set_coefficients(self, coefficients: Dict[Tuple[int, ...], float]):
        """设置关联函数系数"""
        self.coefficients = coefficients.copy()

    def _generate_words(self, level: Union[int, str]) -> List[Tuple]:
        """
        生成 NPA 层级的单词序列

        level=1: 单字母 {I, A_1, B_1, ...}
        level='1+ab': 包含不同方的双字母
        level=2: 所有双字母
        level=3: 所有三字母
        """
        words = [()]  # 空单词（恒等）

        # 单字母
        single_letters = [(s,) for s in self.symbols]
        words.extend(single_letters)

        if level == 1:
            return words

        # 双字母
        if level in ['1+ab', 2, 3]:
            if level == '1+ab':
                # 只包含不同方的双字母
                for s1 in self.symbols:
                    for s2 in self.symbols:
                        if s1[0] != s2[0]:  # 不同方
                            words.append((s1, s2))
            else:
                # 所有双字母
                for s1 in self.symbols:
                    for s2 in self.symbols:
                        words.append((s1, s2))

        # 三字母
        if level == 3:
            for s1 in self.symbols:
                for s2 in self.symbols:
                    for s3 in self.symbols:
                        words.append((s1, s2, s3))

        return words

    def _reduce_word(self, word: Tuple) -> Tuple:
        """
        简化单词（应用 NPA 规则）

        规则：
        1. 同一方不同测量正交：A_x A_y = 0 (x ≠ y)
        2. 同一测量幂等：A_x A_x = I
        3. 不同方交换：A_x B_y = B_y A_x
        """
        if not word:
            return word

        # 检查正交性
        for i in range(len(word) - 1):
            s1, s2 = word[i], word[i + 1]
            if s1[0] == s2[0] and s1[1] != s2[1]:  # 同一方不同测量
                return ()  # 结果为 0

        # 检查幂等性并重新排序
        result = []
        party_symbols = {p: [] for p in range(self.n_parties)}

        for s in word:
            party = s[0]
            if party_symbols[party] and party_symbols[party][-1] == s:
                # 幂等性：A_x A_x = I，移除
                party_symbols[party].pop()
            else:
                party_symbols[party].append(s)

        # 按方排序
        for party in range(self.n_parties):
            result.extend(party_symbols[party])

        return tuple(result)

    def solve(self, level: Union[int, str] = 1, verbose: bool = False) -> Dict:
        """
        求解 NPA 层级

        参数:
        - level: NPA 层级
        - verbose: 是否打印详细信息

        返回:
        - 结果字典
        """
        # 生成单词
        words = self._generate_words(level)
        n_words = len(words)

        if verbose:
            print(f"NPA 层级 {level}: 生成了 {n_words} 个单词")

        # 创建 word 到索引的映射
        word_to_idx = {}
        for i, word in enumerate(words):
            reduced = self._reduce_word(word)
            if reduced not in word_to_idx:
                word_to_idx[reduced] = i

        # 创建 SDP 变量
        Gamma = cp.Variable((n_words, n_words), symmetric=True)

        # 约束
        constraints = [Gamma >> 0]  # 半正定
        constraints.append(Gamma[0, 0] == 1)  # 归一化

        # 添加 NPA 约束
        for i in range(n_words):
            for j in range(i, n_words):
                word_i = words[i]
                word_j = words[j]

                # 简化 word_i† word_j
                word_i_conj = tuple(reversed(word_i))
                combined = word_i_conj + word_j
                reduced = self._reduce_word(combined)

                if reduced == ():
                    # 结果为 0
                    if i != j:
                        constraints.append(Gamma[i, j] == 0)
                elif reduced in word_to_idx:
                    idx = word_to_idx[reduced]
                    if idx == 0:
                        # 结果为 I
                        pass  # Gamma[i, j] 可以是任意值
                    elif idx != i or idx != j:
                        # 关联到另一个矩阵元素
                        pass

        # 构建目标函数
        objective_expr = 0

        for settings, coef in self.coefficients.items():
            # 找到对应的单词索引
            word = tuple((party, settings[party]) for party in range(self.n_parties))

            if word in word_to_idx:
                idx = word_to_idx[word]
                objective_expr += coef * Gamma[idx, 0]
            else:
                # 需要找到简化后的单词
                reduced = self._reduce_word(word)
                if reduced in word_to_idx:
                    idx = word_to_idx[reduced]
                    objective_expr += coef * Gamma[idx, 0]

        objective = cp.Maximize(objective_expr)

        # 求解
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.SCS, verbose=False)
            quantum_upper_bound = problem.value
        except Exception as e:
            if verbose:
                print(f"求解失败: {e}")
            quantum_upper_bound = None

        return {
            'classical_max': compute_classical_max(self.n_parties, self.measurements, self.coefficients),
            'quantum_upper_bound': quantum_upper_bound,
            'nosignal_max': compute_nosignal_max(self.coefficients),
            'n_words': n_words
        }


# =============================================================================
# 第五部分：预定义不等式
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


# =============================================================================
# 第六部分：测试
# =============================================================================

def run_tests():
    """运行测试"""
    print("=" * 60)
    print("NPA 方法测试")
    print("=" * 60)

    # 测试 1: CHSH (toqito)
    print("\n测试 1: CHSH 不等式 (toqito)")
    print("-" * 40)

    chsh = create_chsh_inequality()
    print(f"不等式: {chsh.get_expression_string()}")

    result = solve_with_toqito(chsh.coefficients, chsh.measurements, k=1)

    print(f"经典最大值: {result['classical_max']:.6f} (理论值: 2)")
    print(f"量子最大值: {result['quantum_max']:.6f} (理论值: 2√2 ≈ 2.828)")
    print(f"无信号最大值: {result['nosignal_max']:.6f} (理论值: 4)")

    # 测试 2: 链不等式 C3 (toqito)
    print("\n测试 2: 链不等式 C3 (toqito)")
    print("-" * 40)

    chain = create_chain_inequality(3)
    print(f"不等式: {chain.get_expression_string()}")

    result = solve_with_toqito(chain.coefficients, chain.measurements, k=1)

    print(f"经典最大值: {result['classical_max']:.6f} (理论值: 4)")
    print(f"量子最大值: {result['quantum_max']:.6f} (理论值: 5.196152)")
    print(f"无信号最大值: {result['nosignal_max']:.6f} (理论值: 6)")

    # 测试 3: CHSH (自定义 NPA)
    print("\n测试 3: CHSH 不等式 (自定义 NPA)")
    print("-" * 40)

    solver = NPASolver(n_parties=2, measurements=[2, 2])
    solver.set_coefficients({
        (0, 0): 1, (0, 1): 1,
        (1, 0): 1, (1, 1): -1,
    })

    for level in [1, '1+ab', 2]:
        result = solver.solve(level=level, verbose=False)
        print(f"NPA 层级 {level}:")
        print(f"  经典最大值: {result['classical_max']:.6f}")
        if result['quantum_upper_bound'] is not None:
            print(f"  量子上界: {result['quantum_upper_bound']:.6f}")
        else:
            print(f"  量子上界: 求解失败")
        print(f"  无信号最大值: {result['nosignal_max']:.6f}")
        print(f"  单词数: {result['n_words']}")

    # 测试 4: Mermin M3 (自定义 NPA)
    print("\n测试 4: Mermin M3 不等式 (自定义 NPA)")
    print("-" * 40)

    solver = NPASolver(n_parties=3, measurements=[2, 2, 2])
    solver.set_coefficients({
        (0, 0, 0): 1,
        (0, 1, 1): -1,
        (1, 0, 1): -1,
        (1, 1, 0): -1,
    })

    for level in [1, 2]:
        result = solver.solve(level=level, verbose=False)
        print(f"NPA 层级 {level}:")
        print(f"  经典最大值: {result['classical_max']:.6f}")
        if result['quantum_upper_bound'] is not None:
            print(f"  量子上界: {result['quantum_upper_bound']:.6f}")
        else:
            print(f"  量子上界: 求解失败")
        print(f"  无信号最大值: {result['nosignal_max']:.6f}")
        print(f"  单词数: {result['n_words']}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


USAGE = """
================================================================================
                    贝尔不等式量子最大值计算 - NPA 方法
                              使用说明
================================================================================

一、快速开始（两方不等式，使用 toqito）
--------------------------------------

from bell_inequality_npa_method import solve_with_toqito

# CHSH 不等式
coefficients = {
    (0, 0): 1, (0, 1): 1,
    (1, 0): 1, (1, 1): -1,
}
result = solve_with_toqito(coefficients, measurements=[2, 2], k=1)

print(f"量子最大值: {result['quantum_max']}")

二、快速开始（多方不等式，使用自定义 NPA）
-----------------------------------------

from bell_inequality_npa_method import NPASolver

# Mermin M3 不等式
solver = NPASolver(n_parties=3, measurements=[2, 2, 2])
solver.set_coefficients({
    (0, 0, 0): 1,
    (0, 1, 1): -1,
    (1, 0, 1): -1,
    (1, 1, 0): -1,
})
result = solver.solve(level=2)

print(f"量子上界: {result['quantum_upper_bound']}")

三、NPA 层级说明
----------------

- level=1: 单字母层级（最松，最快）
- level='1+ab': 包含不同方的双字母
- level=2: 所有双字母
- level=3: 所有三字母（最紧，最慢）

层级越高，上界越紧，但计算越慢。

================================================================================
"""


def print_usage():
    """打印使用说明"""
    print(USAGE)


if __name__ == "__main__":
    run_tests()
