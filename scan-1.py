from bell_inequality_unified import BellInequality, solve
from itertools import product

def generate_all_inequalities():
    """生成所有可能的三体2测量不等式（系数为±1）"""
    terms = [
        (0, 0, 0), (0, 0, 1),
        (0, 1, 0), (0, 1, 1),
        (1, 0, 0), (1, 0, 1),
        (1, 1, 0), (1, 1, 1),
    ]

    for coeffs in product([-1, 1], repeat=8):
        coefficients = {term: coeff for term, coeff in zip(terms, coeffs)}
        ineq = BellInequality(                     # ← 先创建对象
            n_parties=3, measurements=[2, 2, 2],
            name=f"三体2测量不等式_{coeffs}"
        )
        ineq.set_coefficients(coefficients)        # ← 再设置系数
        yield ineq                                 # ← yield 的是 BellInequality 对象


def find_violating_inequalities():
    """筛选有量子违背的不等式"""
    violating_ineqs = []
    for ineq in generate_all_inequalities():
        result = solve(ineq, method='qutip', optimize=True, n_random=2000)
        if result['quantum_max'] > result['classical_max']:
            violating_ineqs.append({
                'name': ineq.name,
                'classical_max': result['classical_max'],
                'quantum_max': result['quantum_max'],
                'violation_ratio': result['quantum_max'] / result['classical_max']
            })
    return violating_ineqs


if __name__ == "__main__":
    violating_ineqs = find_violating_inequalities()
    print(f"找到 {len(violating_ineqs)} 个有量子违背的不等式：")
    for ineq in violating_ineqs:
        print(
            f"- {ineq['name']}: 经典最大值={ineq['classical_max']:.4f}, "
            f"量子最大值={ineq['quantum_max']:.4f}, "
            f"违背比={ineq['violation_ratio']:.2f}x"
        )
