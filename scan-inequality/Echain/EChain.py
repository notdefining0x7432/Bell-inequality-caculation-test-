import numpy as np
from scipy.optimize import differential_evolution
from functools import reduce
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time

# ========== 1. 基本定义 ==========
# Pauli 矩阵
sigma = np.array([
    [[0, 1], [1, 0]],  # sigma_x
    [[0, -1j], [1j, 0]],  # sigma_y
    [[1, 0], [0, -1]]  # sigma_z
])


def measurement(theta, phi):
    """单量子比特测量算符 n·σ"""
    n = np.array([np.sin(theta) * np.cos(phi),
                  np.sin(theta) * np.sin(phi),
                  np.cos(theta)])
    return n[0] * sigma[0] + n[1] * sigma[1] + n[2] * sigma[2]


def multi_correlation(psi, ops):
    """多体关联函数：<psi| A⊗B⊗C... |psi>"""
    op_total = reduce(np.kron, ops)
    return np.conj(psi).T @ (op_total @ psi)


def bell_expectation(psi, opsA, opsB, opsC, m):
    """计算 Bell 期望值"""
    # part1: 与 (C1+C2) 相乘的部分
    part1 = 0.0
    for i in range(m - 1):  # i = 0...m-2 (对应原式 i=1...m-1)
        part1 += multi_correlation(psi, [opsA[i], opsB[i], opsC[0]])
        part1 += multi_correlation(psi, [opsA[i], opsB[i], opsC[1]])
        part1 += multi_correlation(psi, [opsA[i + 1], opsB[i], opsC[0]])
        part1 += multi_correlation(psi, [opsA[i + 1], opsB[i], opsC[1]])
    # i = m-1 (原式 i=m)
    part1 += multi_correlation(psi, [opsA[m - 1], opsB[m - 1], opsC[0]])
    part1 += multi_correlation(psi, [opsA[m - 1], opsB[m - 1], opsC[1]])
    part1 -= multi_correlation(psi, [opsA[0], opsB[m - 1], opsC[0]])
    part1 -= multi_correlation(psi, [opsA[0], opsB[m - 1], opsC[1]])

    # part2: 与 (C1-C2) 相乘的部分
    part2 = 0.0
    for i in range(m, 2 * m - 1):  # i = m...2m-2 (原式 i=m+1...2m-1)
        part2 += multi_correlation(psi, [opsA[i], opsB[i], opsC[0]])
        part2 -= multi_correlation(psi, [opsA[i], opsB[i], opsC[1]])
        part2 += multi_correlation(psi, [opsA[i + 1], opsB[i], opsC[0]])
        part2 -= multi_correlation(psi, [opsA[i + 1], opsB[i], opsC[1]])
    # i = 2m-1 (原式 i=2m)
    part2 += multi_correlation(psi, [opsA[2 * m - 1], opsB[2 * m - 1], opsC[0]])
    part2 -= multi_correlation(psi, [opsA[2 * m - 1], opsB[2 * m - 1], opsC[1]])
    part2 -= multi_correlation(psi, [opsA[0], opsB[2 * m - 1], opsC[0]])
    part2 += multi_correlation(psi, [opsA[0], opsB[2 * m - 1], opsC[1]])

    return part1 + part2


def make_ops(vars_, m):
    """从变量构建测量算符列表"""
    numA = 2 * m
    thetaA = vars_[0:numA]
    phiA = vars_[numA:2 * numA]
    thetaB = vars_[2 * numA:3 * numA]
    phiB = vars_[3 * numA:4 * numA]
    thetaC = vars_[4 * numA:4 * numA + 2]
    phiC = vars_[4 * numA + 2:4 * numA + 4]

    opsA = [measurement(thetaA[i], phiA[i]) for i in range(numA)]
    opsB = [measurement(thetaB[i], phiB[i]) for i in range(numA)]
    opsC = [measurement(thetaC[k], phiC[k]) for k in range(2)]
    return opsA, opsB, opsC


# ========== 2. 经典界限 ==========
def classical_bound(m):
    return 4 * m - 4


# ========== 3. 目标函数（给优化器） ==========
def make_target(psi, m):
    """返回一个可供 scipy 最小化的目标函数（负值）"""
    bound = classical_bound(m)

    def target(vars_):
        opsA, opsB, opsC = make_ops(vars_, m)
        val = bell_expectation(psi, opsA, opsB, opsC, m)
        return -np.abs(val) / bound  # 最小化 -L 等价于最大化 L

    return target


def max_L_for_state(psi, m, seed=0):
    """对给定量子态，找到最大 L"""
    numA = 2 * m
    # 变量顺序：thetaA(2m), phiA(2m), thetaB(2m), phiB(2m), thetaC(2), phiC(2)
    bounds = ([[0, np.pi]] * numA + [[0, 2 * np.pi]] * numA +
              [[0, np.pi]] * numA + [[0, 2 * np.pi]] * numA +
              [[0, np.pi]] * 2 + [[0, 2 * np.pi]] * 2)

    target_func = make_target(psi, m)

    # 使用差分进化进行全局优化
    result = differential_evolution(
        target_func, bounds,
        maxiter=500,  # 最大迭代代数
        popsize=15,  # 种群规模，可根据变量数调整
        tol=1e-5,  # 收敛容忍度
        atol=1e-5,
        seed=seed,
        polish=False  # 最后是否用局部优化器精修（可设为 True 但耗时）
    )
    # 返回最大 L（注意取负号）
    return -result.fun


# ========== 4. 量子态定义 ==========
def three_qubit_ghz_state(alpha):
    """|ψ(α)⟩ = cosα|000⟩ + sinα|111⟩"""
    psi = np.zeros(8, dtype=complex)
    psi[0] = np.cos(alpha)  # |000>
    psi[7] = np.sin(alpha)  # |111>
    return psi  # 已经归一化


# ========== 5. 参数设置与扫描 ==========
if __name__ == "__main__":
    m = 4
    n_points = 31
    d = 10.0  # 中心密集参数

    # 非线性变换：使 alpha 在中心区域更密集
    t_range = np.linspace(0, 1, n_points)
    A = np.pi / 2 - d / 12
    B = d
    alpha_range = A * t_range + (B / 3) * (t_range - 0.5) ** 3 + B / 24

    print(f"开始扫描三体纠缠态参数 (m = {m})...")
    start_time = time.time()

    # 串行执行（若想并行，可将 Parallel 替换为 joblib.Parallel）
    # 这里提供并行版本（需安装 joblib）
    try:
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(max_L_for_state)(three_qubit_ghz_state(a), m) for a in alpha_range
        )
    except:
        # 若 joblib 不可用，退回串行
        print("joblib 未安装，使用串行计算。")
        results = []
        for i, a in enumerate(alpha_range):
            L = max_L_for_state(three_qubit_ghz_state(a), m)
            results.append(L)
            print(f"进度: {i + 1}/{n_points}, alpha={a:.4f}, L={L:.6f}, 用时 {time.time() - start_time:.0f}s")

    elapsed = time.time() - start_time
    print(f"扫描完成，总耗时：{elapsed:.0f} 秒")

    # 整理结果
    results = np.array(results)
    data = np.column_stack((alpha_range, results))

    # 打印表格
    print("扫描结果（alpha 与 最大 L）：")
    print("alpha\t\tMax L")
    for a, L in data:
        print(f"{a:.6f}\t{L:.6f}")

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(alpha_range, results, 'o-', markersize=5)
    # plt.axhline(y=2, color='r', linestyle='--', label='Classical bound')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Maximum L')
    plt.title(f'Bell violation for B3 inequality (m = {m})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()