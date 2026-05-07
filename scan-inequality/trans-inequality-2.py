import itertools
import time
import os
from datetime import datetime
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib
matplotlib.use('Agg')                     # 2. 强制使用非 GUI 的 Agg 后端
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# ========== 环境准备 ==========
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

def measurement(theta, phi):
    """返回 n·σ 测量算符，其中 n = (sinθ cosφ, sinθ sinφ, cosθ)"""
    n = np.array([np.sin(theta) * np.cos(phi),
                  np.sin(theta) * np.sin(phi),
                  np.cos(theta)])
    return n[0] * sigma_x + n[1] * sigma_y + n[2] * sigma_z


def correlation(psi, A, B, C):
    """三体关联函数 ⟨ψ| A ⊗ B ⊗ C |ψ⟩"""
    op = np.kron(A, np.kron(B, C))
    return np.conj(psi).T @ (op @ psi)


# ========== 1. 二体 CHSH 等价类生成 ==========
chsh_standard = [(1, (1, 1)), (1, (1, 2)), (1, (2, 1)), (-1, (2, 2))]


def permute_term(term, permA, permB):
    coeff, (i, j) = term
    return (coeff, (permA[i - 1], permB[j - 1]))


def combine_terms(terms):
    """合并相同 indices 的项，去除系数为 0 的项"""
    d = defaultdict(float)
    for coeff, idx in terms:
        d[idx] += coeff
    return [(coeff, idx) for idx, coeff in d.items() if not np.isclose(coeff, 0)]


def permute_inequality(terms, permA, permB):
    """对不等式每一项应用排列，然后合并"""
    new_terms = [permute_term(t, permA, permB) for t in terms]
    return combine_terms(new_terms)


def swap_parties(terms):
    """交换 A 和 B 的索引"""
    new_terms = [(coeff, (j, i)) for coeff, (i, j) in terms]
    return combine_terms(new_terms)


def canonical_inequality(terms):
    """排序并使首项系数为正"""
    sorted_terms = sorted(terms, key=lambda x: (x[1], x[0]))
    if not sorted_terms:
        return []
    first_coeff = sorted_terms[0][0]
    if first_coeff < 0:
        return [(-coeff, idx) for coeff, idx in sorted_terms]
    return sorted_terms


def generate_all_chsh_equivalents(M):
    """生成 M 测量设置下所有二体 CHSH 等价类"""
    if M < 2:
        raise ValueError(f"M 至少为 2，当前 M = {M}")
    perms = list(itertools.permutations(range(1, M + 1)))
    raw = []
    for pA in perms:
        for pB in perms:
            raw.append(permute_inequality(chsh_standard, pA, pB))
    swapped_raw = [swap_parties(ineq) for ineq in raw]
    all_ineqs = raw + swapped_raw
    can = []
    seen = set()
    for ineq in all_ineqs:
        c = canonical_inequality(ineq)
        key = tuple((coeff, idx) for coeff, idx in c)
        if key not in seen:
            seen.add(key)
            can.append(c)
    return can


def inequality_string_2(terms):
    """输出如 '+A1B1+A1B2+A2B1-A2B2'"""
    parts = []
    for coeff, (i, j) in terms:
        sign = '+' if coeff >= 0 else '-'
        parts.append(f"{sign}A{i}B{j}")
    s = ''.join(parts)
    return s.lstrip('+')


# ========== 2. 将二体不等式提升为三体 ==========
def swap_terms_1_2(terms):
    """交换 indices 中的 1 和 2（仅用于二体提升）"""

    def swap_idx(idx):
        i, j = idx
        new_i = 2 if i == 1 else (1 if i == 2 else i)
        new_j = 2 if j == 1 else (1 if j == 2 else j)
        return (new_i, new_j)

    return [(coeff, swap_idx(idx)) for coeff, idx in terms]


def recursive_mabk3(terms_b2):
    """由二体不等式 terms_b2 生成三体不等式"""
    terms_b2_prime = swap_terms_1_2(terms_b2)
    terms3 = []
    for coeff, (i, j) in terms_b2:
        terms3.append((coeff, (i, j, 1)))
        terms3.append((coeff, (i, j, 2)))
    for coeff, (i, j) in terms_b2_prime:
        terms3.append((coeff, (i, j, 1)))
        terms3.append((-coeff, (i, j, 2)))
    combined = defaultdict(float)
    for coeff, idx in terms3:
        combined[idx] += coeff
    result = [(coeff / 2.0, idx) for idx, coeff in combined.items() if not np.isclose(coeff, 0)]
    return result


def inequality_string_3(terms):
    """紧凑格式，如 '+A1B1C1+A1B1C2+A1B2C1...'"""
    parts = []
    for coeff, (i, j, k) in terms:
        sign = '+' if coeff >= 0 else '-'
        parts.append(f"{sign}A{i}B{j}C{k}")
    s = ''.join(parts)
    return s.lstrip('+')


def inequality_tex(terms):
    """LaTeX 格式输出"""
    n_indices = len(terms[0][1]) if terms else 0
    parts = []
    for coeff, idx in terms:
        i, j = idx[0], idx[1]
        k = idx[2] if n_indices == 3 else None
        if np.isclose(coeff, 1.0):
            pre = ""
        elif np.isclose(coeff, -1.0):
            pre = "-"
        else:
            pre = f"{coeff} "
        item = f"A_{{{i}}} B_{{{j}}}"
        if k is not None:
            item += f" C_{{{k}}}"
        parts.append(f"{pre}{item}")
    s = " + ".join(parts)
    s = s.replace("+ -", "- ")
    return s


# ========== 3. 违背值计算函数 ==========
def bell_expression(terms, opsA, opsB, opsC, psi):
    """计算贝尔表达式在给定算符和态下的值"""
    val = 0.0 + 0.0j
    for coeff, (i, j, k) in terms:
        val += coeff * correlation(psi, opsA[i - 1], opsB[j - 1], opsC[k - 1])
    return val


def make_measurements(angles):
    """angles: list of (theta, phi) -> list of 测量算符"""
    return [measurement(theta, phi) for theta, phi in angles]


def ops_from_vars(var_list, n_meas):
    """将变量展平数组转换为三个 party 的测量算符列表"""
    a_angles = var_list[:2 * n_meas].reshape(n_meas, 2)
    b_angles = var_list[2 * n_meas:4 * n_meas].reshape(n_meas, 2)
    c_angles = var_list[4 * n_meas:].reshape(n_meas, 2)
    return (make_measurements(a_angles),
            make_measurements(b_angles),
            make_measurements(c_angles))


def target_function(var_list, psi, terms, n_meas):
    """返回违背值 = |贝尔表达式|/2 （用于最大化）"""
    opsA, opsB, opsC = ops_from_vars(var_list, n_meas)
    be = bell_expression(terms, opsA, opsB, opsC, psi)
    return np.abs(be) / 2.0


def max_violation_for_state(psi, terms, n_meas, n_starts=50):
    """
    多起点局部优化寻找最大违背值。
    变量数 = 3 * n_meas * 2 (θ,φ)
    """
    dim = 6 * n_meas
    bounds = []
    for _ in range(3 * n_meas):
        bounds.append((0.0, np.pi))
        bounds.append((0.0, 2.0 * np.pi))

    best_val = -np.inf
    best_x = None

    def neg_target(vars):
        return -target_function(vars, psi, terms, n_meas)

    for _ in range(n_starts):
        x0 = np.array([np.random.uniform(low, high) for low, high in bounds])
        res = minimize(neg_target, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-12})
        val = -res.fun
        if val > best_val:
            best_val = val
            best_x = res.x
    return best_val


def ghz_state(alpha):
    """生成 |ψ⟩ = cosα|000⟩ + sinα|111⟩"""
    vec = np.array([np.cos(alpha), 0, 0, 0, 0, 0, 0, np.sin(alpha)])
    return vec / np.linalg.norm(vec)


# ========== 4. 主流程（含文件输出） ==========
if __name__ == "__main__":
    # ---------- 创建输出文件夹 ----------
    base_dir = "output_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(base_dir, exist_ok=True)
    log_path = os.path.join(base_dir, "log.txt")

    # 打开日志文件
    log_file = open(log_path, 'w', encoding='utf-8')


    def log_print(*args, **kwargs):
        """同时打印到屏幕和日志文件"""
        print(*args, **kwargs)
        print(*args, **kwargs, file=log_file)
        log_file.flush()  # 确保实时写入


    # ---------- 原有流程 ----------
    M = 4
    log_print(f"M = {M} 时，正在生成二体 CHSH 等价类...")
    equiv_ineqs_2 = generate_all_chsh_equivalents(M)
    log_print(f"二体 CHSH 等价类共 {len(equiv_ineqs_2)} 个：")
    for ineq in equiv_ineqs_2:
        log_print(inequality_string_2(ineq))

    # 将每个二体等价类提升为三体，并记录来源
    raw_data = []
    for ineq in equiv_ineqs_2:
        t3 = recursive_mabk3(ineq)
        swapped = swap_terms_1_2(ineq)
        raw_data.append((t3, ineq, swapped))

    # 按照排序后的三体不等式分组
    grouped = defaultdict(list)
    for item in raw_data:
        t3, ineq, swapped = item
        sorted_t3 = tuple(sorted(t3, key=lambda x: (x[1], x[0])))
        grouped[sorted_t3].append(item)

    all3_ineqs_with_sources = []
    for key, items in grouped.items():
        t3_first = items[0][0]
        sources = [(ineq, swapped) for (_, ineq, swapped) in items]
        all3_ineqs_with_sources.append((t3_first, sources))

    all3_ineqs = [item[0] for item in all3_ineqs_with_sources]
    log_print(f"\n由这些等价类生成的三体不等式（去重后）共 {len(all3_ineqs)} 个：")

    # 打印每个不等式及其来源
    for i, (ineq, sources) in enumerate(all3_ineqs_with_sources, start=1):
        log_print(f"\n====== 不等式 {i} ======")
        log_print(inequality_tex(ineq))
        log_print("由以下二体不等式对提升得到：")
        for j, (b2, b2p) in enumerate(sources, start=1):
            log_print(f"  对 {j}: B2  = {inequality_string_2(b2)},   B2' = {inequality_string_2(b2p)}")

    # ========== 5. 参数扫描：GHZ 态 ==========
    alpha_range = np.linspace(0, np.pi / 2, 21)
    all_scan_results = []
    timing_list = []

    log_print("\n开始扫描每个不等式的违背值...")
    for i, terms in enumerate(all3_ineqs, start=1):
        ineq_str = inequality_string_3(terms)
        log_print(f"\n====== 不等式 {i} ======")
        log_print(inequality_tex(terms))
        log_print("正在扫描违背值...")
        start_time = time.time()


        def compute_for_alpha(alpha):
            psi = ghz_state(alpha)
            return alpha, max_violation_for_state(psi, terms, M, n_starts=50)


        # 使用 joblib 并行计算所有 α
        results = Parallel(n_jobs=-1)(delayed(compute_for_alpha)(a) for a in alpha_range)
        results.sort(key=lambda x: x[0])

        elapsed = time.time() - start_time
        timing_list.append((ineq_str, elapsed))
        log_print(f"扫描用时：{elapsed:.2f} 秒")

        # 输出表格
        header = ["α", "Max L"]
        table = [header] + [[f"{a:.4f}", f"{v:.6f}"] for a, v in results]
        col_widths = [max(len(row[j]) for row in table) for j in range(2)]
        for row in table:
            log_print("  ".join(row[j].ljust(col_widths[j]) for j in range(2)))

        # 保存单张图片（使用编号）
        alphas = [r[0] for r in results]
        max_Ls = [r[1] for r in results]
        fig, ax = plt.subplots()
        ax.plot(alphas, max_Ls, '-o')
        ax.set_xlabel('α')
        ax.set_ylabel('Max L')
        ax.set_title(f'Inequality {i}: {ineq_str}')
        ax.grid(True)
        # 保存图片到输出文件夹
        fig_path = os.path.join(base_dir, f"inequality_{i}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # 不弹出窗口
        log_print(f"图片已保存：{fig_path}")

        all_scan_results.append((ineq_str, results, i))  # 保存编号以便汇总

    # 打印用时汇总
    log_print("\n====== 各不等式扫描用时汇总 ======")
    log_print(f"{'不等式':<30} {'用时 (秒)':>10}")
    for name, t in timing_list:
        log_print(f"{name:<30} {t:>10.2f}")

    # ========== 6. 汇总图片 ==========
    log_print("\n正在生成汇总图片...")
    n_eqs = len(all_scan_results)
    cols = 3
    rows = (n_eqs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if n_eqs > 1 else [axes]
    for idx, (name, results, i) in enumerate(all_scan_results):
        alphas = [r[0] for r in results]
        max_Ls = [r[1] for r in results]
        ax = axes[idx]
        ax.plot(alphas, max_Ls, '-o')
        ax.set_xlabel('α')
        ax.set_ylabel('Max L')
        ax.set_title(f'Ineq {i}: {name}')
        ax.grid(True)
    for idx in range(n_eqs, len(axes)):
        axes[idx].axis('off')
    plt.tight_layout()
    summary_path = os.path.join(base_dir, "summary.png")
    fig.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log_print(f"汇总图片已保存：{summary_path}")

    # 关闭日志文件
    log_file.close()
    log_print(f"所有结果已保存至文件夹：{base_dir}")  # 此时文件中已关闭，但屏幕仍能看到