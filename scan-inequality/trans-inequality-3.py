import itertools
import time
import os
from datetime import datetime
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# ========== 环境准备 ==========
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

def measurement(theta, phi):
    n = np.array([np.sin(theta) * np.cos(phi),
                  np.sin(theta) * np.sin(phi),
                  np.cos(theta)])
    return n[0] * sigma_x + n[1] * sigma_y + n[2] * sigma_z

def correlation(psi, A, B, C):
    op = np.kron(A, np.kron(B, C))
    return np.conj(psi).T @ (op @ psi)

# ========== 1. 二体 CHSH 等价类生成 (保持不变) ==========
chsh_standard = [(1, (1, 1)), (1, (1, 2)), (1, (2, 1)), (-1, (2, 2))]

def permute_term(term, permA, permB):
    coeff, (i, j) = term
    return (coeff, (permA[i - 1], permB[j - 1]))

def combine_terms(terms):
    d = defaultdict(float)
    for coeff, idx in terms:
        d[idx] += coeff
    return [(coeff, idx) for idx, coeff in d.items() if not np.isclose(coeff, 0)]

def permute_inequality(terms, permA, permB):
    new_terms = [permute_term(t, permA, permB) for t in terms]
    return combine_terms(new_terms)

def swap_parties(terms):
    new_terms = [(coeff, (j, i)) for coeff, (i, j) in terms]
    return combine_terms(new_terms)

def canonical_inequality(terms):
    sorted_terms = sorted(terms, key=lambda x: (x[1], x[0]))
    if not sorted_terms:
        return []
    first_coeff = sorted_terms[0][0]
    if first_coeff < 0:
        return [(-coeff, idx) for coeff, idx in sorted_terms]
    return sorted_terms

def generate_all_chsh_equivalents(M):
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
    parts = []
    for coeff, (i, j) in terms:
        sign = '+' if coeff >= 0 else '-'
        parts.append(f"{sign}A{i}B{j}")
    return ''.join(parts).lstrip('+')

# ========== 2. 新的三体构造：利用三个 CHSH 等价类 ==========
def construct_3body_from_three(termsA, termsB, termsC):
    """
    根据公式 B_3 = 1/2[ B_A (C_2+C_3) + B_B (C_1-C_3) + B_C (C_1-C_2) ]
    生成三体不等式项列表。 termsA, termsB, termsC 都是二体不等式
    """
    combined = defaultdict(float)
    # B_A 乘 (C_2 + C_3)
    for coeff, (i, j) in termsA:
        combined[(i, j, 2)] += coeff
        combined[(i, j, 3)] += coeff
    # B_B 乘 (C_1 - C_3)
    for coeff, (i, j) in termsB:
        combined[(i, j, 1)] += coeff
        combined[(i, j, 3)] -= coeff
    # B_C 乘 (C_1 - C_2)
    for coeff, (i, j) in termsC:
        combined[(i, j, 1)] += coeff
        combined[(i, j, 2)] -= coeff
    # 合并后整体除以 2
    result = []
    for idx, coeff in combined.items():
        new_coeff = coeff / 2.0
        if not np.isclose(new_coeff, 0):
            result.append((new_coeff, idx))
    return result

def canonical_inequality_3(terms):
    """三体不等式的规范形式：排序，首项系数为正"""
    sorted_terms = sorted(terms, key=lambda x: (x[1], x[0]))
    if not sorted_terms:
        return []
    first_coeff = sorted_terms[0][0]
    if first_coeff < 0:
        return [(-coeff, idx) for coeff, idx in sorted_terms]
    return sorted_terms

def inequality_string_3(terms):
    parts = []
    for coeff, (i, j, k) in terms:
        sign = '+' if coeff >= 0 else '-'
        parts.append(f"{sign}A{i}B{j}C{k}")
    return ''.join(parts).lstrip('+')

def inequality_tex(terms):
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

# ========== 3. 违背值计算函数 (不变) ==========
def bell_expression(terms, opsA, opsB, opsC, psi):
    val = 0.0 + 0.0j
    for coeff, (i, j, k) in terms:
        val += coeff * correlation(psi, opsA[i - 1], opsB[j - 1], opsC[k - 1])
    return val

def make_measurements(angles):
    return [measurement(theta, phi) for theta, phi in angles]

def ops_from_vars(var_list, n_meas):
    a_angles = var_list[:2 * n_meas].reshape(n_meas, 2)
    b_angles = var_list[2 * n_meas:4 * n_meas].reshape(n_meas, 2)
    c_angles = var_list[4 * n_meas:].reshape(n_meas, 2)
    return (make_measurements(a_angles),
            make_measurements(b_angles),
            make_measurements(c_angles))

def target_function(var_list, psi, terms, n_meas):
    opsA, opsB, opsC = ops_from_vars(var_list, n_meas)
    be = bell_expression(terms, opsA, opsB, opsC, psi)
    return np.abs(be) / 2.0

def max_violation_for_state(psi, terms, n_meas, n_starts=20):
    dim = 6 * n_meas
    bounds = []
    for _ in range(3 * n_meas):
        bounds.append((0.0, np.pi))
        bounds.append((0.0, 2.0 * np.pi))
    best_val = -np.inf
    def neg_target(vars):
        return -target_function(vars, psi, terms, n_meas)
    for _ in range(n_starts):
        x0 = np.array([np.random.uniform(low, high) for low, high in bounds])
        res = minimize(neg_target, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-12})
        val = -res.fun
        if val > best_val:
            best_val = val
    return best_val

def ghz_state(alpha):
    vec = np.array([np.cos(alpha), 0, 0, 0, 0, 0, 0, np.sin(alpha)])
    return vec / np.linalg.norm(vec)

# ========== 4. 主流程 ==========
if __name__ == "__main__":
    # 参数设定
    M = 3                      # 测量设置数，须 >=3 以支持 C1,C2,C3
    N_STARTS = 20              # 优化起点数
    ALPHA_POINTS = 11          # α 扫描点数
    MAX_COMBINATIONS = 3000    # 三体组合数上限，避免组合爆炸

    # 日志文件夹
    base_dir = "output_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(base_dir, exist_ok=True)
    log_path = os.path.join(base_dir, "log.txt")
    log_file = open(log_path, 'w', encoding='utf-8')

    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=log_file)
        log_file.flush()

    # ---------- 生成二体 CHSH 等价类 ----------
    log_print(f"M = {M}，正在生成二体 CHSH 等价类...")
    equiv_ineqs_2 = generate_all_chsh_equivalents(M)
    log_print(f"共得到 {len(equiv_ineqs_2)} 个二体 CHSH 等价类。")
    for idx, ineq in enumerate(equiv_ineqs_2, 1):
        log_print(f"  {idx}: {inequality_string_2(ineq)}")

    # ---------- 用三个等价类构造三体不等式 ----------
    log_print("\n利用三个 CHSH 等价类构造三体不等式...")
    # 生成所有有序三元组 (B_A, B_B, B_C)
    total_comb = len(equiv_ineqs_2)**3
    if total_comb > MAX_COMBINATIONS:
        log_print(f"三元组总数 {total_comb} 超过上限 {MAX_COMBINATIONS}，随机抽取组合。")
        rng = np.random.default_rng(42)
        indices = rng.choice(total_comb, size=MAX_COMBINATIONS, replace=False)
        triples = []
        n_eq = len(equiv_ineqs_2)
        for idx_flat in indices:
            iC = idx_flat % n_eq
            iB = (idx_flat // n_eq) % n_eq
            iA = idx_flat // (n_eq * n_eq)
            triples.append((equiv_ineqs_2[iA], equiv_ineqs_2[iB], equiv_ineqs_2[iC]))
    else:
        triples = list(itertools.product(equiv_ineqs_2, repeat=3))
    log_print(f"实际使用 {len(triples)} 个有序三元组。")

    # 生成三体不等式并去重
    seen_3 = set()
    all3_ineqs_with_sources = []
    for bA, bB, bC in triples:
        t3 = construct_3body_from_three(bA, bB, bC)
        if not t3:
            continue
        t3_canon = canonical_inequality_3(t3)
        key = tuple((coeff, idx) for coeff, idx in t3_canon)
        if key not in seen_3:
            seen_3.add(key)
            all3_ineqs_with_sources.append((t3_canon, (bA, bB, bC)))
    all3_ineqs = [item[0] for item in all3_ineqs_with_sources]
    log_print(f"去重后得到 {len(all3_ineqs)} 个三体不等式。")

    # 打印前几个示例
    show_num = min(5, len(all3_ineqs))
    for i in range(show_num):
        ineq, (bA, bB, bC) = all3_ineqs_with_sources[i]
        log_print(f"\n不等式 {i+1}: {inequality_tex(ineq)}")
        log_print(f"  B_A = {inequality_string_2(bA)}")
        log_print(f"  B_B = {inequality_string_2(bB)}")
        log_print(f"  B_C = {inequality_string_2(bC)}")

    # ---------- 扫描 GHZ 态违背值 ----------
    alpha_range = np.linspace(0, np.pi / 2, ALPHA_POINTS)
    all_scan_results = []
    timing_list = []

    log_print("\n开始扫描每个不等式的违背值...")
    for i, terms in enumerate(all3_ineqs, start=1):
        ineq_str = inequality_string_3(terms)
        log_print(f"\n====== 不等式 {i}/{len(all3_ineqs)} ======")
        log_print(inequality_tex(terms))
        start_time = time.time()

        def compute_for_alpha(alpha):
            psi = ghz_state(alpha)
            return alpha, max_violation_for_state(psi, terms, M, n_starts=N_STARTS)

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

        # 保存单张图片
        alphas = [r[0] for r in results]
        max_Ls = [r[1] for r in results]
        fig, ax = plt.subplots()
        ax.plot(alphas, max_Ls, '-o')
        ax.set_xlabel('α')
        ax.set_ylabel('Max L')
        ax.set_title(f'Ineq {i}')
        ax.grid(True)
        fig_path = os.path.join(base_dir, f"inequality_{i}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        all_scan_results.append((ineq_str, results, i))

    # 用时汇总
    log_print("\n====== 各不等式扫描用时汇总 ======")
    log_print(f"{'不等式':<40} {'用时 (秒)':>10}")
    for name, t in timing_list:
        log_print(f"{name:<40} {t:>10.2f}")

    # 汇总图（仅当不等式数量不过多时绘制）
    if len(all_scan_results) <= 50:
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
            ax.set_title(f'Ineq {i}')
            ax.grid(True)
        for idx in range(n_eqs, len(axes)):
            axes[idx].axis('off')
        plt.tight_layout()
        summary_path = os.path.join(base_dir, "summary.png")
        fig.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        log_print(f"汇总图片已保存：{summary_path}")
    else:
        log_print(f"\n不等式数量过多（{len(all_scan_results)}），跳过汇总图。")

    log_file.close()
    print(f"所有结果已保存至文件夹：{base_dir}")