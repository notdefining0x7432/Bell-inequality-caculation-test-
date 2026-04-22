from bell_inequality_unified import BellInequality, solve
from itertools import product
import sys
import time
import datetime
import os


def generate_all_inequalities():
    """生成所有可能的二体四测量不等式（系数为±1）"""
    terms = [
        (a, b)
        for a in range(4) for b in range(4)
    ]

    for coeffs in product([-1, 1], repeat=16):
        coefficients = {term: coeff for term, coeff in zip(terms, coeffs)}
        ineq = BellInequality(
            n_parties=2, measurements=[4, 4],
            name=f"二体4测量_系数组合"
        )
        ineq.set_coefficients(coefficients)
        yield ineq


def find_violating_inequalities():
    """筛选有量子违背的不等式"""
    violating_ineqs = []
    actual_total = 2 ** 16  # 65536

    print(f"【任务信息】二体四测量总共有 {actual_total:,} 种可能组合")
    print(f"【智能加速】检测到两方不等式，将自动使用 toqito 计算精确量子最大值")
    print(f"【耗时预估】预计需要 10~20 小时 (取决于机器性能)\n")

    start_time = time.time()

    for idx, ineq in enumerate(generate_all_inequalities()):
        # -------- 进度输出 --------
        elapsed = time.time() - start_time
        progress = (idx + 1) / actual_total
        bar_len = 40
        filled = int(bar_len * progress)
        bar = '█' * filled + '░' * (bar_len - filled)

        if progress > 0.01:
            eta = elapsed / progress - elapsed
            if eta > 3600:
                eta_str = f"预计剩余 {eta / 3600:.1f}h"
            else:
                eta_str = f"预计剩余 {eta / 60:.0f}min"
        else:
            eta_str = "预估中..."

        sys.stdout.write(
            f'\r[{bar}] {progress:6.2%}  '
            f'({idx + 1}/{actual_total:,})  '
            f'已找到 {len(violating_ineqs)} 个违背  '
            f'已用时 {elapsed / 3600:.1f}h  {eta_str}'
        )
        sys.stdout.flush()
        # -------- 进度输出结束 --------

        result = solve(ineq, method='auto')

        if result['quantum_max'] > result['classical_max']:
            violating_ineqs.append({
                'name': ineq.name,
                'coefficients': ineq.coefficients,
                'classical_max': result['classical_max'],
                'quantum_max': result['quantum_max'],
                'violation_ratio': result['quantum_max'] / result['classical_max']
            })

    total_time = time.time() - start_time
    print(f"\n\n扫描完成，总用时: {total_time / 3600:.2f} 小时")
    return violating_ineqs


def save_results_to_file(violating_ineqs, total_time_seconds):
    """将违背的不等式完整保存到 txt 文件中"""
    # 按违背比从大到小排序
    violating_ineqs.sort(key=lambda x: x['violation_ratio'], reverse=True)

    # 生成带有时间戳的文件名，防止重复运行时覆盖
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"二体四测量违背结果_{timestamp}.txt"
    filepath = os.path.abspath(filename)  # 获取绝对路径，方便用户找文件

    print(f"\n正在将 {len(violating_ineqs)} 条结果写入文件...")

    with open(filepath, 'w', encoding='utf-8') as f:
        # 写入文件头
        f.write("=" * 70 + "\n")
        f.write("         二体四测量 (2-Party 4-Measurement) 贝尔不等式扫描报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"扫描总组合数: {2 ** 16:,} \n")
        f.write(f"发现违背数量: {len(violating_ineqs)} \n")
        f.write(f"总计算用时:   {total_time_seconds / 3600:.2f} 小时\n")
        f.write(f"生成时间:     {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"计算方法:     toqito (NPA层级, 精确量子最大值)\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("详细违背列表 (按违背强度降序排列)\n")
        f.write("=" * 70 + "\n\n")

        # 逐个写入违背的不等式
        for i, ineq in enumerate(violating_ineqs):
            f.write(f"【第 {i + 1} 个】 违背比 (量子/经典): {ineq['violation_ratio']:.6f}x\n")
            f.write(f"  经典最大值: {ineq['classical_max']:.6f}\n")
            f.write(f"  量子最大值: {ineq['quantum_max']:.6f}\n")
            f.write(f"  系数分布 (测量设置: 系数):\n")

            # 将系数排版成 4x4 的矩阵形式，极其直观
            # (0,0) (0,1) (0,2) (0,3)  对应 A1
            # (1,0) (1,1) (1,2) (1,3)  对应 A2
            # ...
            for a in range(4):
                row_str = "    "
                for b in range(4):
                    coef = ineq['coefficients'][(a, b)]
                    # 格式化为 +1 或 -1，占 4 个字符宽度右对齐
                    row_str += f"{coef:+d}  "
                f.write(f"  A{a + 1}方: {row_str}\n")

            f.write(f"  (注: 行代表A方测量A1~A4，列代表B方测量B1~B4)\n")
            f.write("\n" + "-" * 70 + "\n\n")

    print(f"✅ 文件已成功保存至:\n   {filepath}\n")


if __name__ == "__main__":
    start_time = time.time()
    violating_ineqs = find_violating_inequalities()
    total_time = time.time() - start_time

    if len(violating_ineqs) > 0:
        save_results_to_file(violating_ineqs, total_time)

        # 控制台简要展示 Top 3
        violating_ineqs.sort(key=lambda x: x['violation_ratio'], reverse=True)
        print("【控制台简要展示 - 违背最强的不等式 Top 3】")
        for i, ineq in enumerate(violating_ineqs[:3]):
            print(f"Top {i + 1}: 经典={ineq['classical_max']:.4f}, "
                  f"量子={ineq['quantum_max']:.4f}, "
                  f"违背比={ineq['violation_ratio']:.4f}x")
    else:
        print("\n未找到任何有量子违背的不等式。")
