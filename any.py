from bell_inequality_unified import BellInequality, solve

# ==========================================================
#                       【 配置区 】
#           计算新的不等式时，只需修改这里的参数
# ==========================================================

# 1. 不等式名称（仅用于打印标识，随便起名）
INEQUALITY_NAME = "我的自定义不等式"

# 2. 参与方数（2代表两方如Alice/Bob，3代表三方如Alice/Bob/Charlie）
N_PARTIES = 3

# 3. 每方的测量数量（列表长度必须等于 N_PARTIES）
# 例如：[2, 2] 表示 Alice有2个测量，Bob有2个测量
# 例如：[3, 2] 表示 Alice有3个测量，Bob有2个测量
MEASUREMENTS = [2, 2, 2]

# 4. 关联函数系数（核心！）
# 规则：键是元组，长度等于 N_PARTIES。元组里的数字代表测量编号（从0开始：0=第1个测量，1=第2个测量...）
# 规则：值是该关联项前面的系数（通常是 1 或 -1，也可以是小数）
COEFFICIENTS = {
    # --- 三方示例格式 (PartyA, PartyB, PartyC) ---
    (0, 0, 0): 1,  # 代表 ⟨A1 B1 C1⟩，系数为 +1
    (0, 1, 1): -1,  # 代表 ⟨A1 B2 C2⟩，系数为 -1
    (1, 0, 1): -1,  # 代表 ⟨A2 B1 C2⟩，系数为 -1
    (1, 1, 0): -1,  # 代表 ⟨A2 B2 C1⟩，系数为 -1

    # --- 如果是两方，格式则是 (PartyA, PartyB) ---
    # (0, 0): 1,   # 代表 ⟨A1 B1⟩
    # (0, 1): 1,   # 代表 ⟨A1 B2⟩
    # (1, 0): 1,   # 代表 ⟨A2 B1⟩
    # (1, 1): -1,  # 代表 ⟨A2 B2⟩
}

# 5. 高级参数（通常保持默认即可，遇到多方复杂不等式时可调整）
METHOD = 'auto'  # 'auto'(自动), 'toqito'(仅两方/严格上界), 'qutip'(多方/需优化)
OPTIMIZE = True  # 是否开启测量角度优化（多方不等式建议设为 True 以找到最大违背）
N_RANDOM = 2000  # 随机采样次数（仅 optimize=True 时有效，越大越准但越慢）


# ==========================================================
#                       【 执行区 】
#                 以下代码不需要做任何修改
# ==========================================================

def main():
    print(f"正在计算不等式: {INEQUALITY_NAME} ...")

    # 1. 创建不等式对象
    ineq = BellInequality(
        n_parties=N_PARTIES,
        measurements=MEASUREMENTS,
        name=INEQUALITY_NAME
    )

    # 2. 填入系数
    ineq.set_coefficients(COEFFICIENTS)

    # 3. 执行求解
    result = solve(
        ineq,
        method=METHOD,
        optimize=OPTIMIZE,
        n_random=N_RANDOM
    )

    # 4. 输出结果
    print("\n" + "=" * 45)
    print(f" 不等式名称 : {INEQUALITY_NAME}")
    print(f" 拓扑结构   : {N_PARTIES}方, 测量数={MEASUREMENTS}")
    print(f" 经典最大值 : {result['classical_max']:.6f}")
    print(f" 量子最大值 : {result['quantum_max']:.6f}")
    print(f" 量子违背比 : {result['quantum_max'] / result['classical_max']:.3f} 倍")
    print(f" 实际使用方法: {result['method_used']}")
    print("=" * 45 + "\n")

    return result


if __name__ == "__main__":
    main()
