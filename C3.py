from bell_inequality_unified import create_chain_inequality, solve

# 1. 创建链式不等式 C3
ineq = create_chain_inequality(3)

# 2. 求解不等式（计算经典最大值和量子最大值）
result = solve(ineq)

# 3. 输出结果
print(f"链不等式 C3 经典最大值: {result['classical_max']}")
print(f"链不等式 C3 量子最大值: {result['quantum_max']}")
print(f"使用方法: {result['method_used']}")
