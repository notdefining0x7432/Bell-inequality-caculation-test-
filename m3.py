from bell_inequality_unified import create_mermin_inequality, solve

# 1. 创建 Mermin M3 不等式
ineq = create_mermin_inequality(3)

# 2. 求解不等式（计算经典最大值和量子最大值）
result = solve(ineq)

# 3. 输出结果
print(f"Mermin M3 经典最大值: {result['classical_max']}")
print(f"Mermin M3 量子最大值: {result['quantum_max']}")
print(f"使用方法: {result['method_used']}")