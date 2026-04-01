import numpy as np

def solve_with_greedy(example: dict) -> dict:
    # 1. 解析输入字典
    n = example['product_count']
    m = example['resource_count']
    p = np.array(example['price'])
    f = np.array(example['fixed_cost'])
    alpha = np.array(example['alpha'])
    beta = np.array(example['beta'])
    D = np.array(example['max_demand'])
    R = np.array(example['resource_limit'])
    A = np.array(example['consumption_matrix'])
    
    # 2. 寻找抛物线顶点 (利润最大化点)
    y_peak = np.zeros(n)
    for i in range(n):
        if beta[i] > 0:
            y_peak[i] = max(0, (p[i] - alpha[i]) / (2 * beta[i]))
        else:
            # 当 beta 为 0 时退化为线性。若边际利润为正，则理想产量无限大；为负则不生产
            y_peak[i] = float('inf') if (p[i] - alpha[i]) > 0 else 0.0
            
    ideal_profits = np.zeros(n)
    for i in range(n):
        # 计算在只有初始资源的情况下的最大物理产量
        max_y_physical = min([R[j] / A[j, i] if A[j, i] > 0 else float('inf') for j in range(m)])
        # 理想产量 = min(需求, 经济顶点, 物理极限)
        best_y_ideal = min(D[i], y_peak[i], max_y_physical)
        
        # 计算理想最高利润
        prof = p[i]*best_y_ideal - (alpha[i]*best_y_ideal + beta[i]*(best_y_ideal**2)) - f[i]
        ideal_profits[i] = prof

    # 3. 按照理想最高利润从高到低排序 (贪心优先级)
    sorted_indices = np.argsort(ideal_profits)[::-1]
    
    # 4. 依据优先级依次分配真实资源
    current_R = R.copy()
    x_greedy = np.zeros(n, dtype=int)
    y_greedy = np.zeros(n)
    total_profit = 0.0

    for i in sorted_indices:
        # 如果理论最大利润都是亏的，直接跳过
        if ideal_profits[i] <= 0:
            continue 
            
        # 计算当前剩余资源下能生产的最大物理产量
        max_y_actual = min([current_R[j] / A[j, i] if A[j, i] > 0 else float('inf') for j in range(m)])
        # 实际决定生产的量
        best_y_actual = min(D[i], y_peak[i], max_y_actual)
        
        # 验证该产量下是否依然盈利 (覆盖固定成本)
        actual_prof = p[i]*best_y_actual - (alpha[i]*best_y_actual + beta[i]*(best_y_actual**2)) - f[i]
        
        if actual_prof > 0:
            # 决定生产
            x_greedy[i] = 1
            y_greedy[i] = best_y_actual
            total_profit += actual_prof
            # 扣除消耗的资源
            for j in range(m):
                current_R[j] -= A[j, i] * best_y_actual

    # 5. 封装输出结果
    result_dict = {
        'x': x_greedy.tolist(),
        'y': np.round(y_greedy, 2).tolist(),
        'Z': round(float(total_profit), 2),
        'r': np.round(current_R, 2).tolist()
    }
    
    return result_dict


if __name__ == '__main__':
    from pathlib import Path
    import json
    cic_path = Path(__file__).parent / 'cic'
    with open(cic_path / 'example_problem.json', 'r') as f:
        problem = json.load(f)
    result = solve_with_greedy(problem)
    print(result)