## 1. 项目简介

本项目实现了一个面向 **混合整数优化问题（MILP / MICP）** 的求解框架，支持：
- **MILP（Mixed Integer Linear Programming）**
- **MICP（Mixed Integer Convex Programming）**

核心思路：
> 将问题拆分为 **离散主问题（x） + 连续子问题（y）**，并通过统一 pipeline 进行求解。


------

## 2. 方法概述

### 2.1 子问题分解

固定 $x$ 后：

#### MILP（β = 0）

转化为线性规划：
$$
\max (p - \alpha)^T y
$$

- 使用：`scipy.optimize.linprog (HiGHS)`

------

#### MICP（β > 0）

转化为凸二次优化问题：
$$
\max \sum_i \left((p_i - \alpha_i)y_i - \beta_i y_i^2\right)
$$

- 方法：
  - 拉格朗日对偶
  - 投影梯度下降
  - 回溯线搜索
  - 数值修复

------

### 2.2 主问题（x 的搜索）

当前实现为：

- 随机搜索（baseline）

后续可扩展：

- 贪心策略
- 局部搜索
- Benders / OA
- 量子优化（QAOA 等）

------

### 2.3 整体流程

```
输入 JSON
   ↓
识别问题类型（MILP / MICP）
   ↓
生成候选 x（master）
   ↓
求解子问题（LP / QP）
   ↓
计算目标值 Z
   ↓
更新最优解
   ↓
输出结果
```

------

## 3. 项目结构

```
quantum_opt/
├─ README.md
├─ main.py
│
├─ data/
│  ├─ raw/              # 赛题数据
│  └─ examples/         # 官方样例
│
├─ algorithms/
│  ├─ subproblem/       # 连续子问题求解（核心数值层）
│  │  ├─ router.py      # MILP / MICP 自动分流
│  │  ├─ milp_lp/       # LP 子问题（scipy）
│  │  └─ micp_qp/       # QP 子问题（对偶+PGD）
│  │
│  ├─ master/           # 离散变量 x 的生成
│  │  ├─ classical_master.py
│  │  └─ quantum_master.py   # （预留）
│  │
│  └─ pipeline/         # 总流程
│     ├─ milp_pipeline.py
│     └─ micp_pipeline.py
│
├─ scripts/             # 运行脚本
│  ├─ run.py
│  └─ run_all.py        # （可扩展）
│
├─ outputs/             # 输出结果
│  ├─ results/
│  └─ submissions/
│
└─ tests/               # 单元测试
```

------

## 4. 核心模块说明

### 4.1 subproblem

统一接口：

```
evaluate_subproblem(problem_dict, x)
```

自动分流：

- MILP → `milp_lp.evaluate_x`
- MICP → `micp_qp.evaluate_x`

------

### 4.2 master

负责生成候选解 $x$

当前实现：

- 随机二进制采样

------

### 4.3 pipeline

负责整体求解逻辑：

```
solve_milp(problem_dict)
solve_micp(problem_dict)
```

------

## 5. 输入 / 输出格式

### 输入（JSON）

```
{
  "product_count": n,
  "resource_count": m,
  "price": [...],
  "fixed_cost": [...],
  "alpha": [...],
  "beta": [...],
  "max_demand": [...],
  "resource_limit": [...],
  "consumption_matrix": [...]
}
```

------

### 输出

```
{
  "x": [...],
  "y": [...],
  "Z": ...,
  "r": [...]
}
```

------

## 6. 如何运行

### 单个实例

```
python scripts/run.py --input data/raw/problem_micp_1.json --mode iterative_hybrid
python scripts/run.py --input data/raw/problem_micp_1.json --mode classical
python scripts/run.py --input data/raw/problem_micp_1.json --mode quantum
```

------

### 批量运行

```
python .\scripts\run_batch.py --input_dir data/raw --output_dir output
```

### baseline 对比
```
python /scripts/compare_baseline.py -input data/raw/problem_micp1.json
```

### exact 对比
```
python /scripts/compare_with_exct_opt.py -input data/raw/problem_micp1.json
```
------

