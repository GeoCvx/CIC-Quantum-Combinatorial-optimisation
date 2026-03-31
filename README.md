## 1. 项目简介

本项目实现了一个面向 **混合整数优化问题（MILP / MICP）** 的求解框架，支持：

- **MILP（Mixed Integer Linear Programming）**
- **MICP（Mixed Integer Convex Programming）**

核心思路：

> 将问题拆分为 **离散主问题（x） + 连续子问题（y）**，并通过统一 pipeline 进行求解。

其中：

- 离散变量 $x \in \{0,1\}^n$
- 连续变量 $y \in \mathbb{R}^n$

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

### 4.4 main

统一入口：

```
python main.py
```

自动完成：

- 读取数据
- 判断问题类型
- 调用对应 pipeline
- 输出结果

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
python scripts/run.py --input data/raw/problem_micp_1.json --mode hybrid
python scripts/run.py --input data/raw/problem_micp_1.json --mode classical
python scripts/run.py --input data/raw/problem_micp_1.json --mode quantum
```

或：

```
python main.py
```

------

### 批量运行（建议后续实现）

```
python scripts/run_all.py
```

------

## 7. 当前完成情况

- ✅ MILP 子问题（LP）已实现并测试通过
- ✅ MICP 子问题（QP）已实现并测试通过
- ✅ 子问题统一接口（router）完成
- ✅ 基础 pipeline 已打通
- ✅ 基础测试覆盖（可行性、自洽性、异常）

------

## 8. 后续优化方向

### 算法层

- 贪心初始化
- 局部搜索（flip / swap）
- Benders decomposition
- Outer Approximation (OA)

### 量子方向

- QAOA 求解主问题
- 量子 warm start
- 混合经典-量子 pipeline

