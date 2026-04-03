本项目面向 **混合整数优化问题（MILP / MICP）**，构建了一套 **经典 + 量子（QAOA）混合求解框架**，用于在复杂组合空间中高效搜索高质量解。

------

# 🚀 项目目标

针对如下问题：

- MILP（Mixed-Integer Linear Programming）
- MICP（Mixed-Integer Convex Programming）

目标是：

> 在离散变量 $x \in \{0,1\}^n$ 与连续变量 $y$ 联合优化问题中，设计高效的混合求解算法。

------

# 🧠 方法总览（核心思想）

本项目采用：

> **“离散变量启发式搜索 + 连续子问题精确求解” 的混合优化框架**

整体流程如下：

```
Problem JSON
    ↓
Pipeline
    ↓
Master（搜索 x）
    ├── Classical（多起点局部搜索）
    ├── Quantum（QAOA 生成候选）
    ├── Candidate Selection（筛选）
    ├── Local Refine（局部优化）
    └── Iterative Feedback（迭代更新）
    ↓
Subproblem（固定 x 求最优 y）
    ├── LP（MILP）
    └── QP（MICP）
    ↓
输出最优 (x, y, Z)
```

------

# 📂 项目结构

```
CIC/
│
├── algorithms/
│   ├── pipeline/           		    # 总流程控制（MILP / MICP）
│   ├── master/              		    # 离散变量 x 的搜索
│   │   ├── hybrid_master.py
│   │   ├── iterative_hybrid_master.py
│   │   ├── classical_master.py
│   │   └── quantum_master.py
│   │
│   ├── quantum/            		    # 量子模块（QAOA）
│   │   ├── qubo_builder.py
│   │   ├── ising_mapping.py
│   │   ├── qaoa_circuit.py
│   │   └── qaoa_solver.py
│   │
│   ├── candidate/     		            # 候选解筛选
│   ├── feedback/        		        # bias / tabu 更新
│   ├── local_search/          		    # 局部精修
│   │
│   ├── subproblem/            		    # 连续子问题求解
│   │   ├── milp_lp/
│   │   └── micp_qp/
│   │
│   └── utils/
│
├── scripts/
│   ├── run.py                 			# 主运行脚本
│   ├── run.py                 			# 批量运行脚本
│   ├── compare_baseline.py    			# baseline 对比
│   └── compare_with_exacy_opt.py       # 与最优解对比
│
├── data/
│   ├── raw/                  		    # 输入数据
│   └── example/          		        # 示例数据
│
├── output/              		        # 输出结果（自动生成）
│
└── test/                      			# 单元测试
```

------

# ⚙️ 核心模块说明

## 1️⃣ Pipeline

- `milp_pipeline.py`
- `micp_pipeline.py`

作用：

```
统一调度：
- master（找 x）
- subproblem（求 y）
```

------

## 2️⃣ Master（核心搜索）

### 模式支持：

| 模式             | 说明          |
| ---------------- | ------------- |
| classical        | 纯经典搜索    |
| quantum          | 纯 QAOA       |
| hybrid           | 单次混合      |
| iterative_hybrid | 多轮迭代混合⭐ |

------

### iterative_hybrid 机制：

```
循环：
  1. Classical warm start（计算开销大，暂时舍弃）
  2. QAOA 生成候选
  3. Candidate selection
  4. Subproblem evaluation
  5. Local refine
  6. 更新 bias / tabu / elite
```

特点：

- 避免重复搜索
- 持续改进解质量
- 结合量子采样与经典优化

------

## 3️⃣ Quantum（QAOA）

核心文件：

- `qubo_builder.py`
- `qaoa_solver.py`

作用：

```
x → QUBO → Ising → QAOA → bitstring
```

当前实现：

- p=1 QAOA
- 参数网格搜索（gamma / beta）
- 输出候选解分布

------

## 4️⃣ Subproblem（连续优化）

给定 $x$，求：

```
max Z(x, y)
```

实现：

| 类型 | 方法 |
| ---- | ---- |
| MILP | LP   |
| MICP | QP   |

👉 使用精确求解（不是近似）

------

# 📥 输入格式

JSON字段说明：

| 字段               | 含义     |
| ------------------ | -------- |
| product_count      | 产品数量 |
| resource_count     | 资源数量 |
| price              | 单位收益 |
| fixed_cost         | 固定成本 |
| alpha / beta       | 成本函数 |
| max_demand         | 最大需求 |
| resource_limit     | 资源约束 |
| consumption_matrix | 资源消耗 |

# ▶️ 使用方法

## 单个实例

```
python scripts/run.py --input data/raw/problem_micp_1.json --mode classical
python scripts/run.py --input data/raw/problem_micp_1.json --mode quantum
python scripts/run.py --input data/raw/problem_micp_1.json --mode hybrid
python scripts/run.py --input data/raw/problem_micp_1.json --mode iterative_hybrid
```

------

## 批量运行

```
python .\scripts\run_batch.py --input_dir data/raw
```

输出：

```
output/*.json
```

输出格式：

```
{
  "x": [0,1,1,...],
  "y": [...],
  "Z": 20084.103448,
  "r": [...]
}
```

------

# 📊 对比工具

## baseline 对比

```
python scripts/compare_baseline.py --input data/raw/problem_milp_1.json
```

------

## 最优解对比（小规模）

```
python scripts/compare_with_exact_opt.py --input data/raw/problem_milp_1.json
```
