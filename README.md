# evaluate-for-causalDAG

致力于多维度评估因果图的正确性 — a repository for multi-dimensional causal DAG accuracy evaluation.

## 项目结构 / Project Structure

```
evaluate_causal_dag/
    __init__.py
    methods/
        __init__.py
        falsify/               ← Method 1: Local Markov Condition testing
            __init__.py
            cache.py           ← CITestCache — reuse results across evaluations
            independence_tests.py  ← Fisher's Z, Mutual-Info CI tests
            triplet_selector.py    ← Extract triplets from a DAG
            evaluator.py       ← FalsifyGraphEvaluator (main entry-point)
tests/
    methods/
        test_falsify.py
requirements.txt
setup.py
```

## 第一种方法：基于局部马尔可夫条件的图伪证 / Method 1: Falsify via Local Markov Condition

### 思路 / Approach

受 [dowhy `falsify_graph`](https://github.com/py-why/dowhy/blob/main/dowhy/gcm/falsify.py) 启发，独立实现。

1. **选取三元组**：对 DAG 中每个节点 X，局部马尔可夫条件要求
   `X ⊥ NonDesc(X) \ Pa(X) | Pa(X)`。为每对 (X, Z) 生成三元组。
2. **条件独立性检验**：支持 Fisher's Z（线性、连续变量）和 Mutual-Info（离散变量）两种方法。
3. **缓存**：`CITestCache` 以 (X, Y, conditioning_set) 为对称键缓存检验结果；对略有修改的新图再次评估时，已计算的三元组直接命中缓存，只对新三元组重新检验。
4. **Suggestions（可选）**：对每条边 X→Y 检验 `X ⊥ Y | Pa(Y) \ {X}`，若独立则建议考虑移除该边。
5. **评估结论**：返回 `EvaluationResult`，含违反数量、违反率和 FALSIFIED / NOT FALSIFIED 结论。

### 快速上手 / Quick Start

```python
import networkx as nx
import pandas as pd
import numpy as np
from evaluate_causal_dag.methods.falsify import FalsifyGraphEvaluator, CITestCache

# 构造数据（链式结构 X → Y → Z）
rng = np.random.default_rng(42)
n = 500
x = rng.normal(size=n)
y = 0.8 * x + rng.normal(size=n)
z = 0.6 * y + rng.normal(size=n)
data = pd.DataFrame({"X": x, "Y": y, "Z": z})

# 待评估的因果图
dag = nx.DiGraph([("X", "Y"), ("Y", "Z")])

# 创建评估器（可传入共享缓存）
evaluator = FalsifyGraphEvaluator()

# 第一次评估
result = evaluator.evaluate(dag, data, include_suggestions=True)
print(result.summary())

# 对修改后的图再次评估 — 已有三元组从缓存命中，只计算新的
dag_modified = nx.DiGraph([("X", "Y"), ("Y", "Z"), ("X", "Z")])
result2 = evaluator.evaluate(dag_modified, data)
print(f"n_tests={result2.n_tests}, n_cached={result2.n_cached}")
```

### 安装 / Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### 运行测试 / Run Tests

```bash
pytest tests/ -v
```
