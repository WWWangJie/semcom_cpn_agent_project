# SemCom-CPN Agent Simulation Project

面向 **6G 语义通信（Semantic Communication, SemCom）与算力网络（Computing Power Network, CPN）融合** 的轻量级仿真项目。

本项目围绕“Token + 有限块长（FBL）+ 语义恢复 + 多节点算力调度 + Agent 决策”构建端到端可靠性仿真闭环，可用于科研方案验证、算法原型设计、论文实验雏形构建和技术汇报展示。

## 1. 项目目标

本项目用于验证如下研究思路：

1. 将传统 MEC/CPN 中“基于 CPU 频率的算力分配”抽象为“基于 Token 的语义资源分配”；
2. 将语义 Token 传输过程与 FBL 物理层误码概率建模结合；
3. 将终端语义恢复准确率建模为 Token 数量、任务复杂度、信道可靠性、算力节点能力的联合函数；
4. 通过 Agent 调度器在多终端、多边缘算力节点、多信道状态下进行资源分配；
5. 对比基线策略，观察 Token-aware Agent 调度在低时延和高可靠性约束下的性能优势。

## 2. 项目结构

```text
semcom_cpn_agent_project/
├── semcom_cpn/
│   ├── __init__.py
│   ├── config.py              # 全局仿真参数
│   ├── entities.py            # 终端、任务、信道、算力节点等实体
│   ├── fbl.py                 # 有限块长物理层误码概率模型
│   ├── semantic_model.py      # Token-语义准确率非线性映射
│   ├── schedulers.py          # Agent 调度器与基线调度器
│   ├── simulator.py           # 系统级仿真闭环
│   ├── metrics.py             # 指标统计与汇总
│   └── main.py                # 命令行入口
├── scripts/
│   └── run_experiment.py      # 一键运行实验
├── examples/
│   └── minimal_example.py     # 最小调用示例
├── requirements.txt
└── README.md
```

## 3. 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行默认实验

```bash
python -m semcom_cpn.main --episodes 100 --seed 42
```

或者：

```bash
python scripts/run_experiment.py
```

### 最小示例

```bash
python examples/minimal_example.py
```

## 4. 模型说明

### 4.1 Token 资源抽象

每个语义任务不再直接用原始比特数描述，而是抽象为语义 Token 数：

\[
T_i \in [T_i^{\min}, T_i^{\max}]
\]

Token 数越高，语义恢复准确率越高，但会带来更高传输负载和算力消耗。

### 4.2 FBL 物理层可靠性

传输误块率采用 Polyanskiy 有限块长近似形式：

\[
\varepsilon \approx Q\left(
\frac{\sqrt{m}(C(\gamma)-R)}{\sqrt{V(\gamma)}} \ln 2
\right)
\]

其中，\(m\) 为块长，\(\gamma\) 为信噪比，\(R\) 为编码速率，\(C(\gamma)\) 为信道容量，\(V(\gamma)\) 为信道色散。

### 4.3 语义恢复准确率

语义准确率采用饱和型非线性函数：

\[
A(T) = A_{\max}\left(1-\exp(-\alpha T)\right)
\]

并进一步考虑任务复杂度、计算节点能力和物理层传输成功率的影响。

### 4.4 端到端可靠性

端到端成功概率由物理层传输成功率、语义恢复准确率、计算完成概率共同决定：

\[
P_{\rm e2e} = (1-\varepsilon_{\rm fbl}) \cdot A_{\rm sem} \cdot P_{\rm comp}
\]

端到端失败概率为：

\[
\varepsilon_{\rm e2e} = 1 - P_{\rm e2e}
\]

## 5. 调度器

项目内置三类调度器：

1. `RandomScheduler`：随机选择 Token 数和边缘节点；
2. `GreedyChannelScheduler`：优先选择信道好的节点；
3. `TokenAwareAgentScheduler`：综合信道、Token、算力负载和语义收益进行决策，可视作轻量级 AI Agent 调度器。

## 6. 输出指标

仿真输出包括：

- 平均端到端可靠性；
- 平均语义准确率；
- 平均 FBL 误块率；
- 平均 Token 数；
- 平均计算时延；
- 任务成功率；
- 每个调度器的对比结果。

## 7. 适合扩展的方向

你可以在此基础上继续扩展：

1. 将启发式 Agent 替换为强化学习 Agent；
2. 接入真实论文中的语义准确率-Token 拟合曲线；
3. 增加多跳中继、RIS、NOMA、RSMA 等 6G 物理层机制；
4. 将 CPU/算力模型进一步替换为大模型推理中的 KV Cache、Token/s、上下文窗口等指标；
5. 引入知识库模块，让 Agent 自动从论文中提取模型参数。
