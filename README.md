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

