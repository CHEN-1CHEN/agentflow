# AgentFlow — Multi-Agent LLM Framework

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)


**AgentFlow** 是一个面向大模型（LLM）与智能体（Agent）的模块化框架，实现了**任务规划 → 工具调用 → 执行验证**的完整 Agent 工作流，集成了 **RAG 检索增强生成**、**Function Calling 工具系统**和**多智能体协作编排**。

---

## 核心特性

### 1. 多智能体协作系统（Multi-Agent Orchestration）
基于 **Planner → Executor → Reviewer** 三阶段流水线：

| 智能体 | 角色 | 核心技术 |
|--------|------|----------|
| **PlannerAgent** | 任务分解与规划 | Chain-of-Thought (CoT) 推理，将复杂任务拆解为原子化步骤 |
| **ExecutorAgent** | 工具驱动执行 | ReAct 模式，动态选择工具并执行，支持多轮工具调用 |
| **ReviewerAgent** | 质量验证与反馈 | 多维评估（完整性/正确性/质量），支持重试和纠正机制 |

### 2. RAG 检索增强生成（Retrieval-Augmented Generation）
完整的 RAG 流水线，解决 LLM 知识截止与幻觉问题：
- **文档加载**：支持 PDF、TXT、Markdown、HTML 多格式
- **文本分块**：语义感知的滑动窗口分块策略（chunk_size + overlap）
- **向量嵌入**：基于 Sentence-Transformers 的稠密向量生成
- **向量存储**：ChromaDB / FAISS 双后端支持
- **检索增强**：Top-K 语义检索 + 上下文注入 LLM Prompt

### 3. Function Calling 工具系统
LLM 驱动的自动化工具调用框架，遵循 OpenAI Function Calling 协议：
- **Calculator**：安全数学表达式求值
- **Python Executor**：沙箱化 Python 代码执行
- **Web Search**：DuckDuckGo 实时网页搜索
- **File Operations**：文件读写与工作区管理

### 4. 记忆管理（Memory System）
- **Short-Term Memory**：滑动窗口对话缓冲 + 自动摘要压缩
- **Long-Term Memory**：基于向量嵌入的持久化知识存储与语义检索

---

## 架构设计

```
                            ┌──────────────────────┐
                            │    Orchestrator       │
                            │  (Plan → Exec → Rvw)  │
                            └──────┬───────┬───────┘
                                   │       │
                   ┌───────────────┘       └───────────────┐
                   ▼                                       ▼
        ┌──────────────────┐                    ┌──────────────────┐
        │   PlannerAgent    │                    │  ReviewerAgent    │
        │  • CoT 推理       │                    │  • 多维评估        │
        │  • 任务分解       │                    │  • 反馈纠正        │
        │  • 依赖分析       │                    │  • 质量打分        │
        └──────────────────┘                    └──────────────────┘
                   │                                       ▲
                   │  plan[]                    review     │
                   ▼                                       │
        ┌──────────────────┐                    ┌──────────────────┐
        │  ExecutorAgent    │───────────────────▶│  ToolRegistry     │
        │  • ReAct Pattern  │   tool_calls       │  • Calculator     │
        │  • 工具调度       │                    │  • Python Exec    │
        │  • 多轮执行       │                    │  • Web Search     │
        └──────────────────┘                    │  • File Ops       │
                   │                             └──────────────────┘
                   │ retrieve
                   ▼
        ┌──────────────────┐
        │   RAG Pipeline    │
        │  Load → Chunk →   │
        │  Embed → Store →  │
        │  Retrieve          │
        └──────────────────┘
```

---

## 快速开始

### 安装

```bash
git clone https://github.com/CHEN-1CHEN/agentflow.git
cd agentflow
pip install -r requirements.txt
```

### 配置

编辑 `config.yaml` 或设置环境变量：

```bash
# 使用 OpenAI API
export OPENAI_API_KEY="sk-your-api-key"

# 或使用国产模型 API（如 Qwen / DeepSeek / 智谱）
# 修改 config.yaml 中的 base_url 和 model
```

### 运行 Demo

```bash
# Demo 1: 多智能体协作
python examples/demo_multi_agent.py

# Demo 2: RAG 检索增强生成
python examples/demo_rag.py

# Demo 3: Function Calling 工具调用
python examples/demo_tool_calling.py
```

---

## 代码示例

### 多智能体任务编排

```python
from agentflow.core import LLMClient, Orchestrator
from agentflow.tools import ToolRegistry, CalculatorTool, PythonExecutorTool

llm = LLMClient()
tools = ToolRegistry()
tools.register(CalculatorTool())
tools.register(PythonExecutorTool())

orchestrator = Orchestrator(llm=llm, tools=tools)

result = orchestrator.run("分析Q1销售数据：计算总收入、增长率和Q2预测")
print(result["summary"])
# {"total_steps": 4, "passed": 4, "failed": 0, "average_score": 92.5, "total_tool_calls": 3}
```

### RAG 问答系统

```python
from agentflow.rag import DocumentLoader, TextChunker, Embedder, VectorStore, Retriever

# 构建知识库
docs = DocumentLoader().load("./my_documents/")
chunks = TextChunker(chunk_size=512, chunk_overlap=64).chunk_documents(docs)
chunks = Embedder().embed_documents(chunks)
VectorStore(backend="chromadb").add(chunks)

# 检索增强问答
retriever = Retriever(embedder=Embedder(), vector_store=VectorStore())
results = retriever.retrieve("Transformer 的计算复杂度是多少？")
context = retriever.format_context(results)  # 注入 LLM Prompt
```

### Function Calling

```python
response = llm.chat_with_tools(
    messages=[{"role": "user", "content": "计算 sqrt(256) + 3.14 * 15"}],
    tools=tools.get_schemas(),
    tool_handlers=tools.get_handlers(),
)
print(response["answer"])
# "计算结果为 63.1"
# Tool calls: [{"tool": "calculator", "args": {"expression": "sqrt(256) + 3.14 * 15"}}]
```

---

## 项目结构

```
agentflow/
├── README.md
├── config.yaml                      # 全局配置（LLM/RAG/Agent/Memory/Tools）
├── setup.py
├── requirements.txt
├── src/agentflow/
│   ├── core/
│   │   ├── llm_client.py            # 统一 LLM 客户端（OpenAI/Ollama/国内API）
│   │   └── orchestrator.py          # 多Agent编排器 (Plan → Execute → Review)
│   ├── agents/
│   │   ├── base_agent.py            # Agent 抽象基类
│   │   ├── planner.py               # 规划Agent (CoT 任务分解)
│   │   ├── executor.py              # 执行Agent (ReAct + 工具调用)
│   │   └── reviewer.py              # 审查Agent (多维评估与反馈)
│   ├── rag/
│   │   ├── loader.py                # 文档加载器 (PDF/TXT/MD/HTML)
│   │   ├── chunker.py               # 语义感知文本分块
│   │   ├── embedder.py              # 向量嵌入 (Sentence-Transformers)
│   │   ├── store.py                 # 向量存储 (ChromaDB/FAISS)
│   │   └── retriever.py             # 检索器 + 上下文格式化
│   ├── tools/
│   │   ├── base.py                  # 工具基类 + 注册表 (OpenAI FC Schema)
│   │   ├── calculator.py            # 数学计算器
│   │   ├── python_executor.py       # Python 沙箱执行器
│   │   ├── search.py                # Web 搜索 (DuckDuckGo)
│   │   └── file_ops.py              # 文件操作工具
│   └── memory/
│       ├── buffer.py                # 短期记忆 (滑动窗口 + 摘要压缩)
│       └── store.py                 # 长期记忆 (向量持久化 + 语义检索)
├── examples/
│   ├── demo_multi_agent.py          # 多Agent协作演示
│   ├── demo_rag.py                  # RAG流水线演示
│   └── demo_tool_calling.py         # Function Calling演示
└── tests/
    └── __init__.py
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| **LLM 接口** | OpenAI API、Ollama、OpenAI-Compatible APIs（支持 Qwen/DeepSeek/智谱） |
| **Agent 框架** | 自研 Planner-Executor-Reviewer 多智能体编排 |
| **RAG** | LangChain 文档加载器、Sentence-Transformers 嵌入、ChromaDB/FAISS 向量库 |
| **工具系统** | OpenAI Function Calling 协议、沙箱 Python 执行、DuckDuckGo 搜索 |
| **记忆系统** | 滑动窗口短期记忆、向量嵌入长期记忆 |
| **推理策略** | Chain-of-Thought (CoT)、ReAct、Self-Critique |
| **深度学习** | PyTorch、Transformers |

---

## 核心依赖

- `openai` — LLM API 客户端
- `langchain` + `langchain-community` — 文档加载与 RAG 工具
- `chromadb` — 向量数据库
- `sentence-transformers` — 文本嵌入模型
- `pypdf2` — PDF 解析
- `tiktoken` — Token 计数

---

## 设计理念

1. **模块化**：各组件（Agent/RAG/Tools/Memory）独立可替换，遵循单一职责原则
2. **协议兼容**：工具系统遵循 OpenAI Function Calling 标准，可接入任何兼容 API
3. **工程化**：完整的项目结构、类型提示、配置管理，遵循 Python 最佳实践
4. **可扩展**：通过继承 `BaseAgent` / `BaseTool` 即可扩展自定义能力

---

## 后续规划

- [ ] 集成 LangGraph 实现有状态复杂 Agent 图
- [ ] 支持 LoRA 微调的小模型作为 Agent 推理后端
- [ ] 多模态 Agent（视觉理解 + 文档分析）
- [ ] Agent 评估 Benchmark 套件
- [ ] Web UI 交互界面

---

**Author**: 陈一千 | **Keywords**: LLM Agent, RAG, Function Calling, Multi-Agent, Task Orchestration, AI Agent Framework
