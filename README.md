# AgentFlow 鈥?Multi-Agent LLM Framework

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**AgentFlow** 鏄竴涓潰鍚戝ぇ妯″瀷锛圠LM锛変笌鏅鸿兘浣擄紙Agent锛夌殑妯″潡鍖栨鏋讹紝瀹炵幇浜?*浠诲姟瑙勫垝 鈫?宸ュ叿璋冪敤 鈫?鎵ц楠岃瘉**鐨勫畬鏁?Agent 宸ヤ綔娴侊紝闆嗘垚浜?**RAG 妫€绱㈠寮虹敓鎴?*銆?*Function Calling 宸ュ叿绯荤粺**鍜?*澶氭櫤鑳戒綋鍗忎綔缂栨帓**銆?
---

## 鏍稿績鐗规€?
### 1. 澶氭櫤鑳戒綋鍗忎綔绯荤粺锛圡ulti-Agent Orchestration锛?鍩轰簬 **Planner 鈫?Executor 鈫?Reviewer** 涓夐樁娈垫祦姘寸嚎锛?
| 鏅鸿兘浣?| 瑙掕壊 | 鏍稿績鎶€鏈?|
|--------|------|----------|
| **PlannerAgent** | 浠诲姟鍒嗚В涓庤鍒?| Chain-of-Thought (CoT) 鎺ㄧ悊锛屽皢澶嶆潅浠诲姟鎷嗚В涓哄師瀛愬寲姝ラ |
| **ExecutorAgent** | 宸ュ叿椹卞姩鎵ц | ReAct 妯″紡锛屽姩鎬侀€夋嫨宸ュ叿骞舵墽琛岋紝鏀寔澶氳疆宸ュ叿璋冪敤 |
| **ReviewerAgent** | 璐ㄩ噺楠岃瘉涓庡弽棣?| 澶氱淮璇勪及锛堝畬鏁存€?姝ｇ‘鎬?璐ㄩ噺锛夛紝鏀寔閲嶈瘯鍜岀籂姝ｆ満鍒?|

### 2. RAG 妫€绱㈠寮虹敓鎴愶紙Retrieval-Augmented Generation锛?瀹屾暣鐨?RAG 娴佹按绾匡紝瑙ｅ喅 LLM 鐭ヨ瘑鎴涓庡够瑙夐棶棰橈細
- **鏂囨。鍔犺浇**锛氭敮鎸?PDF銆乀XT銆丮arkdown銆丠TML 澶氭牸寮?- **鏂囨湰鍒嗗潡**锛氳涔夋劅鐭ョ殑婊戝姩绐楀彛鍒嗗潡绛栫暐锛坈hunk_size + overlap锛?- **鍚戦噺宓屽叆**锛氬熀浜?Sentence-Transformers 鐨勭瀵嗗悜閲忕敓鎴?- **鍚戦噺瀛樺偍**锛欳hromaDB / FAISS 鍙屽悗绔敮鎸?- **妫€绱㈠寮?*锛歍op-K 璇箟妫€绱?+ 涓婁笅鏂囨敞鍏?LLM Prompt

### 3. Function Calling 宸ュ叿绯荤粺
LLM 椹卞姩鐨勮嚜鍔ㄥ寲宸ュ叿璋冪敤妗嗘灦锛岄伒寰?OpenAI Function Calling 鍗忚锛?- **Calculator**锛氬畨鍏ㄦ暟瀛﹁〃杈惧紡姹傚€?- **Python Executor**锛氭矙绠卞寲 Python 浠ｇ爜鎵ц
- **Web Search**锛欴uckDuckGo 瀹炴椂缃戦〉鎼滅储
- **File Operations**锛氭枃浠惰鍐欎笌宸ヤ綔鍖虹鐞?
### 4. 璁板繂绠＄悊锛圡emory System锛?- **Short-Term Memory**锛氭粦鍔ㄧ獥鍙ｅ璇濈紦鍐?+ 鑷姩鎽樿鍘嬬缉
- **Long-Term Memory**锛氬熀浜庡悜閲忓祵鍏ョ殑鎸佷箙鍖栫煡璇嗗瓨鍌ㄤ笌璇箟妫€绱?
---

## 鏋舵瀯璁捐

```
                            鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?                            鈹?   Orchestrator       鈹?                            鈹? (Plan 鈫?Exec 鈫?Rvw)  鈹?                            鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹?                                   鈹?      鈹?                   鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?      鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?                   鈻?                                      鈻?        鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?                   鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?        鈹?  PlannerAgent    鈹?                   鈹? ReviewerAgent    鈹?        鈹? 鈥?CoT 鎺ㄧ悊       鈹?                   鈹? 鈥?澶氱淮璇勪及        鈹?        鈹? 鈥?浠诲姟鍒嗚В       鈹?                   鈹? 鈥?鍙嶉绾犳        鈹?        鈹? 鈥?渚濊禆鍒嗘瀽       鈹?                   鈹? 鈥?璐ㄩ噺鎵撳垎        鈹?        鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?                   鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?                   鈹?                                      鈻?                   鈹? plan[]                    review     鈹?                   鈻?                                      鈹?        鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?                   鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?        鈹? ExecutorAgent    鈹傗攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈻垛攤  ToolRegistry     鈹?        鈹? 鈥?ReAct Pattern  鈹?  tool_calls       鈹? 鈥?Calculator     鈹?        鈹? 鈥?宸ュ叿璋冨害       鈹?                   鈹? 鈥?Python Exec    鈹?        鈹? 鈥?澶氳疆鎵ц       鈹?                   鈹? 鈥?Web Search     鈹?        鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?                   鈹? 鈥?File Ops       鈹?                   鈹?                            鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?                   鈹?retrieve
                   鈻?        鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?        鈹?  RAG Pipeline    鈹?        鈹? Load 鈫?Chunk 鈫?  鈹?        鈹? Embed 鈫?Store 鈫? 鈹?        鈹? Retrieve          鈹?        鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?```

---

## 蹇€熷紑濮?
### 瀹夎

```bash
git clone https://github.com/YOUR_USERNAME/agentflow.git
cd agentflow
pip install -r requirements.txt
```

### 閰嶇疆

缂栬緫 `config.yaml` 鎴栬缃幆澧冨彉閲忥細

```bash
# 浣跨敤 OpenAI API
export OPENAI_API_KEY="sk-your-api-key"

# 鎴栦娇鐢ㄥ浗浜фā鍨?API锛堝 Qwen / DeepSeek / 鏅鸿氨锛?# 淇敼 config.yaml 涓殑 base_url 鍜?model
```

### 杩愯 Demo

```bash
# Demo 1: 澶氭櫤鑳戒綋鍗忎綔
python examples/demo_multi_agent.py

# Demo 2: RAG 妫€绱㈠寮虹敓鎴?python examples/demo_rag.py

# Demo 3: Function Calling 宸ュ叿璋冪敤
python examples/demo_tool_calling.py
```

---

## 浠ｇ爜绀轰緥

### 澶氭櫤鑳戒綋浠诲姟缂栨帓

```python
from agentflow.core import LLMClient, Orchestrator
from agentflow.tools import ToolRegistry, CalculatorTool, PythonExecutorTool

llm = LLMClient()
tools = ToolRegistry()
tools.register(CalculatorTool())
tools.register(PythonExecutorTool())

orchestrator = Orchestrator(llm=llm, tools=tools)

result = orchestrator.run("鍒嗘瀽Q1閿€鍞暟鎹細璁＄畻鎬绘敹鍏ャ€佸闀跨巼鍜孮2棰勬祴")
print(result["summary"])
# {"total_steps": 4, "passed": 4, "failed": 0, "average_score": 92.5, "total_tool_calls": 3}
```

### RAG 闂瓟绯荤粺

```python
from agentflow.rag import DocumentLoader, TextChunker, Embedder, VectorStore, Retriever

# 鏋勫缓鐭ヨ瘑搴?docs = DocumentLoader().load("./my_documents/")
chunks = TextChunker(chunk_size=512, chunk_overlap=64).chunk_documents(docs)
chunks = Embedder().embed_documents(chunks)
VectorStore(backend="chromadb").add(chunks)

# 妫€绱㈠寮洪棶绛?retriever = Retriever(embedder=Embedder(), vector_store=VectorStore())
results = retriever.retrieve("Transformer 鐨勮绠楀鏉傚害鏄灏戯紵")
context = retriever.format_context(results)  # 娉ㄥ叆 LLM Prompt
```

### Function Calling

```python
response = llm.chat_with_tools(
    messages=[{"role": "user", "content": "璁＄畻 sqrt(256) + 3.14 * 15"}],
    tools=tools.get_schemas(),
    tool_handlers=tools.get_handlers(),
)
print(response["answer"])
# "璁＄畻缁撴灉涓?63.1"
# Tool calls: [{"tool": "calculator", "args": {"expression": "sqrt(256) + 3.14 * 15"}}]
```

---

## 椤圭洰缁撴瀯

```
agentflow/
鈹溾攢鈹€ README.md
鈹溾攢鈹€ config.yaml                      # 鍏ㄥ眬閰嶇疆锛圠LM/RAG/Agent/Memory/Tools锛?鈹溾攢鈹€ setup.py
鈹溾攢鈹€ requirements.txt
鈹溾攢鈹€ src/agentflow/
鈹?  鈹溾攢鈹€ core/
鈹?  鈹?  鈹溾攢鈹€ llm_client.py            # 缁熶竴 LLM 瀹㈡埛绔紙OpenAI/Ollama/鍥藉唴API锛?鈹?  鈹?  鈹斺攢鈹€ orchestrator.py          # 澶欰gent缂栨帓鍣?(Plan 鈫?Execute 鈫?Review)
鈹?  鈹溾攢鈹€ agents/
鈹?  鈹?  鈹溾攢鈹€ base_agent.py            # Agent 鎶借薄鍩虹被
鈹?  鈹?  鈹溾攢鈹€ planner.py               # 瑙勫垝Agent (CoT 浠诲姟鍒嗚В)
鈹?  鈹?  鈹溾攢鈹€ executor.py              # 鎵цAgent (ReAct + 宸ュ叿璋冪敤)
鈹?  鈹?  鈹斺攢鈹€ reviewer.py              # 瀹℃煡Agent (澶氱淮璇勪及涓庡弽棣?
鈹?  鈹溾攢鈹€ rag/
鈹?  鈹?  鈹溾攢鈹€ loader.py                # 鏂囨。鍔犺浇鍣?(PDF/TXT/MD/HTML)
鈹?  鈹?  鈹溾攢鈹€ chunker.py               # 璇箟鎰熺煡鏂囨湰鍒嗗潡
鈹?  鈹?  鈹溾攢鈹€ embedder.py              # 鍚戦噺宓屽叆 (Sentence-Transformers)
鈹?  鈹?  鈹溾攢鈹€ store.py                 # 鍚戦噺瀛樺偍 (ChromaDB/FAISS)
鈹?  鈹?  鈹斺攢鈹€ retriever.py             # 妫€绱㈠櫒 + 涓婁笅鏂囨牸寮忓寲
鈹?  鈹溾攢鈹€ tools/
鈹?  鈹?  鈹溾攢鈹€ base.py                  # 宸ュ叿鍩虹被 + 娉ㄥ唽琛?(OpenAI FC Schema)
鈹?  鈹?  鈹溾攢鈹€ calculator.py            # 鏁板璁＄畻鍣?鈹?  鈹?  鈹溾攢鈹€ python_executor.py       # Python 娌欑鎵ц鍣?鈹?  鈹?  鈹溾攢鈹€ search.py                # Web 鎼滅储 (DuckDuckGo)
鈹?  鈹?  鈹斺攢鈹€ file_ops.py              # 鏂囦欢鎿嶄綔宸ュ叿
鈹?  鈹斺攢鈹€ memory/
鈹?      鈹溾攢鈹€ buffer.py                # 鐭湡璁板繂 (婊戝姩绐楀彛 + 鎽樿鍘嬬缉)
鈹?      鈹斺攢鈹€ store.py                 # 闀挎湡璁板繂 (鍚戦噺鎸佷箙鍖?+ 璇箟妫€绱?
鈹溾攢鈹€ examples/
鈹?  鈹溾攢鈹€ demo_multi_agent.py          # 澶欰gent鍗忎綔婕旂ず
鈹?  鈹溾攢鈹€ demo_rag.py                  # RAG娴佹按绾挎紨绀?鈹?  鈹斺攢鈹€ demo_tool_calling.py         # Function Calling婕旂ず
鈹斺攢鈹€ tests/
    鈹斺攢鈹€ __init__.py
```

---

## 鎶€鏈爤

| 灞傜骇 | 鎶€鏈?|
|------|------|
| **LLM 鎺ュ彛** | OpenAI API銆丱llama銆丱penAI-Compatible APIs锛堟敮鎸?Qwen/DeepSeek/鏅鸿氨锛?|
| **Agent 妗嗘灦** | 鑷爺 Planner-Executor-Reviewer 澶氭櫤鑳戒綋缂栨帓 |
| **RAG** | LangChain 鏂囨。鍔犺浇鍣ㄣ€丼entence-Transformers 宓屽叆銆丆hromaDB/FAISS 鍚戦噺搴?|
| **宸ュ叿绯荤粺** | OpenAI Function Calling 鍗忚銆佹矙绠?Python 鎵ц銆丏uckDuckGo 鎼滅储 |
| **璁板繂绯荤粺** | 婊戝姩绐楀彛鐭湡璁板繂銆佸悜閲忓祵鍏ラ暱鏈熻蹇?|
| **鎺ㄧ悊绛栫暐** | Chain-of-Thought (CoT)銆丷eAct銆丼elf-Critique |
| **娣卞害瀛︿範** | PyTorch銆乀ransformers |

---

## 鏍稿績渚濊禆

- `openai` 鈥?LLM API 瀹㈡埛绔?- `langchain` + `langchain-community` 鈥?鏂囨。鍔犺浇涓?RAG 宸ュ叿
- `chromadb` 鈥?鍚戦噺鏁版嵁搴?- `sentence-transformers` 鈥?鏂囨湰宓屽叆妯″瀷
- `pypdf2` 鈥?PDF 瑙ｆ瀽
- `tiktoken` 鈥?Token 璁℃暟

---

## 璁捐鐞嗗康

1. **妯″潡鍖?*锛氬悇缁勪欢锛圓gent/RAG/Tools/Memory锛夌嫭绔嬪彲鏇挎崲锛岄伒寰崟涓€鑱岃矗鍘熷垯
2. **鍗忚鍏煎**锛氬伐鍏风郴缁熼伒寰?OpenAI Function Calling 鏍囧噯锛屽彲鎺ュ叆浠讳綍鍏煎 API
3. **宸ョ▼鍖?*锛氬畬鏁寸殑椤圭洰缁撴瀯銆佺被鍨嬫彁绀恒€侀厤缃鐞嗭紝閬靛惊 Python 鏈€浣冲疄璺?4. **鍙墿灞?*锛氶€氳繃缁ф壙 `BaseAgent` / `BaseTool` 鍗冲彲鎵╁睍鑷畾涔夎兘鍔?
---

## 鍚庣画瑙勫垝

- [ ] 闆嗘垚 LangGraph 瀹炵幇鏈夌姸鎬佸鏉?Agent 鍥?- [ ] 鏀寔 LoRA 寰皟鐨勫皬妯″瀷浣滀负 Agent 鎺ㄧ悊鍚庣
- [ ] 澶氭ā鎬?Agent锛堣瑙夌悊瑙?+ 鏂囨。鍒嗘瀽锛?- [ ] Agent 璇勪及 Benchmark 濂椾欢
- [ ] Web UI 浜や簰鐣岄潰

---

## License

MIT License

---

**Author**: Siqi Chen | **Keywords**: LLM Agent, RAG, Function Calling, Multi-Agent, Task Orchestration, AI Agent Framework
