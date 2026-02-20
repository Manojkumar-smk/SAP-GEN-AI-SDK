# SAP Generative AI Hub SDK — Full Interview Learning Guide

> A complete Python reference covering **Orchestration**, **RAG / Grounding**, **Data Masking**, **Content Filtering**, and **LangChain integration** — with line-by-line comments for interview preparation.

---

## ⚠️ Important — Package Name Change

| Status | Package |
|---|---|
| ❌ Deprecated (2024) | `generative-ai-hub-sdk` |
| ✅ Current | `sap-ai-sdk-gen` |

---

## 📦 Installation

```bash
# Core SAP Gen AI SDK
pip install "sap-ai-sdk-gen[all]"

# LangChain integration (optional)
pip install langchain-community langchain-text-splitters faiss-cpu
```

---

## ⚙️ Configuration

Create `~/.aicore/config.json` with your SAP AI Core credentials from SAP BTP Cockpit:

```json
{
    "AICORE_AUTH_URL":                       "https://<tenant>.authentication.sap.hana.ondemand.com/oauth/token",
    "AICORE_CLIENT_ID":                      "your-client-id",
    "AICORE_CLIENT_SECRET":                  "your-client-secret",
    "AICORE_BASE_URL":                       "https://api.ai.core.prod.us20.hanacloud.ondemand.com",
    "AICORE_RESOURCE_GROUP":                 "default",
    "AICORE_ORCHESTRATION_DEPLOYMENT_URL":   "https://<your-orch-deployment-url>"
}
```

Alternatively, export the same keys as environment variables.

---

## 🏗️ Architecture

```
User Input
    ↓
[Prompt Template]  ← define dynamic {{?placeholders}}
    ↓
[Data Masking]     ← anonymize PII before any processing
    ↓
[Grounding]        ← retrieve and inject relevant context (RAG)
    ↓
[LLM]              ← gpt-4o, claude-3.5-sonnet, gemini-1.5-pro, etc.
    ↓
[Content Filter]   ← Azure Content Safety check on output
    ↓
Final Response
```

All modules are configured in a single `OrchestrationConfig` object and executed by `OrchestrationService`.

---

## 📂 File Structure

```
sap_genai_sdk_full_tutorial.py   ← main tutorial file (14 sections)
README.md                        ← this file
```

---

## 📖 Tutorial Sections

| # | Section | Key Class / Function | What You Learn |
|---|---|---|---|
| 1 | Imports | All SDK imports | What every import does |
| 2 | Service Init | `create_orchestration_service()` | Entry point, auth handling |
| 3 | LLM Config | `create_llm()` | Model name, temperature, max_tokens |
| 4 | Prompt Template | `create_template()` | `{{?placeholder}}` syntax |
| 5 | Basic Run | `run_basic_completion()` | Simplest pipeline, response parsing |
| 6 | Streaming | `run_streaming()` | Chunk-by-chunk token output |
| 7 | SAP Help Grounding | `run_with_sap_help_grounding()` | RAG from SAP Help Portal |
| 8 | Vector Grounding | `run_with_vector_grounding()` | RAG from your own documents |
| 9 | Data Masking | `create_data_masking()` | PII anonymization / pseudonymization |
| 10 | Content Filtering | `create_content_filter()` | Azure Safety thresholds |
| 11 | Full Pipeline | `run_full_enterprise_pipeline()` | All modules combined |
| 12 | Chatbot | `SAPChatBot` class | Multi-turn conversation history |
| 13 | LangChain | `run_langchain_rag_pipeline()` | FAISS + LCEL + correct 2024 imports |
| 14 | Main Demo | `main()` | Running all features end-to-end |

---

## 🔑 Core Concepts

### OrchestrationService
The main client that communicates with your SAP AI Core deployment. Reads credentials automatically from config. Use `.run()` for synchronous calls, `.stream()` for real-time streaming.

### OrchestrationConfig
The "blueprint" of your pipeline. Bundles together:
- `template` — prompt structure with placeholders
- `llm` — model selection and parameters
- `grounding` — RAG configuration *(optional)*
- `data_masking` — PII protection *(optional)*
- `filtering` — content safety *(optional)*

### Template & Placeholders
```python
# Placeholder syntax: {{?variable_name}}
UserMessage("Answer this question: {{?user_query}}")

# Fill at runtime
TemplateValue(name="user_query", value="What is SAP BTP?")

# Grounding auto-fills this placeholder — you just include it
UserMessage("Context: {{ ?grounding_response }}\nQuestion: {{ ?user_query }}")
```

---

## 🔍 Grounding (RAG) — Two Modes

### Mode 1 — SAP Help Portal
Use when answering questions about official SAP products and documentation.

```python
DocumentGroundingFilter(
    id="sap_help",
    data_repository_type="help.sap.com"   # built-in SAP source
)
```

### Mode 2 — Custom Vector Store
Use when your own documents (HR policies, manuals, product specs) are indexed in SAP AI Core.

```python
DocumentGroundingFilter(
    id="my_vector_store",
    data_repositories=["your-repo-id"],
    data_repository_type=DataRepositoryType.VECTOR.value,
    search_config={"max_chunk_count": 5}
)
```

> The grounding module automatically searches the knowledge base using `input_params`, then injects retrieved text into `output_param` (i.e., `{{?grounding_response}}`).

---

## 🔒 Data Masking

Protects PII before it reaches the LLM or grounding service.

| Method | Behaviour | Reversible? |
|---|---|---|
| `ANONYMIZATION` | `"John Smith"` → `"[PERSON]"` | ❌ No |
| `PSEUDONYMIZATION` | `"John Smith"` → `"ENTITY_001"` | ✅ Yes |

Supported PII types: `EMAIL`, `PHONE`, `PERSON`, `ADDRESS`, `SAP_IDS_INTERNAL`

---

## 🛡️ Content Filtering

Uses Azure Content Safety. Applied to both user **input** and LLM **output**.

| Threshold | Meaning |
|---|---|
| `0` | Block all (strictest) — recommended for enterprise |
| `2` | Allow low severity |
| `4` | Allow medium severity |
| `6` | Allow most (least strict) |

Categories: `hate`, `self_harm`, `sexual`, `violence`

---

## 💬 Multi-Turn Chatbot

LLMs are stateless — you must pass the entire conversation history on every call.

```python
bot = SAPChatBot(service, llm)
bot.chat("What is SAP BTP?")           # Turn 1
bot.chat("How does it relate to AI Core?")  # Turn 2 — has Turn 1 context
bot.reset()                            # Clear history
```

Internally, `messages_history` is a growing list of `UserMessage` and assistant `Message` objects passed to `.run()` on each turn.

---

## 🔗 LangChain Integration

Use LangChain's SAP proxy wrappers as drop-in replacements for OpenAI classes.

```python
# ✅ Correct 2024+ imports
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS          # NOT langchain.vectorstores
from langchain_text_splitters import RecursiveCharacterTextSplitter  # NOT langchain.text_splitter

# ✅ Use LCEL chains (NOT deprecated RetrievalQA)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

**When to use LangChain vs Orchestration Service:**

| Use Case | Recommendation |
|---|---|
| SAP enterprise apps | ✅ Orchestration Service |
| Custom vector stores (Chroma, Weaviate) | ✅ LangChain |
| Built-in PII masking + content safety | ✅ Orchestration Service |
| Existing LangChain codebase | ✅ LangChain |

---

## 📊 Extracting Results

```python
result = service.run(config=config, template_values=[...])

# Get LLM response text
answer = result.orchestration_result.choices[0].message.content

# Inspect retrieved grounding chunks (for debugging / citations)
chunks = result.module_results.grounding.data['grounding_result']
```

---

## 🎯 Interview Quick-Reference Q&A

**Q: What is the new SAP Gen AI package name?**
`sap-ai-sdk-gen` — `generative-ai-hub-sdk` is deprecated since 2024.

**Q: What is `OrchestrationService`?**
Main client to run/stream AI pipelines on SAP AI Core. Handles OAuth auth automatically.

**Q: What is `OrchestrationConfig`?**
Blueprint that bundles `template + llm + grounding + masking + filtering`.

**Q: What is the template placeholder syntax?**
`{{?variable_name}}` — double curly brace + `?` + name.

**Q: What is grounding / RAG?**
Retrieve relevant document chunks → inject as context → LLM generates grounded answer.

**Q: Two grounding source types?**
`help.sap.com` (SAP official docs) and `DataRepositoryType.VECTOR` (your own documents).

**Q: Anonymization vs Pseudonymization?**
Anon = irreversible `[PERSON]`, Pseudo = reversible `ENTITY_001`.

**Q: Content filter strictness values?**
`0` = strictest (block all), `6` = most permissive.

**Q: How does multi-turn chat work?**
Pass the entire `messages_history` list to `.run()` on every turn.

**Q: What does `TemplateValue` do?**
Maps a placeholder name to its runtime value — like a URL query parameter.

**Q: How do you get grounded results for inspection?**
`result.module_results.grounding.data['grounding_result']`

**Q: What is LCEL?**
LangChain Expression Language — composes chain steps using the `|` pipe operator.

**Q: Correct LangChain imports for 2024+?**
`langchain_community.vectorstores` (not `langchain.vectorstores`) and `langchain_text_splitters` (not `langchain.text_splitter`). Use LCEL instead of deprecated `RetrievalQA`.

---

## 🔗 Useful Links

- [SAP AI Core Documentation](https://help.sap.com/docs/sap-ai-core)
- [SAP Generative AI Hub Overview](https://help.sap.com/docs/sap-ai-core/sap-ai-core-service-guide/generative-ai-hub)
- [SAP Gen AI Codejam Samples (GitHub)](https://github.com/SAP-samples/generative-ai-codejam)
- [PyPI — sap-ai-sdk-gen](https://pypi.org/project/sap-ai-sdk-gen/)

---

## 📝 License

For learning and interview preparation purposes only. SAP and all related trademarks are property of SAP SE.
