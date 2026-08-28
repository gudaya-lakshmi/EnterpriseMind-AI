\# 🧠 EnterpriseMind AI



\### An Agentic RAG-Based Assistant for Enterprise Knowledge Retrieval and Verification



\*\*Developed by Ganipisetty Udaya Lakshmi\*\*  

B.Tech CSE – Artificial Intelligence \& Machine Learning



\---



\## 📌 About the Project



\*\*EnterpriseMind AI\*\* is an Agentic RAG system that allows users to ask questions and generate summaries from enterprise documents.



Unlike a basic RAG system that directly retrieves documents and sends them to an LLM, EnterpriseMind AI uses specialized agents to understand the user's request, generate responses, verify them against retrieved evidence, and provide supporting source citations.



The main goal is to make enterprise document-based AI responses more \*\*grounded, reliable, and verifiable\*\*.



\---



\## ⚙️ How It Works



```text

Enterprise PDF

&#x20;     ↓

Document Loading \& Chunking

&#x20;     ↓

Embeddings

&#x20;     ↓

ChromaDB Vector Database

&#x20;     ↓

User Question

&#x20;     ↓

Router Agent

&#x20;     ↓

Semantic Retrieval

&#x20;     ↓

Cross-Encoder Reranking

&#x20;     ↓

┌─────────────────┬─────────────────────┐

│     QA Path     │  Summarization Path │

│                 │                     │

│  LLM Generator  │   Scope Filtering   │

│                 │          ↓          │

│                 │ Evidence Extraction │

│                 │          ↓          │

│                 │  Summarizer Agent   │

└────────┬────────┴──────────┬──────────┘

&#x20;        └─────────┬─────────┘

&#x20;                  ↓

&#x20;            Verifier Agent

&#x20;                  ↓

&#x20;            PASS / REVISE

&#x20;                  ↓

&#x20;            Citation Agent

&#x20;                  ↓

&#x20;         Grounded Final Answer

```



\---



\## 🤖 Agents



\### 🚦 Router Agent

Understands the user's intent and decides whether the request requires a \*\*direct answer\*\* or a \*\*document summary\*\*.



\### 📝 Summarizer Agent

Creates focused summaries from relevant document evidence. It uses scope filtering and evidence extraction to avoid mixing unrelated information.



\### ✅ Verifier Agent

Checks the generated response against retrieved evidence. If the response is unsupported or inconsistent, it can request a revision before the answer is accepted.



\### 📚 Citation Agent

Finds the document pages that support the verified answer and attaches them as sources.



\---



\## 🔍 Retrieval Pipeline



EnterpriseMind AI uses a two-stage retrieval process:



```text

User Query

&#x20;   ↓

ChromaDB Semantic Search

&#x20;   ↓

Top 20 Candidate Chunks

&#x20;   ↓

Cross-Encoder Reranking

&#x20;   ↓

Top 8 Relevant Chunks

```



This combines fast semantic retrieval with more precise reranking before information is passed to the LLM.



\---



\## ✨ Key Features



\- 📄 Enterprise PDF knowledge retrieval

\- 🔎 Semantic search using embeddings

\- 🗄️ ChromaDB vector storage

\- 🎯 Cross-Encoder reranking

\- 🚦 Intent-based agent routing

\- 💬 Document question answering

\- 📝 Focused document summarization

\- 🧹 Topic-specific evidence extraction

\- ✅ Answer verification

\- 🔁 Self-correction when verification fails

\- 📚 Source and page citations

\- 📊 RAG evaluation using DeepEval



\---



\## 🧪 Example



\*\*User Query\*\*



> Give me the major developments in Microsoft's Intelligent Cloud business during fiscal year 2025.



\*\*System Execution\*\*



```text

Router                 → summarize

Retrieved              → 20 chunks

Reranked               → 8 chunks

Scope Filter           → 3 relevant chunks

Evidence Extraction    → 3 relevant facts

Verifier               → PASS

Citation Support       → FOUND

```



\*\*Final Response\*\*



> Microsoft's Intelligent Cloud business saw significant growth in fiscal year 2025.

>

> • Microsoft Cloud revenue increased 23% to $168.9 billion.  

> • Server products and cloud services revenue increased 23%, driven by 34% growth in Azure and other cloud services revenue.



The response is then linked to supporting pages from the source document.



\---



\## 📊 Evaluation



The RAG pipeline was evaluated using \*\*DeepEval\*\*.



| Metric | Development Score |

|---|---:|

| Faithfulness | 0.833 |

| Answer Relevancy | 1.000 |

| Contextual Precision | 0.925 |

| Contextual Recall | 0.500 |



A separate retrieval benchmark achieved approximately \*\*90% successful retrieval\*\* on the development test questions.



\---



\## 🛠️ Tech Stack



\*\*Python • LangChain • LangGraph • Ollama • Llama 3.2 • ChromaDB • Nomic Embeddings • CrossEncoder • DeepEval\*\*



\---



\## 🚧 Current Development



\### Completed ✅



\*\*RAG Pipeline → Reranking → Router → Summarizer → Evidence Extraction → Verifier → Self-Correction → Citation\*\*



\### Next 🔨



\*\*Security Agent → Supervisor Agent → Authentication \& RBAC → API/UI → Conversation Memory\*\*



\---



\## 🎯 Project Goal



EnterpriseMind AI aims to evolve from a traditional RAG application into a \*\*secure multi-agent enterprise knowledge assistant\*\* capable of retrieving, reasoning over, verifying, and citing information from organizational documents.

