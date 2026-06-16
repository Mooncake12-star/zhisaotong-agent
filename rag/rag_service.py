"""
企业级 RAG 服务 v2 — 优化版
改进点：
  1. BGE Reranker 替代 LLM 重排序（更稳定、更快、更便宜）
  2. 上下文压缩（阈值过滤 + 长度裁切，降低 token 消耗）

安装额外依赖（三选一）：
  pip install sentence-transformers    # BGE Reranker（推荐，最成熟）
  pip install flashrank                # 轻量级 ONNX reranker（无需 PyTorch）
  pip install flagembedding            # FlagEmbedding 官方库

依赖已安装，会自动使用 BGE Reranker；
未安装则降级为 LLM 重排序（原版逻辑），不报错。
"""
#sys寻包逻辑
'''
# 遍历 sys.path 里的每个目录
for dir in sys.path:
    # 拼路径：dir + "/utils/config_handler.py"
    candidate = os.path.join(dir, "utils/config_handler.py")
    # 检查这个路径存不存在
    if os.path.exists(candidate):
        导入它
'''
import sys
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import List, AsyncGenerator
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import logging

from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from model.factory import chat_model
from utils.logger_handler import logger

QUERY_REWRITE_TEMPLATE = """你是检索专家，请根据用户的原始问题和对话历史，生成一个更清晰、更完整的搜索查询。

原始问题：{input}

要求：
- 补充缺失的上下文，使查询独立完整
- 提取核心关键词
- 直接输出改写后的查询，不要多余解释"""

# ──────────────────────────────────────────────
# 1. 专业 Reranker — 惰性加载，首次调用时才下载模型
# ──────────────────────────────────────────────
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

_RERANKER_INSTANCE = None
_HAS_BGE_RERANKER = False

try:
    from sentence_transformers import CrossEncoder
    _HAS_BGE_RERANKER = True
except ImportError:
    logger.warning("[Reranker] sentence-transformers 未安装，降级到 LLM 重排序")


def _get_reranker():
    global _RERANKER_INSTANCE
    global _HAS_BGE_RERANKER
    if _RERANKER_INSTANCE is not None:
        return _RERANKER_INSTANCE
    if not _HAS_BGE_RERANKER:
        return None

    try:
        _RERANKER_INSTANCE = CrossEncoder(RERANKER_MODEL_NAME)
        logger.info(f"[Reranker] 已加载模型: {RERANKER_MODEL_NAME}")
        return _RERANKER_INSTANCE
    except Exception as e:
        logger.warning(f"[Reranker] 模型加载失败，降级到 LLM 重排序: {e}")
        _HAS_BGE_RERANKER = False
        return None


class RagSummarizeService:
    def __init__(self, vector_top_k: int = 10, keyword_top_k: int = 10, final_top_k: int = 6):
        self.vector_store_svc = VectorStoreService()
        self.model = chat_model
        self.vector_top_k = vector_top_k
        self.keyword_top_k = keyword_top_k
        self.final_top_k = final_top_k
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)

        self._init_ensemble_retriever()
        self.chain = self._build_chain()


    #给文档打分，然后从高到低按top_k那文档
    def _init_ensemble_retriever(self):
        """初始化集成检索器：向量检索 + BM25 关键词检索"""
        vector_retriever = self.vector_store_svc.get_retriever()

        raw_chroma = self.vector_store_svc.vector_store
        all_data = raw_chroma.get(include=["documents", "metadatas"])
        texts = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])

        if not texts:
            logger.warning("[检索器] 向量库中无文档，回退到纯向量检索")
            self.ensemble_retriever = vector_retriever
            return

        docs = [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(texts, metadatas or [{}] * len(texts))
        ]
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = self.keyword_top_k
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
        logger.info(f"[检索器] EnsembleRetriever 构建完成（向量+BM25），共 {len(docs)} 个文档")

    # ──────────────────────────────────────────────
    # 2. BGE Reranker — 比 LLM 打分更稳、更快
    # ─────────────────────────────────────────────—
    def _bge_rerank(self, query: str, docs: List[Document]):
        if not docs:
            return docs, []

        reranker_model = _get_reranker()
        if reranker_model is None:
            return self._llm_rerank(query, docs), []
        pairs = [[query, d.page_content[:512]] for d in docs]
        scores = reranker_model.predict(pairs)
        scores = [float(s) for s in scores]

        min_s, max_s = min(scores), max(scores)
        if max_s - min_s > 1e-8:
            scores = [(s - min_s) / (max_s - min_s) for s in scores]
        else:
            scores = [0.5] * len(scores)

        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

        logger.info(f"[BGE Reranker] 完成 {len(docs)} 个文档评分")
        all_scores = [s for s, _ in scored]
        return [d for _, d in scored][:self.final_top_k], all_scores[:self.final_top_k]

    # ──────────────────────────────────────────────
    # 3. LLM 重排序（降级方案）
    # ──────────────────────────────────────────────
    def _llm_rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return docs
        try:
            rerank_prompt = PromptTemplate.from_template(
                "用户问题：{query}\n\n"
                "参考文档：\n{docs_text}\n\n"
                "请从高到低给这些文档与问题的相关性打分（0-10分），"
                "输出格式：每行一个「分数 文档序号」，只输出分数和序号。"
            )
            docs_text = "\n".join(
                f"[{i}] {d.page_content[:200]}"
                for i, d in enumerate(docs)
            )
            result = (rerank_prompt | self.model | StrOutputParser()).invoke(
                {"query": query, "docs_text": docs_text}
            )
            scored = []
            for line in result.strip().split("\n"):
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        score = float(parts[0])
                        idx = int(parts[-1])
                        if 0 <= idx < len(docs):
                            scored.append((score, docs[idx]))
                    except (ValueError, IndexError):
                        pass
            if scored:
                scored.sort(key=lambda x: x[0], reverse=True)
                logger.info(f"[LLM 重排序] {len(scored)} 个文档完成评分")
                return [d for _, d in scored][:self.final_top_k]
            return docs[:self.final_top_k]
        except Exception as e:
            logger.warning(f"[LLM 重排序] 失败，保持原序: {e}")
            return docs[:self.final_top_k]

    # ──────────────────────────────────────────────
    # 4. 上下文压缩 — 按分数阈值过滤 + 截断过长内容
    # ──────────────────────────────────────────────
    SCORE_THRESHOLD = 0.3
    MAX_TOKENS_PER_DOC = 800

    def _compress_context(self, docs: List[Document], scores: List[float] = None) -> List[Document]:
        compressed = []
        for i, doc in enumerate(docs):
            if scores and scores[i] < self.SCORE_THRESHOLD:
                continue
            content = doc.page_content[:self.MAX_TOKENS_PER_DOC]
            compressed.append(Document(page_content=content, metadata=doc.metadata))
        if not compressed:
            return docs[:1]
        return compressed

    # ──────────────────────────────────────────────
    # 5. 执行重排序（自动选择方案）
    # ──────────────────────────────────────────────
    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if _get_reranker() is not None:
            reranked, scores = self._bge_rerank(query, docs)
            compressed = self._compress_context(reranked, scores)
            return compressed
        else:
            llm_reranked = self._llm_rerank(query, docs)
            return self._compress_context(llm_reranked)

    # ──────────────────────────────────────────────
    # 6. 构建最终上下文
    # ──────────────────────────────────────────────
    def _build_context(self, docs: List[Document]) -> str:
        context = ""
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", doc.metadata.get("file_path", "未知来源"))
            context += (
                f"【参考资料{i + 1}】\n"
                f"内容：{doc.page_content}\n"
                f"来源：{source}\n"
                f"相关度排名：{i + 1}\n\n"
            )
        return context

    # ──────────────────────────────────────────────
    # 7. 构建管道
    # ──────────────────────────────────────────────
    def _build_chain(self):
        def safe_rewrite(inputs):
            try:
                chain = (
                    PromptTemplate.from_template(QUERY_REWRITE_TEMPLATE)
                    | self.model
                    | StrOutputParser()
                )
                rewritten = chain.invoke({"input": inputs["input"]})
                rewritten = rewritten.strip().strip('"').strip("'")
                logger.info(f"[查询重写] 原始: {inputs['input']} → 改写: {rewritten}")
                return rewritten
            except Exception as e:
                logger.warning(f"[查询重写] 失败，使用原始查询: {e}")
                return inputs["input"]

        def retrieve_and_process(inputs):
            query = inputs["input"]
            rewritten = inputs["rewritten_query"]
            docs = self.ensemble_retriever.invoke(rewritten)
            logger.info(f"[集成检索] 向量+BM25 共 {len(docs)} 条")

            reranked = self._rerank(query, docs)
            context = self._build_context(reranked)
            return {"input": query, "context": context}

        generate_chain = self.prompt_template | self.model | StrOutputParser()

        return (
            RunnablePassthrough.assign(rewritten_query=RunnableLambda(safe_rewrite))
            | RunnableLambda(retrieve_and_process)
            | generate_chain
        )

    def rag_summarize(self, query: str) -> str:
        return self.chain.invoke({"input": query})

    async def astream_rag_summarize(self, query: str) -> AsyncGenerator[str, None]:
        async for chunk in self.chain.astream({"input": query}):
            yield chunk


if __name__ == '__main__':
    from rag.vector_store import VectorStoreService
    vs = VectorStoreService()
    vs.load_document()

    rag = RagSummarizeService()
    print("=== 企业级 RAG 测试 ===")
    result = rag.rag_summarize("大户型适合哪些扫地机器人")
    print(result)
