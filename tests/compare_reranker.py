import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.rag_service import RagSummarizeService
from model.factory import chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

QUERIES = [
    "大户型适合什么扫地机器人",
    "扫地机器人滚刷缠绕头发怎么处理",
    "扫地机器人多久保养一次",
]

def compare_rerankers():
    rag = RagSummarizeService()
    W = 58

    print(f"+{'-'*W}+")
    print(f"| {'指标':<20} | {'BGE':<14} | {'LLM':<14} |")
    print(f"+{'-'*W}+")

    for query in QUERIES:
        docs = rag.ensemble_retriever.invoke(query)
        bge_docs, bge_scores = rag._bge_rerank(query, docs)
        llm_docs = rag._llm_rerank(query, docs)

        bge_overlap = len(set(d.page_content[:100] for d in bge_docs) &
                          set(d.page_content[:100] for d in llm_docs))
        jaccard = bge_overlap / len(set(
            list(d.page_content[:100] for d in bge_docs) +
            list(d.page_content[:100] for d in llm_docs)
        ))

        print(f"|>> {query:<55}|")
        print(f"| {'原始文档数':<20} | {len(docs):<14} | {len(docs):<14} |")
        print(f"| {'重排后保留':<20} | {len(bge_docs):<14} | {len(llm_docs):<14} |")
        merged = f"| {'Jaccard相似度':<20} | {jaccard:.2f}{'':>27} |"
        print(merged)
        if bge_scores:
            print(f"| {'BGE最高分':<20} | {max(bge_scores):.3f}{'':>26} |")
        bge_top = bge_docs[0].page_content[:29] + ".." if bge_docs and len(bge_docs[0].page_content) > 29 else (bge_docs[0].page_content[:31] if bge_docs else "N/A")
        llm_top = llm_docs[0].page_content[:29] + ".." if llm_docs and len(llm_docs[0].page_content) > 29 else (llm_docs[0].page_content[:31] if llm_docs else "N/A")
        print(f"| {'A组 top-1':<20} | {bge_top:<31} |")
        print(f"| {'B组 top-1':<20} | {llm_top:<31} |")
        print(f"+{'-'*W}+")

    llm_judge(query, bge_docs, llm_docs)

def llm_judge(query, bge_list, llm_list):
    prompt = PromptTemplate.from_template(
        "用户问题：{query}\n\n"
        "【A组】\n{bge_docs}\n\n"
        "【B组】\n{llm_docs}\n\n"
        "哪组文档与问题更相关？只输出A或B并简要说明。"
    )
    bge_text = "\n".join(f"[{i+1}] {d.page_content[:150]}" for i, d in enumerate(bge_list))
    llm_text = "\n".join(f"[{i+1}] {d.page_content[:150]}" for i, d in enumerate(llm_list))
    result = (prompt | chat_model | StrOutputParser()).invoke({
        "query": query, "bge_docs": bge_text, "llm_docs": llm_text
    })
    print(f"LLM裁判结论（最后一题）:\n查询: {query}\n结论:\n{result}")

if __name__ == "__main__":
    compare_rerankers()
