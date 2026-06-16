# 智扫通 AI Agent

基于 **ReAct Agent + RAG + MCP + FastAPI + Redis + SQL ** 的智能客服助手，专为扫地机器人/扫拖一体机器人场景打造。

## 技术栈

| 模块 | 技术 |
|------|------|
| 前端界面 | Streamlit |
| API 服务 | FastAPI + Uvicorn |
| AI Agent | LangChain ReAct Agent |
| 工具协议 | MCP (Model Context Protocol) |
| 向量库 | ChromaDB + BGE Reranker |
| RAG 检索 | Ensemble Retrieval (向量+BM25) + Query Rewrite + Reranker |
| 多模态 | Qwen-VL-Max 图片分析 |
| 大模型 | Qwen3-Max (DashScope) |
| PDF 报告 | fpdf2 |

## 快速开始

### 1. 克隆并配置

```bash
git clone <your-repo-url>
cd Agent智扫通项目
cp .env.example .env
```

编辑 `.env`，填入你的 API Key：

```
DASHSCOPE_API_KEY=sk-xxxxx
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> 如果 BGE Reranker 下载慢，可设置 HuggingFace 镜像：`export HF_ENDPOINT=https://hf-mirror.com`

### 3. 运行

**Streamlit 前端（推荐开发使用）：**

```bash
streamlit run app.py
```

**FastAPI 后端（推荐生产部署）：**

```bash
uvicorn fastapi_app.main:app --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000/docs` 查看 API 文档。

### 4. Docker 部署

```bash
docker compose up -d
```

## 项目结构

```
├── app.py                    # Streamlit 前端入口
├── agent/
│   ├── react_agent.py        # ReAct Agent + MCP 上下文管理
│   └── tools/
│       ├── mcp_server.py     # MCP 工具服务端 (10+ 工具)
│       └── middleware.py     # Agent 中间件 (日志/动态提示词)
├── fastapi_app/
│   ├── main.py               # FastAPI API 端点
│   ├── deps.py               # Agent 生命周期管理
│   └── schemas.py            # Pydantic 数据模型
├── rag/
│   ├── rag_service.py        # 企业级 RAG 服务 (重排序+上下文压缩)
│   └── vector_store.py       # ChromaDB 向量存储
├── config/                   # YAML 配置文件
├── prompts/                  # 提示词模板
├── data/                     # 知识库原始数据
├── model/
│   └── factory.py            # 模型工厂 (ChatTongyi / Embedding)
├── utils/                    # 工具模块
├── tests/                    # 单元测试
└── docker-compose.yml        # Docker 编排
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 (含 MCP/RAG 状态) |
| POST | `/chat/stream` | SSE 流式对话 |
| POST | `/chat` | 同步对话 |
| POST | `/history/clear` | 清空对话历史 |
| POST | `/upload` | 上传图片 |
| GET | `/report/{filename}` | 下载 PDF 报告 |

## 可用工具

| 工具 | 说明 |
|------|------|
| `rag_summarize` | 本地知识库 RAG 检索 |
| `sougou_search` | 联网实时搜索 |
| `get_current_weather` | 天气查询 |
| `product_recommend` | 产品推荐 (户型/预算/地面/宠物) |
| `price_query` | 产品价格查询 |
| `image_analyze` | 多模态图片分析 |
| `export_report` | PDF 报告导出 |
| `get_user_id` / `get_current_month` | 用户身份/月份获取 |
| `fetch_external_data` | 用户使用记录查询 |
| `fill_context_for_report` | 报告生成上下文注入 |

## 测试

```bash
pytest tests/ -v
```
