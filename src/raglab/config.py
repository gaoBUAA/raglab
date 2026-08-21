"""集中式配置：所有可调参数都通过环境变量（RAGLAB_ 前缀）或 .env 注入。"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置。示例见 .env.example。"""

    model_config = SettingsConfigDict(env_prefix="RAGLAB_", env_file=".env", extra="ignore")

    # ---- LLM ----
    llm_provider: str = "openai_compat"  # openai_compat | mock
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    # ---- Embedding ----
    embedding_provider: str = "openai_compat"  # openai_compat | mock
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_api_key: str = ""  # 为空时复用 llm_api_key

    # ---- Retrieval ----
    top_k: int = 5
    rrf_k: int = 60  # Reciprocal Rank Fusion 常数
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # ---- Agent ----
    agent_max_iterations: int = 5

    # ---- Storage ----
    data_dir: Path = Path("data")

    @property
    def effective_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.llm_api_key
