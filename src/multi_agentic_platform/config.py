from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MAP_", env_file=".env")

    provider: str = "mock"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    hf_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_tokens: int = 800
    temperature: float = 0.2

    rag_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rag_chunk_size: int = 600
    rag_chunk_overlap: int = 120

    # Comma-separated list of MCP server names, e.g. "filesystem,github"
    mcp_server_names: str = ""
    # Per-server env vars expected pattern:
    # MAP_MCP_<NAME>_COMMAND, MAP_MCP_<NAME>_ARGS (space-separated args)


settings = Settings()
