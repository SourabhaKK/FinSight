from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM provider
    llm_provider: str = "ollama"

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    # Model artefact paths
    distilbert_model_path: str = "artefacts/distilbert_finsight.pt"
    baseline_model_path: str = "artefacts/baseline_pipeline.joblib"
    urgency_model_path: str = "artefacts/urgency_pipeline.joblib"

    # API settings
    log_level: str = "INFO"
    max_text_length: int = 10000
    min_text_length: int = 10


settings = Settings()
