from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENSWINDLE_", env_file=".env", extra="ignore"
    )

    llm_model: str = "vercel_ai_gateway/deepseek/deepseek-v4-flash"
    mock_llm: bool = False
    cors_origins: str = "http://localhost:5173"
    llm_max_reprompts: int = 2
    finished_match_ttl_seconds: int = 3600
    max_finished_matches: int = 1000

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
