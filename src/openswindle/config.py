import json
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
    # JSON object merged into every completion request (provider extras such
    # as disabling thinking mode).
    llm_extra_body: str = ""
    # Models whose provider rejects response_format; JSON mode is skipped for
    # these instead of burning one failed call per process to find out.
    json_mode_unsupported_models: str = "vercel_ai_gateway/deepseek/deepseek-v4-flash"
    finished_match_ttl_seconds: int = 3600
    max_finished_matches: int = 1000

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def llm_extra_body_dict(self) -> dict:
        return json.loads(self.llm_extra_body) if self.llm_extra_body else {}

    @property
    def json_mode_unsupported_set(self) -> set[str]:
        return {
            m.strip() for m in self.json_mode_unsupported_models.split(",") if m.strip()
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
