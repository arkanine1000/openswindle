import json
from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENSWINDLE_", env_file=".env", extra="ignore"
    )

    llm_model: str = "deepseek/deepseek-v4-flash"
    mock_llm: bool = False
    cors_origins: str = "http://localhost:5174"
    llm_max_reprompts: int = 2
    # JSON object merged into every completion request (provider extras, e.g.
    # OpenRouter's unified reasoning control: {"reasoning": {"effort": "none"}}).
    llm_extra_body: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "openrouter_api_key"),
    )
    finished_match_ttl_seconds: int = 3600
    max_finished_matches: int = 1000

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def llm_extra_body_dict(self) -> dict:
        return json.loads(self.llm_extra_body) if self.llm_extra_body else {}


@lru_cache
def get_settings() -> Settings:
    return Settings()
