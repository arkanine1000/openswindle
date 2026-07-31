import json
from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENSWINDLE_", env_file=".env", extra="ignore"
    )

    # Server default when a match's MatchConfig.llm_model is unset (client
    # didn't choose one). Not itself restricted to models.LLMModel, since an
    # operator may run a model that isn't offered to clients.
    llm_model: str = "deepseek/deepseek-v4-flash"
    mock_llm: bool = False
    cors_origins: str = "http://localhost:5174"
    llm_max_reprompts: int = 2
    # Per-request HTTP timeout for the OpenRouter client — the SDK default
    # (600s) is a batch-workload number, not a live-turn one. See
    # CHANGELOG.md for the incident and the number this is tuned against.
    llm_timeout_seconds: float = 45.0
    # Hard per-match ceiling on NPC decisions, independent of which model is
    # in play. A round's bid space is a strict total order over (quantity,
    # face) pairs, so it can have at most total_dice*4 moves before a call is
    # forced; summing that across every round from a match's max possible
    # total dice (2 * MatchConfig.dice_per_player, capped at 6) down to 2
    # gives an exact worst-case bound of ~308 moves. 320 leaves headroom.
    # Once hit, remaining decisions fall back to the scripted policy.
    llm_max_calls_per_match: int = 320
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
