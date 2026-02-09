from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    cohere_api_key: str | None = os.getenv("COHERE_API_KEY")
    pinecone_api_key: str | None = os.getenv("PINECONE_API_KEY")


def load_config() -> AppConfig:
    return AppConfig()
