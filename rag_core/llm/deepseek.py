"""OpenAI-compatible LLM gateway client.

DeepSeek is used only as the development placeholder. The competition gateway
can be selected with LLM_BASE_URL, LLM_API_KEY and LLM_MODEL.
"""
from langchain_openai import ChatOpenAI

from config import get_config


def create_llm(streaming: bool = True) -> ChatOpenAI:
    """Create the configured OpenAI-compatible gateway client."""
    config = get_config()
    if not config.llm_api_key:
        raise ValueError(
            "LLM_API_KEY 未设置。请配置比赛网关密钥；开发阶段也可使用 "
            "DEEPSEEK_API_KEY。"
        )
    return ChatOpenAI(
        model=config.llm_model,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        streaming=streaming,
        temperature=0.3,
    )


def create_deepseek_llm(streaming: bool = True) -> ChatOpenAI:
    """Backward-compatible alias."""
    return create_llm(streaming=streaming)
