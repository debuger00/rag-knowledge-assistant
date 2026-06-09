"""DeepSeek LLM 封装 — 通过 LangChain ChatOpenAI 调用。"""
from langchain_openai import ChatOpenAI

from config import get_config


def create_deepseek_llm(streaming: bool = True) -> ChatOpenAI:
    """创建 DeepSeek ChatOpenAI 实例。

    DeepSeek API 兼容 OpenAI SDK 格式。
    """
    config = get_config()
    if not config.deepseek_api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY 未设置。请在 .env 文件或环境变量中配置。"
        )
    return ChatOpenAI(
        model=config.deepseek_model,
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_base_url,
        streaming=streaming,
        temperature=0.3,
    )
