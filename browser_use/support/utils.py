import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from browser_use.support.constants import *
from browser_use.support.llm import DeepSeekR1ChatOpenAI, DeepSeekR1ChatOllama


import logging
import sys
from functools import wraps
from typing import Any, Callable, Coroutine, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

# Define generic type variables for return type and parameters
R = TypeVar('R')
P = ParamSpec('P')


def get_screen_resolution():
    if sys.platform == "darwin":  # macOS
        try:
            from AppKit import NSScreen
            screen = NSScreen.mainScreen().frame()
            return {"width": int(screen.size.width), "height": int(screen.size.height)}
        except ImportError:
            print("AppKit is not available. Make sure you are running this on macOS.")
        except Exception as e:
            print(f"Error retrieving macOS screen resolution: {e}")
        return {"width": 2560, "height": 1664}

    else:  # Windows & Linux
        try:
            from screeninfo import get_monitors
            monitors = get_monitors()
            if not monitors:
                raise Exception("No monitors detected.")
            monitor = monitors[0]
            return {"width": monitor.width, "height": monitor.height}
        except ImportError:
            print("screeninfo package not found. Install it using 'pip install screeninfo'.")
        except Exception as e:
            print(f"Error retrieving screen resolution: {e}")

        return {"width": 1920, "height": 1080}


def get_window_adjustments():
    """Returns recommended x, y offsets for window positioning"""
    if sys.platform == "darwin":  # macOS
        return -4, 24  # macOS has a small title bar, no border
    elif sys.platform == "win32":  # Windows
        return -8, 0  # Windows has a border on the left
    else:  # Linux
        return 0, 0


def time_execution_sync(additional_text: str = '') -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f'{additional_text} Execution time: {execution_time:.2f} seconds')
            return result

        return wrapper

    return decorator


def time_execution_async(
        additional_text: str = '',
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
    def decorator(func: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, Coroutine[Any, Any, R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f'{additional_text} Execution time: {execution_time:.2f} seconds')
            return result

        return wrapper

    return decorator


def singleton(cls):
    instance = [None]

    def wrapper(*args, **kwargs):
        if instance[0] is None:
            instance[0] = cls(*args, **kwargs)
        return instance[0]

    return wrapper

def handle_api_key_error(provider: str, env_var: str):
    """
    Handles the missing API key error by raising a gr.Error with a clear message.
    """
    provider_display = PROVIDER_DISPLAY_NAMES.get(provider, provider.upper())
    print(
        f"💥 {provider_display} API key not found! 🔑 Please set the "
        f"`{env_var}` environment variable or provide it in the UI."
    )


def get_latest_files(directory: str, file_types: list = ['.webm', '.zip', '.json']) -> Dict[str, Optional[str]]:
    """Get the latest recording and trace files"""
    latest_files: Dict[str, Optional[str]] = {ext: None for ext in file_types}

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return latest_files

    for file_type in file_types:
        try:
            matches = list(Path(directory).rglob(f"*{file_type}"))
            if matches:
                latest = max(matches, key=lambda p: p.stat().st_mtime)
                if time.time() - latest.stat().st_mtime > 1.0:
                    latest_files[file_type] = str(latest)
        except Exception as e:
            print(f"Error getting latest {file_type} file: {e}")

    return latest_files


def get_latest_file(directory: str) -> str:
    """Get the latest recording and trace files"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return ""
    latest_file = ""
    try:
        matches = list(Path(directory).rglob(f"*"))
        if matches:
            latest = max(matches, key=lambda p: p.stat().st_mtime)
            if time.time() - latest.stat().st_mtime > 1.0:
                latest_file = str(latest)
    except Exception as e:
        print(f"Error getting latest file: {e}")

    return latest_file


def get_llm_model(provider: str, **kwargs):
    """
    获取LLM 模型
    :param provider: 模型类型
    :param kwargs:
    :return:
    """
    if provider not in ["ollama"]:
        env_var = f"{provider.upper()}_API_KEY"
        api_key = kwargs.get("api_key", "") or os.getenv(env_var, "")
        if not api_key:
            handle_api_key_error(provider, env_var)
        kwargs["api_key"] = api_key

    if provider == "anthropic":
        if not kwargs.get("base_url", ""):
            base_url = "https://api.anthropic.com"
        else:
            base_url = kwargs.get("base_url")

        return ChatAnthropic(
            model_name=kwargs.get("model_name", "claude-3-5-sonnet-20241022"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == 'mistral':
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("MISTRAL_ENDPOINT", "https://api.mistral.ai/v1")
        else:
            base_url = kwargs.get("base_url")
        if not kwargs.get("api_key", ""):
            api_key = os.getenv("MISTRAL_API_KEY", "")
        else:
            api_key = kwargs.get("api_key")

        return ChatMistralAI(
            model=kwargs.get("model_name", "mistral-large-latest"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == "openai":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")
        else:
            base_url = kwargs.get("base_url")

        return ChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == "deepseek":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("DEEPSEEK_ENDPOINT", "")
        else:
            base_url = kwargs.get("base_url")

        if kwargs.get("model_name", "deepseek-chat") == "deepseek-reasoner":
            return DeepSeekR1ChatOpenAI(
                model=kwargs.get("model_name", "deepseek-reasoner"),
                temperature=kwargs.get("temperature", 0.0),
                base_url=base_url,
                api_key=api_key,
            )
        else:
            return ChatOpenAI(
                model=kwargs.get("model_name", "deepseek-chat"),
                temperature=kwargs.get("temperature", 0.0),
                base_url=base_url,
                api_key=api_key,
            )
    elif provider == "google":
        return ChatGoogleGenerativeAI(
            model=kwargs.get("model_name", "gemini-2.0-flash-exp"),
            temperature=kwargs.get("temperature", 0.0),
            google_api_key=api_key,
        )
    elif provider == "ollama":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
        else:
            base_url = kwargs.get("base_url")

        if "deepseek-r1" in kwargs.get("model_name", "qwen2.5:7b"):
            return DeepSeekR1ChatOllama(
                model=kwargs.get("model_name", "deepseek-r1:14b"),
                temperature=kwargs.get("temperature", 0.0),
                num_ctx=kwargs.get("num_ctx", 32000),
                base_url=base_url,
            )
        else:
            return ChatOllama(
                model=kwargs.get("model_name", "qwen2.5:7b"),
                temperature=kwargs.get("temperature", 0.0),
                num_ctx=kwargs.get("num_ctx", 32000),
                num_predict=kwargs.get("num_predict", 1024),
                base_url=base_url,
            )
    elif provider == "azure_openai":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        else:
            base_url = kwargs.get("base_url")
        api_version = kwargs.get("api_version", "") or os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        return AzureChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.0),
            api_version=api_version,
            azure_endpoint=base_url,
            api_key=api_key,
        )
    elif provider == "alibaba":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("ALIBABA_ENDPOINT", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        else:
            base_url = kwargs.get("base_url")

        return ChatOpenAI(
            model=kwargs.get("model_name", "qwen-plus"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )

    elif provider == "moonshot":
        return ChatOpenAI(
            model=kwargs.get("model_name", "moonshot-v1-32k-vision-preview"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=os.getenv("MOONSHOT_ENDPOINT"),
            api_key=os.getenv("MOONSHOT_API_KEY"),
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def filter_history_result_true(history_item):
    return history_item["result"][-1]["is_done"] is not True


def remove_completed_from_history(history):
    return list(filter(filter_history_result_true, history))


def filter_history_result_error_free(history_item):
    return history_item["result"][-1].get("error") is None


def remove_error_from_history(history):
    return list(filter(filter_history_result_error_free, history))


def fix_history_step_numbers(history):
    for idx, history_item in enumerate(history):
        history_item["metadata"]["step_number"] = idx + 1


def move_xpath_from_interacted_element_to_action(history):
    for history_item in history:
        for idx, action in enumerate(history_item["model_output"]["action"]):
            interacted_element = history_item["state"]["interacted_element"][idx]
            action_name = None
            for action_name in action:
                pass
            if action[action_name].get("index") is not None and action[action_name].get("xpath") is None:
                action[action_name]["xpath"] = interacted_element["xpath"]


def fix_history_save_to_file(history, file_path):
    remove_error_from_history(history)
    fix_history_step_numbers(history)
    move_xpath_from_interacted_element_to_action(history)
    final = {"history": history}
    json.dump(final, open(file_path, 'w'))
