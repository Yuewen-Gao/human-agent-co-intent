"""
LLM Client Manager: Supports multiple LLM providers (OpenAI, Azure OpenAI, Claude, etc.)

This module provides a unified interface for different LLM providers.
"""

import os
import threading
from typing import Optional, Any, Dict
from abc import ABC, abstractmethod

# Total wall-clock limit for one OpenAI-compatible call (SDK retries included),
# and the SDK's retry count. The SDK's own read timeout only bounds the gap
# between received bytes, so a slow trickle could otherwise hold a turn for an hour.
DEFAULT_OPENAI_TIMEOUT_SECONDS = 60.0
DEFAULT_OPENAI_MAX_RETRIES = 1
# Token usage of the latest OpenAI-compatible call made by this thread (read by experiment scripts for cost).
_LAST_USAGE = threading.local()


def last_usage() -> Optional[Dict[str, Any]]:
    """Token usage of this thread's latest OpenAI-compatible call, or None."""
    return getattr(_LAST_USAGE, "value", None)


def _usage_dict(response: Any) -> Optional[Dict[str, Any]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    details = getattr(usage, "completion_tokens_details", None)
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details is not None else None,
        "cached_tokens": cached,
    }


def _openai_compatible_base_url(base_url: str) -> str:
    """Normalize a provider host to the OpenAI-compatible `/v1` API root."""
    normalized = base_url.strip().rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def _env_number(name: str, default: float, cast=float) -> Any:
    try:
        value = cast(os.getenv(name, default))
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _call_with_deadline(fn, seconds: float):
    """Run ``fn`` in a daemon thread and give up after ``seconds``.

    The abandoned call finishes in the background; its result is discarded.
    """
    outcome: Dict[str, Any] = {}

    def run():
        try:
            outcome["value"] = fn()
        except BaseException as error:  # re-raised in the caller's thread
            outcome["error"] = error

    worker = threading.Thread(target=run, daemon=True, name="llm-call")
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimeoutError(f"LLM call exceeded {seconds:g} s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _optional_positive_int(value: Any) -> Optional[int]:
    """Return a configured positive integer, or ``None`` for a safe fallback."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


class LLMClient(ABC):
    """Abstract base class for LLM clients"""

    def supports_multimodal_images(self) -> bool:
        """True if chat.completions accepts OpenAI-style image_url parts in message content."""
        return False
    
    @abstractmethod
    def chat_completions_create(
        self,
        messages: list,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Create a chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (optional, may use default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Response content as string
        """
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client"""

    def supports_multimodal_images(self) -> bool:
        return True
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = _env_number("OPENAI_TIMEOUT_SECONDS", DEFAULT_OPENAI_TIMEOUT_SECONDS)
        self.max_retries = _env_number("OPENAI_MAX_RETRIES", DEFAULT_OPENAI_MAX_RETRIES, int)
        try:
            from openai import OpenAI, Timeout
            client_kwargs = {
                "api_key": api_key,
                # Per-attempt limits; the total deadline in chat_completions_create covers retries.
                "timeout": Timeout(max(self.timeout_seconds - 10.0, 1.0), connect=5.0),
                "max_retries": self.max_retries,
            }
            if base_url:
                client_kwargs["base_url"] = _openai_compatible_base_url(base_url)
            self.client = OpenAI(**client_kwargs)
            self.default_model = default_model or "gpt-4o-mini"
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
    
    def chat_completions_create(
        self,
        messages: list,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        model = model or self.default_model
        if getattr(self, "reasoning_effort", None) and "reasoning_effort" not in kwargs:
            kwargs["reasoning_effort"] = self.reasoning_effort
        request_kwargs = {
            "model": model,
            "messages": messages,
            **kwargs,
        }
        if not str(model).startswith("gpt-5"):
            request_kwargs["temperature"] = temperature
        if max_tokens is not None:
            # GPT-5.6's Chat Completions endpoint rejects the legacy
            # `max_tokens` field in favour of `max_completion_tokens`.
            token_field = "max_completion_tokens" if str(model).startswith("gpt-5") else "max_tokens"
            request_kwargs[token_field] = max_tokens

        response = _call_with_deadline(
            lambda: self.client.chat.completions.create(**request_kwargs),
            getattr(self, "timeout_seconds", DEFAULT_OPENAI_TIMEOUT_SECONDS),
        )
        _LAST_USAGE.value = _usage_dict(response)
        return response.choices[0].message.content


class DeepSeekClient(OpenAIClient):
    """OpenAI-compatible client with explicit DeepSeek thinking control."""

    def supports_multimodal_images(self) -> bool:
        # The DeepSeek chat endpoint does not accept image_url content parts.
        return False

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        joint_diagnosis_max_tokens: Optional[int] = None,
    ):
        super().__init__(api_key=api_key, base_url=base_url, default_model=default_model)
        self.reasoning_effort = reasoning_effort
        self.joint_diagnosis_max_tokens = _optional_positive_int(joint_diagnosis_max_tokens)

    def chat_completions_create(
        self,
        messages: list,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        if self.reasoning_effort and "reasoning_effort" not in kwargs:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return super().chat_completions_create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )


class AzureOpenAIClient(LLMClient):
    """Azure OpenAI API client"""

    def supports_multimodal_images(self) -> bool:
        return True
    
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        api_version: str = "2024-12-01-preview",
        deployment: Optional[str] = None
    ):
        try:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                api_key=api_key,
            )
            self.deployment = deployment
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
    
    def chat_completions_create(
        self,
        messages: list,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        # For Azure, use deployment name as model
        deployment = model or self.deployment
        if not deployment:
            raise ValueError("Azure OpenAI requires deployment name (model parameter)")
        
        # Build call arguments
        call_kwargs = {
            'model': deployment,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens or 4096,
        }
        
        # Merge with additional kwargs (including response_format if provided)
        call_kwargs.update(kwargs)
        
        response = self.client.chat.completions.create(**call_kwargs)
        
        return response.choices[0].message.content


class ClaudeClient(LLMClient):
    """Anthropic Claude API client"""
    
    def __init__(self, api_key: str):
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
    
    def chat_completions_create(
        self,
        messages: list,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        model = model or "claude-3-5-sonnet-20241022"
        max_tokens = max_tokens or 4096
        
        # Convert messages format for Claude (Claude uses different format)
        # Claude expects system message separately and messages without system role
        system_message = None
        claude_messages = []
        
        for msg in messages:
            if msg.get('role') == 'system':
                system_message = msg.get('content')
            else:
                claude_messages.append({
                    'role': msg.get('role'),
                    'content': msg.get('content')
                })
        
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_message,
            messages=claude_messages,
            **kwargs
        )
        
        # Claude returns content as a list of text blocks
        if response.content and len(response.content) > 0:
            return response.content[0].text
        return ""


class MockLLMClient(LLMClient):
    """Mock LLM client for testing (returns empty actions)"""
    
    def chat_completions_create(
        self,
        messages: list,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        import json
        return json.dumps({
            "planning": "Mock planning: I will wait and observe.",
            "actions": []
        })


def create_llm_client(config: Optional[Dict[str, Any]] = None) -> Optional[LLMClient]:
    """
    Create an LLM client based on configuration.
    
    Configuration can be provided via:
    1. config dict parameter
    2. Environment variables
    
    Environment variables (for Azure OpenAI):
    - LLM_PROVIDER: "azure", "openai", "deepseek", "claude", or "mock"
    - AZURE_OPENAI_ENDPOINT: Azure endpoint URL
    - AZURE_OPENAI_API_KEY: Azure API key
    - AZURE_OPENAI_DEPLOYMENT: Deployment name (e.g., "gpt-4o")
    - AZURE_OPENAI_API_VERSION: API version (default: "2024-12-01-preview")
    - OPENAI_API_KEY: OpenAI API key
    - OPENAI_REASONING_EFFORT: optional reasoning effort for OpenAI reasoning models (unset: model default)
    - OPENAI_TIMEOUT_SECONDS: total time limit per call, retries included (default 60)
    - OPENAI_MAX_RETRIES: SDK retries per call (default 1)
    - DEEPSEEK_API_KEY: DeepSeek API key
    - DEEPSEEK_BASE_URL: DeepSeek OpenAI-compatible base URL
    - DEEPSEEK_MODEL: DeepSeek model name
    - DEEPSEEK_REASONING_EFFORT: one of none, low, high, or max
    - DEEPSEEK_JOINT_DIAGNOSIS_MAX_TOKENS: optional H/W completion-token budget
    - ANTHROPIC_API_KEY: Anthropic API key
    
    Args:
        config: Optional configuration dict with provider settings
    
    Returns:
        LLMClient instance or None if no valid configuration found
    """
    # Use config if provided, otherwise read from environment
    if config:
        provider = config.get('provider', 'mock').lower()
    else:
        provider = os.getenv('LLM_PROVIDER', 'mock').lower()
    
    try:
        if provider == 'azure':
            # Azure OpenAI configuration
            if config:
                endpoint = config.get('endpoint')
                api_key = config.get('api_key')
                deployment = config.get('deployment')
                api_version = config.get('api_version', '2024-12-01-preview')
            else:
                endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
                api_key = os.getenv('AZURE_OPENAI_API_KEY')
                deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o')
                api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')
            
            if not endpoint or not api_key:
                print('[LLMClient] Warning: Azure OpenAI endpoint or API key not set. Using mock LLM.')
                return MockLLMClient()
            
            if not deployment:
                print('[LLMClient] Warning: Azure OpenAI deployment not set. Using mock LLM.')
                return MockLLMClient()
            
            return AzureOpenAIClient(
                api_key=api_key,
                endpoint=endpoint,
                api_version=api_version,
                deployment=deployment
            )
        
        elif provider == 'openai':
            # OpenAI configuration
            if config:
                api_key = config.get('api_key')
                base_url = config.get('base_url')
                default_model = config.get('model')
                reasoning_effort = config.get('reasoning_effort')
            else:
                api_key = os.getenv('OPENAI_API_KEY')
                base_url = os.getenv('OPENAI_BASE_URL')
                default_model = os.getenv('OPENAI_MODEL')
                reasoning_effort = os.getenv('OPENAI_REASONING_EFFORT')
            
            if not api_key:
                print('[LLMClient] Warning: OPENAI_API_KEY not set. Using mock LLM.')
                return MockLLMClient()
            
            return OpenAIClient(
                api_key=api_key,
                base_url=base_url,
                default_model=default_model,
                reasoning_effort=reasoning_effort or None,
            )

        elif provider == 'deepseek':
            # DeepSeek exposes an OpenAI-compatible Chat Completions API, but
            # keep its credentials and model selection separate from the
            # project's OpenAI-compatible proxy profiles.
            if config:
                api_key = config.get('api_key')
                base_url = config.get('base_url')
                default_model = config.get('model')
                reasoning_effort = config.get('reasoning_effort')
                joint_diagnosis_max_tokens = config.get('joint_diagnosis_max_tokens')
            else:
                api_key = os.getenv('DEEPSEEK_API_KEY')
                base_url = os.getenv('DEEPSEEK_BASE_URL')
                default_model = os.getenv('DEEPSEEK_MODEL')
                reasoning_effort = os.getenv('DEEPSEEK_REASONING_EFFORT')
                joint_diagnosis_max_tokens = os.getenv('DEEPSEEK_JOINT_DIAGNOSIS_MAX_TOKENS')

            if not api_key:
                print('[LLMClient] Warning: DEEPSEEK_API_KEY not set. Using mock LLM.')
                return MockLLMClient()

            return DeepSeekClient(
                api_key=api_key,
                base_url=base_url,
                default_model=default_model or 'deepseek-v4-pro',
                reasoning_effort=reasoning_effort,
                joint_diagnosis_max_tokens=_optional_positive_int(joint_diagnosis_max_tokens),
            )
        
        elif provider == 'claude':
            # Anthropic Claude configuration
            if config:
                api_key = config.get('api_key')
            else:
                api_key = os.getenv('ANTHROPIC_API_KEY')
            
            if not api_key:
                print('[LLMClient] Warning: ANTHROPIC_API_KEY not set. Using mock LLM.')
                return MockLLMClient()
            
            return ClaudeClient(api_key=api_key)
        
        else:
            # Default to mock
            print(f'[LLMClient] Unknown provider "{provider}". Using mock LLM.')
            return MockLLMClient()
    
    except Exception as e:
        print(f'[LLMClient] Error creating LLM client: {e}. Using mock LLM.')
        return MockLLMClient()
