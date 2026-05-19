import asyncio
import json
import re
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from agent_knowledge_harvester.config import AnalysisStage, Settings
from agent_knowledge_harvester.utils.token_counter import estimate_tokens

LLMStage = AnalysisStage
LLM_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
LLM_MAX_ATTEMPTS = 3


class LLMConfigStatus(BaseModel):
    stage: str
    configured: bool
    model: str
    base_url: str
    missing: list[str] = Field(default_factory=list)


class LLMJsonResult(BaseModel):
    stage: str
    model: str
    prompt_tokens_estimate: int
    completion_tokens_estimate: int
    payload: dict[str, Any]


class LLMTextResult(BaseModel):
    stage: str
    model: str
    prompt_tokens_estimate: int
    completion_tokens_estimate: int
    content: str


class OpenAICompatibleLLMClient:
    """Minimal OpenAI-compatible JSON chat client."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def config_status(self, stage: LLMStage) -> LLMConfigStatus:
        api_key = self.settings.api_key_for_stage(stage)
        missing = [] if api_key else ["api_key"]
        return LLMConfigStatus(
            stage=stage,
            configured=bool(api_key),
            model=self.settings.model_for_stage(stage),
            base_url=str(self.settings.base_url_for_stage(stage)),
            missing=missing,
        )

    async def chat_json(
        self,
        stage: LLMStage,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        use_response_format: bool = True,
    ) -> LLMJsonResult:
        status = self.config_status(stage)
        if not status.configured:
            raise RuntimeError(
                f"LLM stage '{stage}' is not configured; missing: {', '.join(status.missing)}"
            )

        api_key = self.settings.api_key_for_stage(stage)
        base_url = str(self.settings.base_url_for_stage(stage)).rstrip("/")
        model = self.settings.model_for_stage(stage)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if use_response_format:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = f"{base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await post_chat_completion_with_retries(client, url, headers, payload)
            if response.status_code == 400 and use_response_format:
                payload.pop("response_format", None)
                response = await post_chat_completion_with_retries(client, url, headers, payload)
            response.raise_for_status()
            response_payload = response.json()

        content = response_payload["choices"][0]["message"]["content"]
        parsed = parse_json_content(content)
        return LLMJsonResult(
            stage=stage,
            model=model,
            prompt_tokens_estimate=estimate_tokens(system_prompt + "\n" + user_prompt),
            completion_tokens_estimate=estimate_tokens(content),
            payload=parsed,
        )

    async def chat_text(
        self,
        stage: LLMStage,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> LLMTextResult:
        """Call an OpenAI-compatible chat endpoint and return raw text content."""
        status = self.config_status(stage)
        if not status.configured:
            raise RuntimeError(
                f"LLM stage '{stage}' is not configured; missing: {', '.join(status.missing)}"
            )

        api_key = self.settings.api_key_for_stage(stage)
        base_url = str(self.settings.base_url_for_stage(stage)).rstrip("/")
        model = self.settings.model_for_stage(stage)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = f"{base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await post_chat_completion_with_retries(client, url, headers, payload)
            response.raise_for_status()
            response_payload = response.json()

        content = response_payload["choices"][0]["message"]["content"]
        return LLMTextResult(
            stage=stage,
            model=model,
            prompt_tokens_estimate=estimate_tokens(system_prompt + "\n" + user_prompt),
            completion_tokens_estimate=estimate_tokens(content),
            content=content,
        )


async def post_chat_completion_with_retries(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> httpx.Response:
    """Post to a chat completion endpoint with small transient-error retries."""
    last_error: Exception | None = None
    for attempt in range(LLM_MAX_ATTEMPTS):
        try:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code in LLM_RETRY_STATUS_CODES:
                response.raise_for_status()
            return response
        except (httpx.TransportError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            if isinstance(exc, httpx.HTTPStatusError):
                status_code = exc.response.status_code
                if status_code not in LLM_RETRY_STATUS_CODES:
                    raise
            last_error = exc
            if attempt == LLM_MAX_ATTEMPTS - 1:
                raise
            await asyncio.sleep(1.5 * (2**attempt))
    if last_error:
        raise last_error
    raise RuntimeError("LLM request failed without an exception")


def normalize_stage(value: str) -> Literal["screening", "extraction", "validation", "linking"]:
    if value not in {"screening", "extraction", "validation", "linking"}:
        raise ValueError("stage must be one of: screening, extraction, validation, linking")
    return value  # type: ignore[return-value]


def parse_json_content(content: str) -> dict[str, Any]:
    """Parse strict JSON, with a fallback for fenced or prefixed model output."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL)
        candidate = match.group(1) if match else extract_json_object(content)
        parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed


def extract_json_object(content: str) -> str:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response did not contain a JSON object")
    return content[start : end + 1]
