import json
import logging
from pathlib import Path
from typing import Any, Callable, TypeVar

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, ClaudeSDKError, ResultMessage
from pydantic import BaseModel, ValidationError

log = logging.getLogger("aim")

T = TypeVar("T", bound=BaseModel)


class AgentError(RuntimeError):
    pass


def dump(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [dump(v) for v in value]
    return value


def to_prompt(payload: dict[str, Any]) -> str:
    return json.dumps({k: dump(v) for k, v in payload.items()}, indent=2, ensure_ascii=False)


def agent_options(
    system_prompt: str,
    output_model: type[BaseModel],
    tools: list[str] | None = None,
    cwd: Path | None = None,
    max_turns: int = 4,
) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        system_prompt=system_prompt,
        tools=tools or [],
        allowed_tools=tools or [],
        cwd=cwd,
        max_turns=max_turns,
        setting_sources=[],
        strict_mcp_config=True,
        output_format={"type": "json_schema", "schema": output_model.model_json_schema()},
    )


def failure_detail(result: ResultMessage | None) -> str:
    if result is None:
        return "no result message"
    parts = [f"subtype={result.subtype}", f"is_error={result.is_error}"]
    if result.api_error_status:
        parts.append(f"API status {result.api_error_status}")
    if result.errors:
        parts.append(f"errors={result.errors}")
    if result.result:
        parts.append(f"result: {result.result[:300]}")
    return ", ".join(parts)


async def send(client: ClaudeSDKClient, agent: str, prompt: str, output_model: type[T]) -> T:
    await client.query(prompt)
    result = None
    async for message in client.receive_response():
        if isinstance(message, ResultMessage):
            result = message
    if result is None or result.is_error or result.structured_output is None:
        raise AgentError(f"{agent} produced no structured output ({failure_detail(result)})")
    log.info("%s finished in %d turns ($%.4f)", agent, result.num_turns, result.total_cost_usd or 0)
    try:
        return output_model.model_validate(result.structured_output)
    except ValidationError as e:
        raise AgentError(f"{agent} returned output that does not match {output_model.__name__}: {e}") from e


async def ask(
    agent: str,
    system_prompt: str,
    payload: dict[str, Any],
    output_model: type[T],
    check: Callable[[T], None] = lambda _: None,
    retries: int = 2,
    **options: Any,
) -> T:
    prompt = to_prompt(payload)
    feedback = ""
    for attempt in range(retries + 1):
        try:
            async with ClaudeSDKClient(agent_options(system_prompt, output_model, **options)) as client:
                output = await send(client, agent, prompt + feedback, output_model)
            check(output)
            return output
        except (AgentError, ClaudeSDKError, ValueError) as e:
            log.warning("%s attempt %d/%d failed: %s", agent, attempt + 1, retries + 1, e)
            feedback = f"\n\nYour previous answer was rejected for this reason, fix it:\n{e}"
            if attempt == retries:
                raise AgentError(f"{agent} failed after {retries + 1} attempts: {e}") from e
