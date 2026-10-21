import json
import re
from typing import TypeVar

from openai import APITimeoutError, OpenAI
from pydantic import BaseModel, ValidationError

from config import BASE_URL, LLM_MODEL, LLM_REASONING_EFFORT, LLM_TIMEOUT

client = OpenAI(
    base_url=BASE_URL,
    api_key="lm_studio",
    timeout=LLM_TIMEOUT,
    max_retries=0,
)

T = TypeVar("T", bound=BaseModel)


def ask_model(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    response_format: dict | None = None,
    reasoning_effort: str | None = LLM_REASONING_EFFORT,
) -> str:
    kwargs = {}
    if response_format:
        kwargs["response_format"] = response_format
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        **kwargs,
    )
    return strip_thinking(response.choices[0].message.content or "")


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def _extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end != -1 else text


def schema_example(node: dict, defs: dict) -> object:
    if "$ref" in node:
        node = defs[node["$ref"].split("/")[-1]]
    kind = node.get("type")
    if kind == "object":
        return {
            name: schema_example(prop, defs)
            for name, prop in node.get("properties", {}).items()
        }
    if kind == "array":
        return [schema_example(node.get("items", {}), defs)]
    if kind in ("number", "integer"):
        return 0.0 if kind == "number" else 0
    if kind == "boolean":
        return True
    return f"<{node.get('description', node.get('title', 'текст'))}>"


def ask_structured(
    system_prompt: str,
    user_prompt: str,
    schema: type[T],
    strict_attempts: int = 2,
    reasoning_effort: str | None = LLM_REASONING_EFFORT,
) -> T:
    json_schema = schema.model_json_schema()
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "strict": True,
            "schema": json_schema,
        },
    }
    example = schema_example(json_schema, json_schema.get("$defs", {}))
    example_hint = (
        "\n\nОтветь ТОЛЬКО JSON-объектом такой структуры "
        "(в угловых скобках — что подставить, сами скобки не пиши):\n"
        + json.dumps(example, ensure_ascii=False, indent=2)
    )

    errors: list[str] = []
    for attempt in range(strict_attempts + 1):
        strict = attempt < strict_attempts
        raw = ""
        try:
            if strict:
                raw = ask_model(
                    system_prompt,
                    user_prompt,
                    response_format=response_format,
                    reasoning_effort=reasoning_effort,
                )
            else:
                raw = ask_model(
                    system_prompt + example_hint,
                    user_prompt,
                    reasoning_effort=reasoning_effort,
                )
            return schema.model_validate_json(_extract_json(raw))
        except APITimeoutError as error:
            raise RuntimeError(
                f"Модель не ответила за {LLM_TIMEOUT:.0f} с. Увеличь "
                "LLM_TIMEOUT или отключи размышления: "
                "LLM_REASONING_EFFORT=none"
            ) from error
        except Exception as error:
            mode = "json_schema" if strict else "пример"
            errors.append(
                f"попытка {attempt + 1} ({mode}): {type(error).__name__}: "
                f"{str(error)[:300]} | ответ: {raw[:200]!r}"
            )

    raise RuntimeError(
        f"Модель не вернула корректный JSON ({schema.__name__}):\n"
        + "\n".join(errors)
    )
