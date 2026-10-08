from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from l2l.errors import TranslationError, fail_config, fail_http
from l2l.monads import Ok, Result, result_bind

SOURCE_TOKEN = "<<<SOURCE>>>"
A_TOKEN = "<<<TRANSLATION_A>>>"
B_TOKEN = "<<<TRANSLATION_B>>>"
PROMPT_TOKEN = "<<<CURRENT_PROMPT>>>"
HISTORY_TOKEN = "<<<HISTORY>>>"
CRITIQUES_TOKEN = "<<<CRITIQUES>>>"


@dataclass(frozen=True)
class EndpointReply:
    text: str
    body: Mapping[str, Any]


def compact(text: str, limit: int = 400) -> str:
    return " ".join(text.split())[:limit]


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def chat_payload(
    model: str,
    content: str,
    temperature: float,
    maximum_tokens: int,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": maximum_tokens,
        **dict(extra),
    }


def endpoint_error(reply: EndpointReply) -> str | None:
    reported = reply.body.get("error")
    return None if reported is None else json.dumps(reported)[:400]


def reply_content(reply: EndpointReply) -> Result[str, TranslationError]:
    failure = endpoint_error(reply)
    if failure is not None:
        return fail_http("protocol", "endpoint error: %s" % failure)

    choices = reply.body.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, dict) else None
    choice = message.get("content") if isinstance(message, dict) else None
    if isinstance(choice, str) and choice.strip():
        return Ok(choice)

    finish = first.get("finish_reason") if isinstance(first, dict) else None
    reasoning = isinstance(message, dict) and bool(message.get("reasoning"))
    hint = (
        "; raise --call-max-tokens or cap reasoning, e.g. --judge-params "
        '\'{"reasoning": {"effort": "low"}}\''
        if finish == "length"
        else ""
    )
    return fail_http(
        "protocol",
        "empty content (finish_reason=%s, reasoning=%s)%s: %s"
        % (finish, reasoning, hint, compact(reply.text)),
    )


def extract_object(text: str) -> Result[dict[str, Any], TranslationError]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return fail_config("no JSON object in reply: %s" % compact(text))

    try:
        parsed: dict[str, Any] = json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        return fail_config("reply is not valid JSON: %s" % error)

    return Ok(parsed)


def usable_verdict(
    parsed: Result[dict[str, Any], TranslationError],
) -> Result[dict[str, Any], TranslationError]:
    def decide(verdict: dict[str, Any]) -> Result[dict[str, Any], TranslationError]:
        if verdict.get("winner") in ("A", "B", "tie"):
            return Ok(verdict)
        return fail_config("winner must be A, B, or tie")

    return result_bind(parsed, decide)


def error_verdict(text: str) -> dict[str, Any]:
    return {
        "winner": "error",
        "critique": "judge reply was unusable: %s" % compact(text),
        "evidence": [],
    }


def strip_reply(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        closing = len(lines) >= 2 and lines[-1].strip().startswith("```")
        stripped = "\n".join(lines[1:-1] if closing else lines[1:]).strip()
    return stripped


def fill_judge(template: str, source: str, left: str, right: str) -> str:
    return (
        template.replace(SOURCE_TOKEN, source)
        .replace(A_TOKEN, left)
        .replace(B_TOKEN, right)
    )


def fill_rewrite(
    template: str, current_prompt: str, history_text: str, critiques: str
) -> str:
    return (
        template.replace(PROMPT_TOKEN, current_prompt)
        .replace(HISTORY_TOKEN, history_text)
        .replace(CRITIQUES_TOKEN, critiques)
    )


def order_outcome(winner: str, candidate_side: str) -> str:
    if winner not in ("A", "B"):
        return "tie"
    if winner == candidate_side:
        return "candidate"
    return "incumbent"


def feedback_lines(
    chapter: str, first: Mapping[str, Any], second: Mapping[str, Any]
) -> tuple[str, ...]:
    def describe_verdict(label: str, verdict: Mapping[str, Any]) -> str:
        critique = str(verdict.get("critique") or "").strip()
        raw = verdict.get("evidence")
        evidence = "; ".join(str(item) for item in raw) if isinstance(raw, list) else ""
        return "%s [%s] %s | evidence: %s" % (
            chapter,
            label,
            critique,
            evidence or "none quoted",
        )

    return (
        describe_verdict("incumbent shown as A", first),
        describe_verdict("candidate shown as A", second),
    )
