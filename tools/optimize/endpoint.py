from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from l2l.errors import TranslationError, describe, fail_http
from l2l.http import HttpRequest, curl_open, http_request, parse_json_body
from l2l.monads import (
    IO,
    Err,
    Ok,
    Result,
    io_and_then,
    io_bind,
    io_map,
    io_result,
    io_result_bind,
    result_bind,
)
from l2l.settings import Config

from .environment import Api, OpenEndpoint, OptimizationEnvironment
from .replies import EndpointReply, chat_payload, compact, reply_content

CHAT_ATTEMPTS = 4


def endpoint_config(api: Api, timeout: float) -> Config:
    return Config(
        base_url=api.base_url,
        api_key=api.api_key,
        model="",
        timeout=timeout,
        maximum_tokens=0,
        parameters=MappingProxyType({}),
    )


def curl_endpoint(
    api: Api, environment: Mapping[str, str], timeout: float
) -> OpenEndpoint:
    def open_endpoint(
        payload: Mapping[str, Any],
    ) -> IO[Result[EndpointReply, TranslationError]]:
        request: HttpRequest = http_request(endpoint_config(api, timeout), payload)

        def loaded(
            raw: Result[bytes, TranslationError],
        ) -> Result[EndpointReply, TranslationError]:
            parsed = parse_json_body(raw)
            if isinstance(parsed, Err):
                return parsed
            text, body = parsed.value
            if not isinstance(body, dict):
                return fail_http(
                    "protocol", "non-object endpoint reply: %s" % compact(text)
                )
            return Ok(EndpointReply(text, body))

        def respond(
            response: Any,
        ) -> IO[Result[EndpointReply, TranslationError]]:
            return io_map(response.body(), loaded)

        return io_result_bind(
            io_result(curl_open(request, timeout, environment)), respond
        )

    return open_endpoint


def complete(
    optimization_environment: OptimizationEnvironment,
    model: str,
    content: str,
    temperature: float,
    extra: Mapping[str, Any],
    label: str,
) -> IO[Result[str, TranslationError]]:
    payload = chat_payload(
        model,
        content,
        temperature,
        optimization_environment.options.call_maximum_tokens,
        extra,
    )

    def at(index: int) -> IO[Result[str, TranslationError]]:
        def decided(
            reply: Result[EndpointReply, TranslationError],
        ) -> IO[Result[str, TranslationError]]:
            outcome = result_bind(reply, reply_content)
            if isinstance(outcome, Ok) or index >= CHAT_ATTEMPTS:
                return io_result(outcome)

            def waited(_: None) -> IO[Result[str, TranslationError]]:
                return at(index + 1)

            return io_bind(
                optimization_environment.warn(
                    "%s: attempt %d/%d failed (%s)"
                    % (label, index, CHAT_ATTEMPTS, describe(outcome.error))
                ),
                lambda _: io_and_then(
                    IO(lambda: optimization_environment.sleep(2.0 * index)),
                    waited(None),
                ),
            )

        return io_bind(optimization_environment.opener(payload), decided)

    return at(1)
