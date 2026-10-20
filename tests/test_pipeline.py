import io
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from fakes import (
    FakeHttp,
    FakePlainResponse,
    FakeStreamResponse,
    chunk_pass,
    make_console,
    make_context,
    paragraph_pass,
    stream_chunks,
    update_pass,
    with_usage,
)

from l2l.errors import (
    InvalidOutputError,
    TranslationError,
    describe,
    fail_http,
)
from l2l.glossary import GlossaryEntry
from l2l.http import chat
from l2l.monads import (
    IO,
    Err,
    Ok,
    Result,
    fold_io,
    io_pure,
    io_result,
)
from l2l.pipeline import (
    State,
    analyze_document,
    conclude_pass_result,
    conclude_state,
    run_pipeline,
)
from l2l.settings import PassDefinition
from l2l.text import THINK_OPEN, Usage

USAGE = {"prompt_tokens": 5, "completion_tokens": 6, "cost": 0.2}


def test_fold_io_handles_thousands_of_items() -> None:
    def step(pair: tuple[int, None], _: int) -> IO[Result[tuple[int, None], str]]:
        return io_pure(Ok((pair[0] + 1, None)))

    outcome = fold_io(range(5000), step, Ok((0, None))).run()
    assert isinstance(outcome, Ok)
    assert outcome.value[0] == 5000


def test_fold_io_short_circuits_on_error() -> None:
    calls: list[int] = []

    def step(pair: tuple[int, None], item: int) -> IO[Result[tuple[int, None], str]]:
        calls.append(item)
        if item == 1:
            return io_result(Err("boom"))

        return io_pure(Ok((pair[0] + 1, None)))

    outcome = fold_io(range(4), step, Ok((0, None))).run()
    assert isinstance(outcome, Err)
    assert outcome.error == "boom"
    assert calls == [0, 1]


def test_chat_streamed_reports_usage() -> None:
    console, stderr = make_console()
    http = FakeHttp([FakeStreamResponse(with_usage(stream_chunks("Hi"), USAGE))])
    run_context = make_context(console, http.open)
    result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Ok)
    assert result.value.text == "Hi"
    assert result.value.usage == Usage(5, 6, 0.2)


def test_chat_streamed_filters_think_prefix() -> None:
    console, stderr = make_console()
    chunks = stream_chunks("<think>h</think>", "Body")
    http = FakeHttp([FakeStreamResponse(with_usage(chunks, USAGE))])
    run_context = make_context(console, http.open)
    result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Ok)
    assert result.value.text == "Body"


def test_chat_plain_when_stream_disabled() -> None:
    console, stderr = make_console()
    body = {
        "choices": [{"message": {"content": "Plain"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
    }
    http = FakeHttp([FakePlainResponse(body)])
    run_context = make_context(console, http.open)
    result = chat(run_context, "s", "u", "m", {"stream": False}, Usage()).run()
    assert isinstance(result, Ok)
    assert result.value.text == "Plain"
    payload = json.loads(http.requests[0].body)
    assert payload["stream"] is False


def test_chat_plain_swallows_unterminated_think_prefix() -> None:
    console, stderr = make_console()
    body = {
        "choices": [{"message": {"content": THINK_OPEN + "reasoning without close"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
    }
    http = FakeHttp([FakePlainResponse(body)])
    run_context = make_context(console, http.open)
    result = chat(run_context, "s", "u", "m", {"stream": False}, Usage()).run()
    assert isinstance(result, Ok)
    assert result.value.text == ""


def test_chat_rejects_oversized_request() -> None:
    console, stderr = make_console()
    http = FakeHttp([])
    run_context = make_context(console, http.open, maximum_tokens=1)
    result = chat(run_context, "s", "u", "m", {}, Usage()).run()
    assert isinstance(result, Err)
    assert "over the 1-token budget" in describe(result.error)
    assert http.requests == []


def test_run_pipeline_runs_analysis_before_translation() -> None:
    console, stderr = make_console()
    brief = with_usage(stream_chunks("Names: Qin Yu."), USAGE)
    translated = with_usage(stream_chunks("Hello."), USAGE)
    http = FakeHttp([FakeStreamResponse(brief), FakeStreamResponse(translated)])
    run_context = make_context(console, http.open)
    analysis = PassDefinition("prep", "Summarise.", "analysis", {}, None, False)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context, (analysis, chunk_pass()), "你好。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n"
    second_user = json.loads(http.requests[1].body)["messages"][1]["content"]
    assert "Names: Qin Yu." in second_user


def test_run_pipeline_wraps_analysis_failure() -> None:
    console, stderr = make_console()

    def open_fail(request: Any, timeout: float) -> Result[Any, TranslationError]:
        return fail_http("unreachable", "down")

    run_context = make_context(console, open_fail)
    analysis = PassDefinition("prep", "Summarise.", "analysis", {}, None, False)
    code = run_pipeline(run_context, (analysis,), "你好。", 0.0, io.StringIO()).run()
    assert code == 1
    logged = stderr.getvalue()
    assert "pass [prep] failed" in logged
    assert "could not reach endpoint: down" in logged


def test_run_pipeline_repairs_non_ascii_via_llm() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("Hi 中"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Hi there"), USAGE)),
        ]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context, (chunk_pass(ascii_output=True),), "文本。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hi there\n"
    assert len(http.requests) == 2


def test_run_pipeline_drops_non_ascii_after_failed_repairs() -> None:
    console, stderr = make_console()
    stubborn = "中 x"
    http = FakeHttp(
        [FakeStreamResponse(with_usage(stream_chunks(stubborn), USAGE))]
        + [
            FakeStreamResponse(with_usage(stream_chunks(stubborn), USAGE))
            for _ in range(3)
        ]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context, (chunk_pass(ascii_output=True),), "文本。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == " x\n"
    logged = stderr.getvalue()
    assert "still contained non-ASCII characters (中) after 3 LLM attempts" in logged
    assert len(http.requests) == 4


def test_retranslation_retranslates_an_echoed_paragraph() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(
                with_usage(stream_chunks("Hello.\n\n世界。\n\nThree."), USAGE)
            ),
            FakeStreamResponse(with_usage(stream_chunks("World."), USAGE)),
        ]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context,
        (chunk_pass(retranslate_untranslated=True),),
        "你好。\n\n世界。\n\n三。",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n\nThree.\n"
    assert len(http.requests) == 2
    logged = stderr.getvalue()
    assert "1 of 3 paragraph(s) look untranslated" in logged
    assert "(more than half)" not in logged
    retry = json.loads(http.requests[1].body)
    assert retry["messages"][0]["content"] == "T."
    retry_user = retry["messages"][1]["content"]
    assert retry_user.startswith("Context paragraphs (reference only")
    assert "你好。" in retry_user
    assert "三。" in retry_user
    assert retry_user.endswith("世界。")


def test_ascii_stage_preserves_retranslation_validated_and_problem() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("世界。"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("世界。"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("世界。"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("(world)."), USAGE)),
        ]
    )
    run_context = make_context(console, http.open, best_effort=True)
    pass_definition = chunk_pass(ascii_output=True, retranslate_untranslated=True)
    state = State("Hello.\n\n世界。", None, Usage())
    result = conclude_state(
        run_context, pass_definition, state, ("你好。", "世界。")
    ).run()
    assert isinstance(result, Ok)
    assert result.value.text == "Hello.\n\n(world)."
    assert result.value.validated is False
    assert result.value.problem is not None
    assert "世界。" in result.value.problem
    assert len(http.requests) == 4


def test_retranslation_accepts_non_latin_targets() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [FakeStreamResponse(with_usage(stream_chunks("Привет.\n\nМир."), USAGE))]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context,
        (chunk_pass(retranslate_untranslated=True),),
        "你好。\n\n世界。",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Привет.\n\nМир.\n"
    assert len(http.requests) == 1
    assert "look untranslated" not in stderr.getvalue()


def test_retranslation_retranslates_only_flagged_paragraphs() -> None:
    console, _ = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("Hello.\n\n世界。"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("World."), USAGE)),
        ]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context,
        (chunk_pass(retranslate_untranslated=True),),
        "你好。\n\n世界。",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n"
    assert len(http.requests) == 2


def test_retranslation_fails_when_paragraph_still_untranslated() -> None:
    console, stderr = make_console()
    echo = "你好。\n\n世界。"
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks(echo), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(echo), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("你好。"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("世界。"), USAGE)),
        ]
    )
    run_context = make_context(console, http.open)
    code = run_pipeline(
        run_context,
        (chunk_pass(retranslate_untranslated=True),),
        echo,
        0.0,
        io.StringIO(),
    ).run()
    assert code == 1
    assert len(http.requests) == 4
    logged = stderr.getvalue()
    assert "2 of 2 paragraph(s) look untranslated (more than half)" in logged
    assert (
        "source script cjk, target script unknown; using the ASCII-letter rule"
        in logged
    )
    assert "still untranslated after retranslation" in logged
    assert "keeping the last reply" not in logged
    assert "你好。" in logged


def test_retranslation_ignores_letterless_paragraph_with_numbers() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [FakeStreamResponse(with_usage(stream_chunks("Hello.\n\n2078!"), USAGE))]
    )
    run_context = make_context(console, http.open, verbose=True)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context,
        (chunk_pass(retranslate_untranslated=True),),
        "你好。\n\n2078年！",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\n2078!\n"
    assert len(http.requests) == 1
    assert "look untranslated" not in stderr.getvalue()


def test_retranslation_accepts_paragraph_already_in_target_script() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("Hello.\n\nBIU!"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("BIU!"), USAGE)),
        ]
    )
    run_context = make_context(console, http.open, verbose=True)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context,
        (chunk_pass(retranslate_untranslated=True),),
        "你好。\n\nBIU！",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nBIU!\n"
    assert len(http.requests) == 2
    assert "already in the target script" in stderr.getvalue()


def test_retranslation_target_script_results_are_cached(tmp_path: Path) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()

    def run() -> tuple[int, str, int]:
        console, _ = make_console()
        http = FakeHttp(
            [
                FakeStreamResponse(with_usage(stream_chunks("Hello.\n\nBIU!"), USAGE)),
                FakeStreamResponse(with_usage(stream_chunks("BIU!"), USAGE)),
            ]
        )
        run_context = make_context(
            console,
            http.open,
            cache_directory=str(cache_directory),
            use_cache=True,
        )
        stdout = io.StringIO()
        code = run_pipeline(
            run_context,
            (chunk_pass(retranslate_untranslated=True),),
            "你好。\n\nBIU！",
            0.0,
            stdout,
        ).run()
        return code, stdout.getvalue(), len(http.requests)

    first_code, first_output, first_calls = run()
    second_code, second_output, second_calls = run()
    assert (first_code, second_code) == (0, 0)
    assert (first_calls, second_calls) == (2, 0)
    assert first_output == second_output == "Hello.\n\nBIU!\n"


def test_retranslation_best_effort_keeps_untranslated_paragraph() -> None:
    console, stderr = make_console()
    echo = "你好。\n\n世界。"
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks(echo), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(echo), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("你好。"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("世界。"), USAGE)),
        ]
    )
    run_context = make_context(console, http.open, best_effort=True)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context,
        (chunk_pass(retranslate_untranslated=True),),
        echo,
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "你好。\n\n世界。\n"
    assert "keeping the last reply" in stderr.getvalue()
    assert len(http.requests) == 4


def test_retranslation_disabled_by_default() -> None:
    console, _ = make_console()
    http = FakeHttp(
        [FakeStreamResponse(with_usage(stream_chunks("你好。\n\n世界。"), USAGE))]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context, (chunk_pass(),), "你好。\n\n世界。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == "你好。\n\n世界。\n"
    assert len(http.requests) == 1


def test_retranslation_skipped_when_paragraph_count_differs() -> None:
    console, _ = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(
                with_usage(stream_chunks("Hello.\n\n世界。\n\n世界。"), USAGE)
            )
        ]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context,
        (chunk_pass(retranslate_untranslated=True),),
        "你好。\n\n世界。",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert len(http.requests) == 1


def test_retranslation_ignores_letterless_paragraphs() -> None:
    console, _ = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("你好。\n\n123"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE)),
        ]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context,
        (chunk_pass(retranslate_untranslated=True),),
        "你好。\n\n123",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\n123\n"
    assert len(http.requests) == 2


def test_retranslation_runs_before_ascii_enforcement() -> None:
    console, _ = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("你好，世界。"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks('Hello, "world".'), USAGE)),
        ]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context,
        (chunk_pass(ascii_output=True, retranslate_untranslated=True),),
        "你好，世界。",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == 'Hello, "world".\n'
    assert len(http.requests) == 2


def test_retranslation_results_are_cached(tmp_path: Path) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()

    def run() -> tuple[int, str, int]:
        console, _ = make_console()
        echo = "你好。\n\n世界。"
        http = FakeHttp(
            [
                FakeStreamResponse(with_usage(stream_chunks(echo), USAGE)),
                FakeStreamResponse(with_usage(stream_chunks(echo), USAGE)),
                FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE)),
                FakeStreamResponse(with_usage(stream_chunks("World."), USAGE)),
            ]
        )
        run_context = make_context(
            console,
            http.open,
            cache_directory=str(cache_directory),
            use_cache=True,
        )
        stdout = io.StringIO()
        code = run_pipeline(
            run_context,
            (chunk_pass(retranslate_untranslated=True),),
            "你好。\n\n世界。",
            0.0,
            stdout,
        ).run()
        return code, stdout.getvalue(), len(http.requests)

    first_code, first_output, first_calls = run()
    second_code, second_output, second_calls = run()
    assert (first_code, second_code) == (0, 0)
    assert (first_calls, second_calls) == (4, 0)
    assert first_output == second_output == "Hello.\n\nWorld.\n"


def test_retranslation_majority_retries_pass_then_succeeds() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("你好。\n\n世界。"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Hello.\n\nWorld."), USAGE)),
        ]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context,
        (chunk_pass(retranslate_untranslated=True),),
        "你好。\n\n世界。",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n"
    assert len(http.requests) == 2
    logged = stderr.getvalue()
    assert "2 of 2 paragraph(s) look untranslated (more than half)" in logged
    assert "retranslating them" not in logged


def test_retranslation_majority_retries_then_falls_back_per_paragraph() -> None:
    console, stderr = make_console()
    echo = "你好。\n\n世界。"
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks(echo), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(echo), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("World."), USAGE)),
        ]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context,
        (chunk_pass(retranslate_untranslated=True),),
        echo,
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n"
    assert len(http.requests) == 4
    logged = stderr.getvalue()
    assert logged.count("(more than half)") == 1
    assert "2 of 2 paragraph(s) look untranslated; retranslating them" in logged


def test_retranslation_majority_retry_uses_fresh_cache_keys(tmp_path: Path) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    echo = "你好。\n\n世界。"

    def first_run() -> tuple[int, str, int]:
        console, _ = make_console()
        http = FakeHttp(
            [
                FakeStreamResponse(with_usage(stream_chunks(echo), USAGE)),
                FakeStreamResponse(
                    with_usage(stream_chunks("Hello.\n\nWorld."), USAGE)
                ),
            ]
        )
        run_context = make_context(
            console,
            http.open,
            cache_directory=str(cache_directory),
            use_cache=True,
        )
        stdout = io.StringIO()
        code = run_pipeline(
            run_context,
            (chunk_pass(retranslate_untranslated=True),),
            echo,
            0.0,
            stdout,
        ).run()
        return code, stdout.getvalue(), len(http.requests)

    def second_run() -> tuple[int, str, int]:
        console, _ = make_console()
        http = FakeHttp([FakeStreamResponse(with_usage(stream_chunks("BAD"), USAGE))])
        run_context = make_context(
            console,
            http.open,
            cache_directory=str(cache_directory),
            use_cache=True,
        )
        stdout = io.StringIO()
        code = run_pipeline(
            run_context,
            (chunk_pass(retranslate_untranslated=True),),
            echo,
            0.0,
            stdout,
        ).run()
        return code, stdout.getvalue(), len(http.requests)

    first_code, first_output, first_calls = first_run()
    second_code, second_output, second_calls = second_run()
    assert (first_code, second_code) == (0, 0)
    assert (first_calls, second_calls) == (2, 0)
    assert first_output == second_output == "Hello.\n\nWorld.\n"


def test_retranslation_does_not_replay_an_echo_cached_by_the_pass(
    tmp_path: Path,
) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()

    def run(replies: list[FakeStreamResponse]) -> tuple[int, str, int]:
        console, _ = make_console()
        http = FakeHttp(replies)
        run_context = make_context(
            console,
            http.open,
            cache_directory=str(cache_directory),
            use_cache=True,
        )
        stdout = io.StringIO()
        code = run_pipeline(
            run_context,
            (paragraph_pass(retranslate_untranslated=True),),
            "你好。",
            0.0,
            stdout,
        ).run()
        return code, stdout.getvalue(), len(http.requests)

    first_code, first_output, first_calls = run(
        [
            FakeStreamResponse(with_usage(stream_chunks("你好。"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE)),
        ]
    )
    second_code, second_output, second_calls = run(
        [FakeStreamResponse(with_usage(stream_chunks("SURPRISE"), USAGE))]
    )
    assert (first_code, second_code) == (0, 0)
    assert (first_calls, second_calls) == (2, 0)
    assert first_output == second_output == "Hello.\n"


def test_failed_retranslation_is_not_cached(tmp_path: Path) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()

    def run(replies: list[FakeStreamResponse]) -> tuple[int, int]:
        console, _ = make_console()
        http = FakeHttp(replies)
        run_context = make_context(
            console,
            http.open,
            cache_directory=str(cache_directory),
            use_cache=True,
        )
        code = run_pipeline(
            run_context,
            (paragraph_pass(retranslate_untranslated=True),),
            "你好。",
            0.0,
            io.StringIO(),
        ).run()
        return code, len(http.requests)

    first_code, first_calls = run(
        [
            FakeStreamResponse(with_usage(stream_chunks("你好。"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("你好。"), USAGE)),
        ]
    )
    second_code, second_calls = run(
        [FakeStreamResponse(with_usage(stream_chunks("你好。"), USAGE))]
    )
    assert (first_code, second_code) == (1, 1)
    assert (first_calls, second_calls) == (2, 1)
    assert len(list(cache_directory.glob("*.txt"))) == 1


def test_analysis_brief_participates_in_cache_identity(tmp_path: Path) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()

    def run(
        analysis_instruction: str, replies: list[FakeStreamResponse]
    ) -> tuple[int, str, int]:
        console, _ = make_console()
        http = FakeHttp(replies)
        run_context = make_context(
            console,
            http.open,
            cache_directory=str(cache_directory),
            use_cache=True,
        )
        stdout = io.StringIO()
        code = run_pipeline(
            run_context,
            (
                PassDefinition(
                    "prep", analysis_instruction, "analysis", {}, None, False
                ),
                chunk_pass(),
            ),
            "你好。",
            0.0,
            stdout,
        ).run()
        return code, stdout.getvalue(), len(http.requests)

    first_code, first_output, first_calls = run(
        "Glossary: 你好 = Alice.",
        [
            FakeStreamResponse(with_usage(stream_chunks("Use Alice."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Alice says hello."), USAGE)),
        ],
    )
    second_code, second_output, second_calls = run(
        "Glossary: 你好 = Bob.",
        [
            FakeStreamResponse(with_usage(stream_chunks("Use Bob."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Bob says hello."), USAGE)),
        ],
    )
    assert (first_code, second_code) == (0, 0)
    assert (first_calls, second_calls) == (2, 2)
    assert first_output == "Alice says hello.\n"
    assert second_output == "Bob says hello.\n"


def test_retranslation_paragraph_mode_skips_majority_retry() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("你好。"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("世界。"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("World."), USAGE)),
        ]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context,
        (paragraph_pass(retranslate_untranslated=True),),
        "你好。\n\n世界。",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n"
    assert len(http.requests) == 4
    logged = stderr.getvalue()
    assert "(more than half)" not in logged
    assert "2 of 2 paragraph(s) look untranslated; retranslating them" in logged


def test_retranslation_at_exact_half_keeps_per_paragraph_path() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(
                with_usage(stream_chunks("Hello.\n\n世界。\n\nWarm.\n\n冬天。"), USAGE)
            ),
            FakeStreamResponse(with_usage(stream_chunks("World."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Winter."), USAGE)),
        ]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context,
        (chunk_pass(retranslate_untranslated=True),),
        "你好。\n\n世界。\n\n春天。\n\n冬天。",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n\nWarm.\n\nWinter.\n"
    assert len(http.requests) == 3
    logged = stderr.getvalue()
    assert "2 of 4 paragraph(s) look untranslated; retranslating them" in logged
    assert "(more than half)" not in logged


def test_retranslation_skips_analysis_passes() -> None:
    console, _ = make_console()
    brief = with_usage(stream_chunks("Brief."), USAGE)
    http = FakeHttp([FakeStreamResponse(brief)])
    run_context = make_context(console, http.open)
    analysis = PassDefinition(
        "prep", "Summarise.", "analysis", {}, None, False, None, True
    )
    stdout = io.StringIO()
    code = run_pipeline(run_context, (analysis,), "你好。", 0.0, stdout).run()
    assert code == 0
    assert stdout.getvalue() == "你好。\n"
    assert len(http.requests) == 1


def test_ascii_enforcement_skips_analysis_passes() -> None:
    console, _ = make_console()
    brief = with_usage(stream_chunks("Brief."), USAGE)
    http = FakeHttp([FakeStreamResponse(brief)])
    run_context = make_context(console, http.open)
    analysis = PassDefinition("prep", "Summarise.", "analysis", {}, None, True)
    stdout = io.StringIO()
    code = run_pipeline(run_context, (analysis,), "你好。\n\n世界。", 0.0, stdout).run()
    assert code == 0
    assert stdout.getvalue() == "你好。\n\n世界。\n"
    assert len(http.requests) == 1


def test_run_pipeline_translates() -> None:
    console, stderr = make_console()
    chunks = with_usage(stream_chunks("Hello.\n\nWorld."), USAGE)
    http = FakeHttp([FakeStreamResponse(chunks)])
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context, (chunk_pass(),), "你好。\n\n世界。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n"
    assert "TOTAL" in stderr.getvalue()


def test_run_pipeline_logs_finished_units_when_verbose() -> None:
    console, stderr = make_console()
    chunks = with_usage(stream_chunks("Hello."), USAGE)
    http = FakeHttp([FakeStreamResponse(chunks)])
    run_context = make_context(console, http.open, verbose=True)
    stdout = io.StringIO()
    code = run_pipeline(run_context, (chunk_pass(),), "你好。", 0.0, stdout).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n"
    assert "l2l: [translate] unit 1/1 done" in stderr.getvalue()


def test_run_pipeline_total_elapsed_measures_since_started() -> None:
    console, stderr = make_console()
    http = FakeHttp([FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE))])
    readings: list[float] = []
    current = 100.0

    def clock() -> float:
        nonlocal current
        current += 5.0
        readings.append(current)
        return current

    run_context = make_context(console, http.open, clock=clock)
    stdout = io.StringIO()
    started = clock()
    code = run_pipeline(run_context, (chunk_pass(),), "你好。", started, stdout).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n"
    elapsed = readings[-1] - readings[0]
    total = [line for line in stderr.getvalue().splitlines() if "TOTAL" in line][0]
    assert "TOTAL: %.1fs" % elapsed in total
    rate = USAGE["completion_tokens"] / elapsed if elapsed > 0 else 0.0
    assert "%.1f tok/s" % rate in total


def test_run_pipeline_reports_unit_failure() -> None:
    console, stderr = make_console()

    def open_fail(request: Any, timeout: float) -> Result[Any, TranslationError]:
        return fail_http("unreachable", "down")

    run_context = make_context(console, open_fail)
    code = run_pipeline(
        run_context, (chunk_pass(),), "你好。", 0.0, io.StringIO()
    ).run()
    assert code == 1
    logged = stderr.getvalue()
    assert "failed on unit 1" in logged
    assert "could not reach endpoint: down" in logged


def test_ensure_paragraphs_propagates_a_failed_first_attempt() -> None:
    console, stderr = make_console()

    def open_fail(request: Any, timeout: float) -> Result[Any, TranslationError]:
        return fail_http("unreachable", "down")

    run_context = make_context(console, open_fail)
    code = run_pipeline(
        run_context, (chunk_pass(ensure_paragraphs=True),), "你好。", 0.0, io.StringIO()
    ).run()
    assert code == 1
    assert "failed on unit 1" in stderr.getvalue()


def test_ensure_paragraphs_reports_a_failed_retry_attempt() -> None:
    console, stderr = make_console()
    stubborn = "One.\n\nTwo.\n\nThree."
    requests: list[Any] = []

    def open_thrice_then_fail(
        request: Any, timeout: float
    ) -> Result[Any, TranslationError]:
        requests.append(request)
        if len(requests) <= 3:
            return Ok(FakeStreamResponse(with_usage(stream_chunks(stubborn), USAGE)))

        return fail_http("unreachable", "down")

    run_context = make_context(console, open_thrice_then_fail)
    code = run_pipeline(
        run_context,
        (chunk_pass(ensure_paragraphs=True),),
        "你好。\n\n世界。",
        0.0,
        io.StringIO(),
    ).run()
    assert code == 1
    logged = stderr.getvalue()
    assert "re-running the pass with one call per paragraph" in logged
    assert "failed on unit 1" in logged
    assert len(requests) == 6


def test_best_effort_paragraph_pass_continues_when_counts_still_differ() -> None:
    console, stderr = make_console()
    stubborn = "One.\n\nTwo."
    http = FakeHttp(
        [FakeStreamResponse(with_usage(stream_chunks(stubborn), USAGE))] * 3
    )
    run_context = make_context(console, http.open, best_effort=True)
    stdout = io.StringIO()
    merged = PassDefinition("translate", "T.", "paragraph", {}, None, False, True)
    code = run_pipeline(run_context, (merged,), "你好。", 0.0, stdout).run()
    assert code == 0
    assert stdout.getvalue() == stubborn + "\n"
    logged = stderr.getvalue()
    assert "failed validation 2 time(s)" in logged
    assert "output has 2 paragraph(s), source has 1" in logged
    assert "continuing" in logged
    assert len(http.requests) == 3


def test_strict_paragraph_pass_fails_when_counts_still_differ() -> None:
    console, stderr = make_console()
    stubborn = "One.\n\nTwo."
    http = FakeHttp(
        [FakeStreamResponse(with_usage(stream_chunks(stubborn), USAGE))] * 3
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    merged = PassDefinition("translate", "T.", "paragraph", {}, None, False, True)
    code = run_pipeline(run_context, (merged,), "你好。", 0.0, stdout).run()
    assert code == 1
    assert stdout.getvalue() == ""
    logged = stderr.getvalue()
    assert "failed validation 2 time(s)" in logged
    assert "continuing" in logged
    assert "pass [translate] output still failed validation" in logged
    assert "still failed validation after 2 repair attempt(s)" in logged
    assert "--best-effort" in logged
    assert len(http.requests) == 3


def test_conclude_pass_result_attaches_the_pass_name_to_invalid_output() -> None:
    console, _ = make_console()
    run_context = make_context(console, FakeHttp([]).open)
    state = State("One.\n\nTwo.", None, Usage(), validated=False, problem="bad shape")
    result = conclude_pass_result(run_context, chunk_pass(), Ok(state))
    assert isinstance(result, Err)
    assert isinstance(result.error, InvalidOutputError)
    assert result.error.pass_name == "translate"
    assert result.error.attempts == run_context.settings.unit_fix_attempts
    assert result.error.problem == "bad shape"


def test_paragraph_pass_after_growth_passes_units_without_source() -> None:
    console, _ = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("Hello.\n\nWorld."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE)),
        ]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context, (chunk_pass(), paragraph_pass()), "你好。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.World.\n"
    assert len(http.requests) == 2
    users = [
        json.loads(request.body)["messages"][1]["content"] for request in http.requests
    ]
    assert "你好。" in users[1]


def test_refine_pass_echoing_source_language_is_repaired_with_draft() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("Hello.\n\nWorld."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("你好。\n\n世界。"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("你好。\n\n世界。"), USAGE)),
            FakeStreamResponse(
                with_usage(stream_chunks("Hello again.\n\nWorld again."), USAGE)
            ),
        ]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    refine = PassDefinition("refine", "R.", "chunk", {}, None, False, False, False)
    code = run_pipeline(
        run_context, (chunk_pass(), refine), "你好。\n\n世界。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello again.\n\nWorld again.\n"
    assert len(http.requests) == 4
    initial_user = json.loads(http.requests[1].body)["messages"][1]["content"]
    assert "Source text:\n你好。\n\n世界。" in initial_user
    assert "Current draft:\nHello.\n\nWorld." in initial_user
    retry_user = json.loads(http.requests[2].body)["messages"][1]["content"]
    assert "does not satisfy the output rules" in retry_user
    assert "source language (cjk script)" in retry_user
    assert "Source text:\n你好。\n\n世界。" in retry_user
    assert "Current draft:\nHello.\n\nWorld." in retry_user
    assert "failed validation" not in stderr.getvalue()


def test_run_pipeline_cache_hit(tmp_path: Path) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    chunks = with_usage(stream_chunks("Hello."), USAGE)

    def run() -> tuple[int, str, int]:
        console, stderr = make_console()
        http = FakeHttp([FakeStreamResponse(chunks)])
        run_context = make_context(
            console, http.open, cache_directory=str(cache_directory), use_cache=True
        )
        stdout = io.StringIO()
        code = run_pipeline(run_context, (chunk_pass(),), "你好。", 0.0, stdout).run()
        return code, stdout.getvalue(), len(http.requests)

    first_code, first_output, first_calls = run()
    second_code, second_output, second_calls = run()
    assert (first_code, second_code) == (0, 0)
    assert (first_calls, second_calls) == (1, 0)
    assert first_output == second_output == "Hello.\n"


def test_analysis_fails_fast_when_document_needs_multiple_parts() -> None:
    console, stderr = make_console()
    http = FakeHttp([])
    run_context = make_context(console, http.open, maximum_tokens=40)
    analysis_pass = PassDefinition("prep", "Brief.", "analysis", {}, None, False)
    result = analyze_document(
        run_context, analysis_pass, (), "一。二。三。四。", Usage()
    ).run()
    assert isinstance(result, Err)
    assert "requires analysis in" in describe(result.error)
    assert http.requests == []


def test_ensure_paragraphs_retries_pass_in_paragraph_mode() -> None:
    console, stderr = make_console()
    stubborn = "One.\n\nTwo.\n\nThree."
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks(stubborn), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(stubborn), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(stubborn), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("World."), USAGE)),
        ]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context,
        (chunk_pass(ensure_paragraphs=True),),
        "你好。\n\n世界。",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n"
    logged = stderr.getvalue()
    assert "output has 3 paragraph(s), source has 2" in logged
    assert "re-running the pass with one call per paragraph" in logged
    assert "failed validation 2 time(s)" in logged
    assert len(http.requests) == 5


def test_ensure_paragraphs_best_effort_warns_when_retry_still_differs() -> None:
    console, stderr = make_console()
    stubborn = "One.\n\nTwo.\n\nThree."
    hello = "Hello.\n\nSurprise."
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks(stubborn), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(stubborn), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(stubborn), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(hello), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(hello), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(hello), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("World."), USAGE)),
        ]
    )
    run_context = make_context(console, http.open, best_effort=True)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context,
        (chunk_pass(ensure_paragraphs=True),),
        "你好。\n\n世界。",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nSurprise.\n\nWorld.\n"
    logged = stderr.getvalue()
    assert "still differs (3 vs 2)" in logged
    assert "continuing" in logged
    assert len(http.requests) == 7


def test_strict_chunk_pass_escalates_before_failing_on_exhaustion() -> None:
    console, stderr = make_console()
    stubborn = "One.\n\nTwo.\n\nThree."
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks(stubborn), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(stubborn), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(stubborn), USAGE)),
        ]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context,
        (chunk_pass(ensure_paragraphs=True),),
        "你好。\n\n世界。",
        0.0,
        stdout,
    ).run()
    assert code == 1
    assert stdout.getvalue() == ""
    logged = stderr.getvalue()
    assert "re-running the pass with one call per paragraph" in logged
    assert "failed validation 2 time(s)" in logged
    assert "pass [translate] output still failed validation" in logged
    assert len(http.requests) == 9


def test_unit_validation_repairs_hallucinated_paragraph(tmp_path: Path) -> None:
    console, stderr = make_console()
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    http = FakeHttp(
        [
            FakeStreamResponse(
                with_usage(
                    stream_chunks(
                        "# T.\n\n Shen Yue never expected Qin Yu to say"
                        " that.\n\n She froze for a moment."
                    ),
                    USAGE,
                )
            ),
            FakeStreamResponse(with_usage(stream_chunks("Chapter 4: Ruined."), USAGE)),
        ]
    )
    run_context = make_context(
        console, http.open, cache_directory=str(cache_directory), use_cache=True
    )
    stdout = io.StringIO()
    code = run_pipeline(
        run_context,
        (paragraph_pass(),),
        "第413章 被亲妈祸害的女孩 2",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Chapter 4: Ruined.\n"
    assert len(http.requests) == 2
    first_user = json.loads(http.requests[0].body)["messages"][1]["content"]
    assert "reference only" not in first_user
    retry_user = json.loads(http.requests[1].body)["messages"][1]["content"]
    assert "does not satisfy the output rules" in retry_user
    assert "Shen Yue" in retry_user
    cached = list(cache_directory.glob("*.txt"))
    assert len(cached) == 1
    assert cached[0].read_text(encoding="utf-8") == "Chapter 4: Ruined."


def test_unit_validation_strict_mode_fails_and_skips_cache(tmp_path: Path) -> None:
    console, stderr = make_console()
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    bad = "One.\n\nTwo."
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks(bad), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(bad), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(bad), USAGE)),
        ]
    )
    run_context = make_context(
        console, http.open, cache_directory=str(cache_directory), use_cache=True
    )
    stdout = io.StringIO()
    code = run_pipeline(run_context, (paragraph_pass(),), "你好。", 0.0, stdout).run()
    assert code == 1
    assert stdout.getvalue() == ""
    logged = stderr.getvalue()
    assert "failed validation 2 time(s)" in logged
    assert "pass [translate] output still failed validation" in logged
    assert "still failed validation after 2 repair attempt(s)" in logged
    assert len(http.requests) == 3
    assert list(cache_directory.glob("*.txt")) == []


def test_unit_validation_best_effort_warns_and_skips_cache(tmp_path: Path) -> None:
    console, stderr = make_console()
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    bad = "One.\n\nTwo."
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks(bad), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(bad), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(bad), USAGE)),
        ]
    )
    run_context = make_context(
        console,
        http.open,
        cache_directory=str(cache_directory),
        use_cache=True,
        best_effort=True,
    )
    stdout = io.StringIO()
    code = run_pipeline(run_context, (paragraph_pass(),), "你好。", 0.0, stdout).run()
    assert code == 0
    assert stdout.getvalue() == "One.\n\nTwo.\n"
    logged = stderr.getvalue()
    assert "failed validation 2 time(s)" in logged
    assert "uncached" in logged
    assert len(http.requests) == 3
    assert list(cache_directory.glob("*.txt")) == []


def test_strict_mode_fails_the_run_on_empty_replies() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks(""), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(""), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(""), USAGE)),
        ]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    merged = PassDefinition("translate", "T.", "paragraph", {}, None, False, True)
    code = run_pipeline(run_context, (merged,), "你好。", 0.0, stdout).run()
    assert code == 1
    assert stdout.getvalue() == ""
    logged = stderr.getvalue()
    assert "the reply was empty" in logged
    assert "pass [translate] output still failed validation" in logged
    assert "still failed validation after 2 repair attempt(s)" in logged
    assert len(http.requests) == 3


def test_unit_validation_repairs_dropped_chunk_paragraph() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Hello.\n\nWorld."), USAGE)),
        ]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context,
        (chunk_pass(ensure_paragraphs=True),),
        "你好。\n\n世界。",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n"
    assert len(http.requests) == 2
    retry_user = json.loads(http.requests[1].body)["messages"][1]["content"]
    assert "has 1 paragraph(s) but the source has 2" in retry_user


def test_paragraph_check_disabled_without_ensure_paragraphs() -> None:
    console, stderr = make_console()
    http = FakeHttp([FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE))])
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context, (chunk_pass(),), "你好。\n\n世界。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n"
    assert len(http.requests) == 1
    assert "paragraph(s), source has" not in stderr.getvalue()


def test_ensure_paragraphs_tolerance_accepts_small_drift() -> None:
    console, stderr = make_console()
    drifting = "One.\n\nTwo.\n\nThree."
    http = FakeHttp([FakeStreamResponse(with_usage(stream_chunks(drifting), USAGE))])
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context, (chunk_pass(ensure_paragraphs=1),), "你好。\n\n世界。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == drifting + "\n"
    assert len(http.requests) == 1
    assert "output has" not in stderr.getvalue()


def test_ensure_paragraphs_tolerance_exceeded_retries_in_paragraph_mode() -> None:
    console, stderr = make_console()
    wild = "One.\n\nTwo.\n\nThree.\n\nFour."
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks(wild), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(wild), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(wild), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("World."), USAGE)),
        ]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context, (chunk_pass(ensure_paragraphs=1),), "你好。\n\n世界。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n"
    logged = stderr.getvalue()
    assert "output has 4 paragraph(s), source has 2 (allowed ±1)" in logged
    assert "re-running the pass with one call per paragraph" in logged
    assert len(http.requests) == 5


def test_paragraph_mode_stays_strict_despite_tolerance() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("One.\n\nTwo."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("One."), USAGE)),
        ]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    merged = PassDefinition("translate", "T.", "paragraph", {}, None, False, 2)
    code = run_pipeline(run_context, (merged,), "你好。", 0.0, stdout).run()
    assert code == 0
    assert stdout.getvalue() == "One.\n"
    assert len(http.requests) == 2
    retry_user = json.loads(http.requests[1].body)["messages"][1]["content"]
    assert "has 2 paragraph(s) but the source has 1" in retry_user
    assert "(allowed" not in retry_user


def test_unit_validation_rejects_implausible_length() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("word " * 40), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Okay."), USAGE)),
        ]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(run_context, (paragraph_pass(),), "嗯。", 0.0, stdout).run()
    assert code == 0
    assert stdout.getvalue() == "Okay.\n"
    assert len(http.requests) == 2
    retry_user = json.loads(http.requests[1].body)["messages"][1]["content"]
    assert "tokens against a source" in retry_user


def test_paragraph_mode_supplies_neighbour_context() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("One."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Two."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Three."), USAGE)),
        ]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context, (paragraph_pass(),), "一。\n\n二。\n\n三。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == "One.\n\nTwo.\n\nThree.\n"
    users = [
        json.loads(request.body)["messages"][1]["content"] for request in http.requests
    ]
    assert users[0].count("reference only") == 1
    assert "二。" in users[0]
    assert "三。" not in users[0]
    assert users[1].count("reference only") == 1
    assert "一。" in users[1]
    assert "三。" in users[1]
    assert users[2].count("reference only") == 1
    assert "二。" in users[2]
    assert "一。" not in users[2]


def test_chunk_planning_splits_paragraphs_to_request_budget() -> None:
    console, stderr = make_console()
    paragraph = "a" * 600
    text = paragraph + "\n\n" + paragraph
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("One."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Two."), USAGE)),
        ]
    )
    run_context = make_context(console, http.open, maximum_tokens=200)
    stdout = io.StringIO()
    code = run_pipeline(run_context, (chunk_pass(),), text, 0.0, stdout).run()
    assert code == 0
    assert stdout.getvalue() == "One.\n\nTwo.\n"
    assert len(http.requests) == 2
    assert json.loads(http.requests[0].body)["messages"][1]["content"] == paragraph
    assert json.loads(http.requests[1].body)["messages"][1]["content"] == paragraph
    assert "budget" not in stderr.getvalue()


def test_chunk_planning_fails_fast_on_oversized_paragraph() -> None:
    console, stderr = make_console()
    http = FakeHttp([])
    run_context = make_context(console, http.open, maximum_tokens=200)
    code = run_pipeline(
        run_context, (chunk_pass(),), "a" * 1200, 0.0, io.StringIO()
    ).run()
    assert code == 1
    assert http.requests == []
    logged = stderr.getvalue()
    assert "pass [translate] failed" in logged
    assert "over the 200-token budget" in logged


def test_later_pass_with_draft_splits_to_request_budget() -> None:
    console, stderr = make_console()
    paragraph = "a" * 400
    text = paragraph + "\n\n" + paragraph
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("One.\n\nTwo."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("One."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Two."), USAGE)),
        ]
    )
    run_context = make_context(console, http.open, maximum_tokens=400)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context, (chunk_pass(), chunk_pass("polish")), text, 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == "One.\n\nTwo.\n"
    assert len(http.requests) == 3
    assert "budget" not in stderr.getvalue()


def glossary_pairs(document: Any) -> list[tuple[str, str | None]]:
    return [(item["original"], item.get("translated")) for item in document]


def test_run_pipeline_updates_glossary_mid_run(tmp_path: Path) -> None:
    glossary = tmp_path / "glossary.json"
    glossary.write_text(
        '[{"original": "南京", "translated": "Nanjing"}]', encoding="utf-8"
    )
    extraction = FakeStreamResponse(
        with_usage(
            stream_chunks(
                "```json\n"
                + json.dumps(
                    [
                        {"original": "南京", "translated": "Nanking"},
                        {
                            "original": "大观园",
                            "translated": "Grand View Garden",
                        },
                    ],
                    ensure_ascii=False,
                )
                + "\n```"
            ),
            USAGE,
        )
    )
    http = FakeHttp(
        [
            extraction,
            FakeStreamResponse(with_usage(stream_chunks("Brief."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE)),
        ]
    )
    console, stderr = make_console()
    run_context = make_context(
        console,
        http.open,
        glossary_entries=(
            GlossaryEntry("南京", MappingProxyType({"translated": "Nanjing"})),
        ),
        glossary_update_path=str(glossary),
    )
    stdout = io.StringIO()
    code = run_pipeline(
        run_context,
        (
            update_pass(target_language="Chinese"),
            PassDefinition("prep", "Brief.", "analysis", {}, None, False),
            chunk_pass(),
            paragraph_pass("review"),
        ),
        "你好。",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n"

    document = json.loads(glossary.read_text(encoding="utf-8"))
    assert glossary_pairs(document) == [
        ("南京", "Nanjing"),
        ("大观园", "Grand View Garden"),
    ]

    extraction_payload = json.loads(http.requests[0].body)
    extraction_user = extraction_payload["messages"][1]["content"]
    assert "Target language: Chinese" in extraction_user
    assert "Already decided" in extraction_user
    assert '"original": "南京"' in extraction_user
    assert "你好。" in extraction_user

    for request in http.requests[1:]:
        system = json.loads(request.body)["messages"][0]["content"]
        assert "Glossary (mandatory)" in system
        assert '- 南京 = "Nanjing"' in system
        assert '- 大观园 = "Grand View Garden"' in system

    total_line = next(
        line for line in stderr.getvalue().splitlines() if line.startswith("TOTAL")
    )
    assert "prompt=20" in total_line
    assert "completion=24" in total_line


def test_update_glossary_cached_reply_still_merges_and_writes(
    tmp_path: Path,
) -> None:
    glossary = tmp_path / "glossary.json"
    glossary.write_text("[]", encoding="utf-8")
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    extraction = FakeStreamResponse(
        with_usage(
            stream_chunks(
                '[{"original": "大观园", "translated": "Grand View Garden"}]'
            ),
            USAGE,
        )
    )

    def run() -> tuple[int, str]:
        console, _stderr = make_console()
        http = FakeHttp([extraction])
        stdout = io.StringIO()
        code = run_pipeline(
            make_context(
                console,
                http.open,
                cache_directory=str(cache_directory),
                use_cache=True,
                glossary_update_path=str(glossary),
            ),
            (update_pass(),),
            "你好。",
            0.0,
            stdout,
        ).run()
        return code, stdout.getvalue()

    first_code, first_output = run()
    second_code, second_output = run()
    assert (first_code, second_code) == (0, 0)
    assert first_output == second_output == "你好。\n"
    document = json.loads(glossary.read_text(encoding="utf-8"))
    assert glossary_pairs(document) == [
        ("大观园", "Grand View Garden"),
    ]


def test_update_glossary_cache_respects_effective_target_language(
    tmp_path: Path,
) -> None:
    glossary = tmp_path / "glossary.json"
    glossary.write_text("[]", encoding="utf-8")
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    extraction = FakeStreamResponse(
        with_usage(
            stream_chunks(
                '[{"original": "大观园", "translated": "Grand View Garden"}]'
            ),
            USAGE,
        )
    )

    def run(target_language: str) -> tuple[FakeHttp, int]:
        console, _stderr = make_console()
        http = FakeHttp([extraction])
        stdout = io.StringIO()
        code = run_pipeline(
            make_context(
                console,
                http.open,
                cache_directory=str(cache_directory),
                use_cache=True,
                glossary_update_path=str(glossary),
                target_language=target_language,
            ),
            (update_pass(),),
            "你好。",
            0.0,
            stdout,
        ).run()
        return http, code

    first_http, first_code = run("English")
    second_http, second_code = run("French")
    assert (first_code, second_code) == (0, 0)
    assert len(first_http.requests) == 1
    assert len(second_http.requests) == 1
    second_payload = json.loads(second_http.requests[0].body)
    assert "Target language: French" in second_payload["messages"][1]["content"]


def test_update_glossary_pass_without_glossary_path_fails_the_run() -> None:
    console, stderr = make_console()
    http = FakeHttp([])
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(run_context, (update_pass(),), "你好。", 0.0, stdout).run()
    assert code == 1
    assert http.requests == []
    assert (
        "an update_glossary pass requires at least one --glossary file on the "
        "command line" in stderr.getvalue()
    )
