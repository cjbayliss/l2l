from __future__ import annotations


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"

    return "%dm%ds" % (int(seconds // 60), int(seconds % 60))


def usage_line(
    label: str, elapsed: float, prompt_tokens: int, completion_tokens: int, cost: float
) -> str:
    return "%s: %s, prompt=%d, completion=%d, %.1f tok/s, cost=$%.6f" % (
        label,
        fmt_duration(elapsed),
        prompt_tokens,
        completion_tokens,
        completion_tokens / elapsed if elapsed > 0 else 0.0,
        cost,
    )


def stage_done_line(
    elapsed: float, prompt_tokens: int, completion_tokens: int, cost: float
) -> str:
    if prompt_tokens != 0 or completion_tokens != 0 or cost != 0.0:
        return usage_line("Done", elapsed, prompt_tokens, completion_tokens, cost)

    return "Done."


def analysis_info_message(pass_name: str, characters: int, tokens: int) -> str:
    return "l2l: [%s] whole-document analysis (%d characters, ~%d tokens)" % (
        pass_name,
        characters,
        tokens,
    )


def unit_failed_validation_final(
    pass_name: str, index: int, total: int, attempts: int, problem: str
) -> str:
    return (
        "l2l: [%s] unit %d/%d failed validation %d time(s); "
        "last problem: %s. Keeping the last reply, uncached"
        % (pass_name, index, total, attempts, problem)
    )


def unit_failed_validation_attempt(
    pass_name: str, index: int, total: int, problem: str, attempt: int, attempts: int
) -> str:
    return "l2l: [%s] unit %d/%d failed validation (%s); repair attempt %d/%d" % (
        pass_name,
        index,
        total,
        problem,
        attempt,
        attempts,
    )


def unit_no_source_message(pass_name: str, index: int, total: int) -> str:
    return (
        "l2l: [%s] unit %d/%d has no matching source; "
        "passing it through unchanged" % (pass_name, index, total)
    )


def unit_done_message(pass_name: str, index: int, total: int) -> str:
    return "l2l: [%s] unit %d/%d done" % (pass_name, index, total)


def unit_cache_hit_message(pass_name: str, index: int, total: int) -> str:
    return "l2l: [%s] unit %d/%d cache hit" % (pass_name, index, total)


def pass_cache_hit_message(pass_name: str) -> str:
    return f"l2l: [{pass_name}] cache hit"


def pass_started_message(number: int, total: int, pass_name: str) -> str:
    return "Starting pass %d/%d [%s]..." % (number, total, pass_name)


def paragraph_mismatch_message(
    pass_name: str, count: int, source: int, tolerance: int, action: str
) -> str:
    allowed = "" if tolerance <= 0 else " (allowed ±%d)" % tolerance
    return "l2l: [%s] output has %d paragraph(s), source has %d%s; %s" % (
        pass_name,
        count,
        source,
        allowed,
        action,
    )


def paragraph_still_differs_message(pass_name: str, count: int, source: int) -> str:
    return "l2l: [%s] paragraph count still differs (%d vs %d); continuing" % (
        pass_name,
        count,
        source,
    )


def ascii_cache_hit_message() -> str:
    return "l2l: ascii: cache hit"


def ascii_retry_message(attempt: int, attempts: int) -> str:
    return "l2l: ascii: attempt %d/%d still non-ASCII; retrying" % (attempt, attempts)


def ascii_mechanical_message(index: int, total: int) -> str:
    return "l2l: ascii: paragraph %d/%d converted mechanically" % (
        index + 1,
        total,
    )


def ascii_llm_message(index: int, total: int) -> str:
    return (
        "l2l: ascii: paragraph %d/%d still non-ASCII; asking the "
        "LLM to repair it" % (index + 1, total)
    )


def retranslate_info_message(pass_name: str, flagged: int, total: int) -> str:
    return "l2l: [%s] %d of %d paragraph(s) look untranslated; retranslating them" % (
        pass_name,
        flagged,
        total,
    )


def retranslate_skipped_message(pass_name: str, found: int, source: int) -> str:
    return (
        "l2l: [%s] output has %d paragraph(s) but the source has %d; "
        "skipping the untranslated-paragraph check" % (pass_name, found, source)
    )


def retranslate_unit_message(pass_name: str, index: int, total: int) -> str:
    return "l2l: [%s] retranslating flagged paragraph %d/%d" % (
        pass_name,
        index + 1,
        total,
    )


def done_in_message(elapsed: float) -> str:
    return f"l2l: done in {elapsed:.1f}s"
