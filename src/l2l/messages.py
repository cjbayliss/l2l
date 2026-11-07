from __future__ import annotations


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"

    return f"{int(seconds // 60)}m{int(seconds % 60)}s"


def usage_line(
    label: str, elapsed: float, prompt_tokens: int, completion_tokens: int, cost: float
) -> str:
    return (
        f"{label}: {format_duration(elapsed)}, prompt={prompt_tokens}, "
        f"completion={completion_tokens}, "
        f"{completion_tokens / elapsed if elapsed > 0 else 0.0:.1f} tok/s, "
        f"cost=${cost:.6f}"
    )


def stage_done_line(
    elapsed: float, prompt_tokens: int, completion_tokens: int, cost: float
) -> str:
    if prompt_tokens != 0 or completion_tokens != 0 or cost != 0.0:
        return usage_line("Done", elapsed, prompt_tokens, completion_tokens, cost)

    return "Done."


def analysis_information_message(pass_name: str, characters: int, tokens: int) -> str:
    return (
        f"l2l: [{pass_name}] whole-document analysis "
        f"({characters} characters, ~{tokens} tokens)"
    )


def glossary_update_information_message(
    pass_name: str, target_language: str, characters: int, tokens: int
) -> str:
    return (
        f"l2l: [{pass_name}] whole-document glossary update for {target_language} "
        f"({characters} characters, ~{tokens} tokens)"
    )


def unit_failed_validation_final(
    pass_name: str, index: int, total: int, attempts: int, problem: str, kept: bool
) -> str:
    kept_note = " Keeping the last reply, uncached" if kept else ""
    return (
        f"l2l: [{pass_name}] unit {index}/{total} failed validation {attempts} "
        f"time(s); last problem: {problem}.{kept_note}"
    )


def unit_failed_validation_attempt(
    pass_name: str, index: int, total: int, problem: str, attempt: int, attempts: int
) -> str:
    return (
        f"l2l: [{pass_name}] unit {index}/{total} failed validation ({problem}); "
        f"repair attempt {attempt}/{attempts}"
    )


def unit_no_source_message(pass_name: str, index: int, total: int) -> str:
    return (
        f"l2l: [{pass_name}] unit {index}/{total} has no matching source; "
        "passing it through unchanged"
    )


def unit_passed_through_message(pass_name: str, index: int, total: int) -> str:
    return (
        f"l2l: [{pass_name}] unit {index}/{total} has no translatable text; "
        "passing it through unchanged"
    )


def unit_done_message(pass_name: str, index: int, total: int) -> str:
    return f"l2l: [{pass_name}] unit {index}/{total} done"


def unit_cache_hit_message(pass_name: str, index: int, total: int) -> str:
    return f"l2l: [{pass_name}] unit {index}/{total} cache hit"


def pass_cache_hit_message(pass_name: str) -> str:
    return f"l2l: [{pass_name}] cache hit"


def pass_started_message(number: int, total: int, pass_name: str) -> str:
    return f"Starting pass {number}/{total} [{pass_name}]..."


def paragraph_mismatch_message(
    pass_name: str, count: int, source: int, tolerance: int, action: str
) -> str:
    allowed = "" if tolerance <= 0 else f" (allowed ±{tolerance})"
    return (
        f"l2l: [{pass_name}] output has {count} paragraph(s), source has {source}"
        f"{allowed}; {action}"
    )


def paragraph_still_differs_message(pass_name: str, count: int, source: int) -> str:
    return (
        f"l2l: [{pass_name}] paragraph count still differs ({count} vs {source}); "
        "continuing"
    )


def ascii_cache_hit_message() -> str:
    return "l2l: ascii: cache hit"


def ascii_retry_message(attempt: int, attempts: int) -> str:
    return f"l2l: ascii: attempt {attempt}/{attempts} still non-ASCII; retrying"


def ascii_mechanical_message(index: int, total: int) -> str:
    return f"l2l: ascii: paragraph {index + 1}/{total} converted mechanically"


def ascii_llm_message(index: int, total: int) -> str:
    return (
        f"l2l: ascii: paragraph {index + 1}/{total} still non-ASCII; asking the "
        "LLM to repair it"
    )


def retranslate_information_message(pass_name: str, flagged: int, total: int) -> str:
    return (
        f"l2l: [{pass_name}] {flagged} of {total} paragraph(s) look untranslated; "
        "retranslating them"
    )


def retranslate_majority_message(pass_name: str, flagged: int, total: int) -> str:
    return (
        f"l2l: [{pass_name}] {flagged} of {total} paragraph(s) look untranslated "
        "(more than half); re-running the pass"
    )


def retranslate_skipped_message(pass_name: str, found: int, source: int) -> str:
    return (
        f"l2l: [{pass_name}] output has {found} paragraph(s) but the source has "
        f"{source}; skipping the untranslated-paragraph check"
    )


def retranslation_target_script_message(
    pass_name: str, index: int, total: int, sample: str
) -> str:
    return (
        f"l2l: [{pass_name}] paragraph {index + 1}/{total} still looks untranslated "
        f"but is already in the target script; keeping {sample!r}"
    )


def untranslated_kept_message(
    pass_name: str, index: int, total: int, sample: str
) -> str:
    return (
        f"l2l: [{pass_name}] paragraph {index + 1}/{total} is still untranslated "
        f"after retranslation: {sample!r}; keeping the last reply"
    )


def untranslated_problem(index: int, sample: str) -> str:
    return (
        f"paragraph {index + 1} is still untranslated after retranslation: {sample!r}"
    )


def retranslation_check_message(
    target_language: str,
    source_script_description: str,
    target_scripts_description: str,
) -> str:
    source = source_script_description or "?"
    if not target_scripts_description:
        return (
            f"l2l: retranslate check: target language {target_language} "
            f"(scripts unknown), source script {source}; using the ASCII-letter rule"
        )

    return (
        f"l2l: retranslate check: target language {target_language} "
        f"(scripts: {target_scripts_description}), source script {source}"
    )


def retranslate_unit_message(pass_name: str, index: int, total: int) -> str:
    return f"l2l: [{pass_name}] retranslating flagged paragraph {index + 1}/{total}"


def chunk_budget_violation_message(
    pass_name: str, index: int, tokens: int, budget: int
) -> str:
    return (
        f"[{pass_name}] paragraph {index + 1} is ~{tokens} source tokens, over the "
        f"~{budget}-token source budget for a single request in this pass; raise "
        "api.max_tokens or shorten the input"
    )


def done_in_message(elapsed: float) -> str:
    return f"l2l: done in {elapsed:.1f}s"
