from __future__ import annotations

from dataclasses import replace
lazy from collections.abc import Mapping

from l2l.config.api import merged_api_settings
from l2l.config.documents import read_setup_document, resolve_config_path
from l2l.config.glossaries import (
    require_glossary_for_update_passes,
    resolve_glossaries,
    setup_with_glossary,
)
from l2l.config.languages import resolve_source_language, resolve_target_language
from l2l.config.models import DocumentResult, Glossaries, ResolvedPasses, SetupResult
from l2l.config.parsing import apply_default_options
from l2l.config.resolution import (
    finalize_passes,
    resolve_filter_settings,
    resolve_passes,
)
from l2l.errors import TranslationError, fail_config
from l2l.languages import expected_target_scripts, scripts_description
from l2l.monads import (
    IO,
    Ok,
    Result,
    io_bind,
    io_map,
    io_pair,
    io_result,
    result_bind,
    result_map,
)
from l2l.settings import (
    Arguments,
    Config,
    PassDefinition,
    Setup,
    effective_target_language,
)


def validate_ascii_target_language(
    setup: Setup,
) -> Result[Setup, TranslationError]:
    conflicting = tuple(
        (
            pass_definition,
            effective_target_language(
                setup.target_language, pass_definition.target_language
            ),
        )
        for pass_definition in setup.passes
        if pass_definition.ascii is True
    )

    def problem(
        indexed: tuple[int, tuple[PassDefinition, str]],
    ) -> str | None:
        position, (pass_definition, language) = indexed
        scripts = expected_target_scripts(language)
        if not scripts or "Latn" in scripts:
            return None

        return (
            f"pass {position} [{pass_definition.name}]: ascii = true but target "
            f"language {language} uses the {scripts_description(scripts)} script; "
            "set ascii = false or choose a Latin-script target"
        )

    problems = tuple(
        message
        for message in map(problem, enumerate(conflicting, 1))
        if message is not None
    )
    if problems:
        return fail_config(problems[0])

    return Ok(setup)


def build_setup(
    config: Config,
    pair: Result[ResolvedPasses, TranslationError],
    target_language: str,
    source_language: str,
) -> Result[Setup, TranslationError]:
    def with_options(resolved: ResolvedPasses) -> Result[Setup, TranslationError]:
        option_defaults = resolved.option_defaults
        effective = option_defaults.ensure_paragraphs
        global_ensure = (
            effective if option_defaults.ensure_paragraphs_explicit else None
        )
        return result_bind(
            finalize_passes(
                apply_default_options(resolved.pass_drafts, option_defaults, effective),
                target_language,
                source_language,
                resolved.global_prompts,
                resolved.global_style_patch,
                resolved.global_output_patch,
                global_ensure,
            ),
            lambda finalized: validate_ascii_target_language(
                Setup(
                    config=config,
                    passes=finalized.passes,
                    ensure_paragraphs=effective,
                    target_language=target_language,
                    source_language=source_language,
                    config_warnings=finalized.warnings,
                )
            ),
        )

    return result_bind(pair, with_options)


def assemble_setup(
    arguments: Arguments,
    environment: Mapping[str, str],
    path: str,
    document_result: DocumentResult,
    resolve_pipeline_passes: bool,
) -> IO[SetupResult]:
    def setup_with(
        pair: tuple[
            Result[ResolvedPasses, TranslationError],
            Result[Glossaries, TranslationError],
        ],
    ) -> SetupResult:
        resolved, glossaries = pair
        return result_bind(
            merged_api_settings(environment, path, document_result),
            lambda config: result_bind(
                resolve_target_language(path, document_result),
                lambda target_language: result_bind(
                    resolve_source_language(path, document_result),
                    lambda source_language: result_bind(
                        build_setup(config, resolved, target_language, source_language),
                        lambda setup: result_bind(
                            glossaries,
                            lambda resolved_glossaries: result_bind(
                                require_glossary_for_update_passes(
                                    arguments.glossary,
                                    setup_with_glossary(setup, resolved_glossaries),
                                ),
                                lambda complete_setup: result_map(
                                    resolve_filter_settings(path, document_result),
                                    lambda filter_settings: replace(
                                        complete_setup, filter=filter_settings
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

    return io_map(
        io_pair(
            resolve_passes(path, document_result)
            if resolve_pipeline_passes
            else io_result(Ok(ResolvedPasses.empty())),
            resolve_glossaries(path, document_result, arguments.glossary),
        ),
        setup_with,
    )


def load_setup(
    arguments: Arguments,
    environment: Mapping[str, str],
    resolve_pipeline_passes: bool = True,
) -> IO[SetupResult]:
    def with_path(path: str) -> IO[SetupResult]:
        requested = isinstance(arguments.config, str) and bool(arguments.config.strip())
        return io_bind(
            read_setup_document(path, requested),
            lambda document_result: assemble_setup(
                arguments,
                environment,
                path,
                document_result,
                resolve_pipeline_passes,
            ),
        )

    return io_bind(resolve_config_path(arguments.config, environment), with_path)
