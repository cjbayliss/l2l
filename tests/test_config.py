from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from l2l.config import (
    api_settings_from_arguments,
    api_settings_from_environment,
    apply_default_options,
    build_config,
    build_setup,
    document_api_settings,
    document_passes,
    document_target_language,
    load_instruction_text,
    load_setup,
    maximum_tokens_api_setting,
    merged_api_settings,
    parse_options_table,
    parse_pass_table,
    pass_definition_from,
    read_document,
    resolve_config_path,
    resolve_passes,
    string_api_setting,
    timeout_api_setting,
    validate_ascii_target_language,
    validate_document,
)
from l2l.errors import FileError, TranslationError, describe
from l2l.glossary import GlossaryEntry
from l2l.monads import Err, Ok, Result, io_pure
from l2l.report import mask_api_key, setup_report, setup_warnings
from l2l.settings import (
    DEFAULT_API_SETTINGS,
    Arguments,
    Config,
    OptionDefaults,
    PartialApiSettings,
    PassDefinition,
    Setup,
    effective_target_language,
    pass_salt,
    resolve_call_settings,
    retranslation_salt,
)
from l2l.text import CACHE_SALT_VERSION, cache_key


def ok_document(document: dict[str, Any]) -> Result[dict[str, Any], TranslationError]:
    return Ok(document)


def test_api_setting_parsers() -> None:
    base = PartialApiSettings()
    invalid_model = string_api_setting(
        "f", base, {"base_url": " http://x ", "model": 3}
    )
    assert isinstance(invalid_model, Err)
    assert "[api] model" in describe(invalid_model.error)

    parsed = string_api_setting("f", base, {"base_url": " http://x ", "api_key": "k"})
    assert isinstance(parsed, Ok)
    assert parsed.value.base_url == "http://x"
    assert parsed.value.api_key == "k"

    for invalid_timeout in (0, True, "5", -3):
        assert isinstance(
            timeout_api_setting("f", base, {"timeout": invalid_timeout}), Err
        )
    parsed_timeout = timeout_api_setting("f", base, {"timeout": 5})
    assert isinstance(parsed_timeout, Ok)
    assert parsed_timeout.value.timeout == 5.0

    for invalid_maximum in (-1, 1.5, 2.5):
        assert isinstance(
            maximum_tokens_api_setting("f", base, {"max_tokens": invalid_maximum}),
            Err,
        )
    parsed_maximum = maximum_tokens_api_setting("f", base, {"max_tokens": 7})
    assert isinstance(parsed_maximum, Ok)
    assert parsed_maximum.value.maximum_tokens == 7


def test_document_api_settings_rejects_invalid_tables() -> None:
    outcomes = (
        ({"api": {"nope": 1}}, "unknown key(s): nope"),
        ({"api": "nope"}, "[api] must be a table"),
        ({"api": {"params": ["x"]}}, "[api] params must be a table"),
        (
            {"api": {"params": {"t": object()}}},
            "[api] params must contain only JSON values",
        ),
    )
    for document, fragment in outcomes:
        result = document_api_settings("f", document)
        assert isinstance(result, Err)
        assert fragment in describe(result.error)


def test_merge_api_settings_params_deep_merge() -> None:
    merged = PartialApiSettings(model="a", parameters={"x": 1, "y": 1}).merge(
        PartialApiSettings(parameters={"y": 2, "z": 3}, model="b")
    )
    assert merged == PartialApiSettings(model="b", parameters={"x": 1, "y": 2, "z": 3})


def test_api_settings_from_environment_and_arguments() -> None:
    invalid_environment = (
        ({"TRANSLATE_MODEL": " m ", "TRANSLATE_TIMEOUT": "bogus"}, "TRANSLATE_TIMEOUT"),
        ({"TRANSLATE_MAX_TOKENS": "bogus"}, "TRANSLATE_MAX_TOKENS"),
        ({"TRANSLATE_TIMEOUT": "-1"}, "TRANSLATE_TIMEOUT must be positive"),
    )
    for environment, fragment in invalid_environment:
        result = api_settings_from_environment(environment)
        assert isinstance(result, Err)
        assert fragment in describe(result.error)

    parsed_environment = api_settings_from_environment(
        {"TRANSLATE_BASE_URL": "http://e", "TRANSLATE_MAX_TOKENS": "9"}
    )
    assert isinstance(parsed_environment, Ok)
    assert parsed_environment.value.base_url == "http://e"
    assert parsed_environment.value.maximum_tokens == 9

    arguments = Arguments(
        config=None,
        base_url=None,
        api_key="k",
        model=None,
        timeout=2.0,
        maximum_tokens=None,
        no_cache=False,
        ensure_paragraphs=False,
        verbose=False,
        print_log_path=False,
        cache_directory=None,
    )
    parsed_arguments = api_settings_from_arguments(arguments)
    assert isinstance(parsed_arguments, Ok)
    assert parsed_arguments.value == PartialApiSettings(api_key="k", timeout=2.0)


def arguments_with_numeric_flags(
    timeout: float | None, maximum_tokens: int | None
) -> Arguments:
    return Arguments(
        config=None,
        base_url=None,
        api_key=None,
        model=None,
        timeout=timeout,
        maximum_tokens=maximum_tokens,
        no_cache=False,
        ensure_paragraphs=False,
        verbose=False,
        print_log_path=False,
        cache_directory=None,
    )


def test_numeric_flags_are_validated() -> None:
    empty: Result[dict[str, Any], TranslationError] = Ok({})
    for arguments, fragment in (
        (arguments_with_numeric_flags(-5.0, None), "--timeout must be positive"),
        (arguments_with_numeric_flags(None, 0), "--maximum-tokens must be positive"),
    ):
        direct = api_settings_from_arguments(arguments)
        assert isinstance(direct, Err)
        assert fragment in describe(direct.error)
        merged = merged_api_settings(arguments, {}, "user", empty, None, empty)
        assert isinstance(merged, Err)
        assert fragment in describe(merged.error)


def test_merged_api_settings_precedence() -> None:
    arguments = Arguments(
        config=None,
        base_url=None,
        api_key=None,
        model="arg-model",
        timeout=None,
        maximum_tokens=None,
        no_cache=False,
        ensure_paragraphs=False,
        verbose=False,
        print_log_path=False,
        cache_directory=None,
    )
    environment = {
        "TRANSLATE_BASE_URL": "http://env",
        "TRANSLATE_TIMEOUT": "5",
        "TRANSLATE_API_KEY": "env-key",
    }
    user = ok_document({"api": {"timeout": 30, "params": {"a": 1}}})
    selected: Result[dict[str, Any], TranslationError] = Ok({})
    result = merged_api_settings(arguments, environment, "user", user, None, selected)
    assert isinstance(result, Ok)
    config = result.value
    assert config.base_url == "http://env"
    assert config.api_key == "env-key"
    assert config.model == "arg-model"
    assert config.timeout == 5.0
    assert config.parameters["a"] == 1


def test_build_config_missing() -> None:
    result = build_config(DEFAULT_API_SETTINGS)
    assert isinstance(result, Err)
    assert "missing required API settings" in describe(result.error)


def test_validate_document() -> None:
    assert isinstance(validate_document("f", {}), Ok)
    assert isinstance(validate_document("f", {"glossary": ["a.json"]}), Ok)
    assert isinstance(validate_document("f", {"target_language": "English"}), Ok)
    result = validate_document("f", {"weird": 1})
    assert isinstance(result, Err)
    assert "weird" in describe(result.error)
    for document in (
        {"glossary": "a.json"},
        {"glossary": [1]},
        {"glossary": [""]},
    ):
        bad_glossary = validate_document("f", document)
        assert isinstance(bad_glossary, Err)
        assert "glossary must be a list of file paths" in describe(bad_glossary.error)


def test_parse_options_table() -> None:
    assert parse_options_table("f", {"ascii": True}) == Ok(OptionDefaults(ascii=True))
    assert parse_options_table("f", {"ensure_paragraphs": True}) == Ok(
        OptionDefaults(ensure_paragraphs=True)
    )
    assert parse_options_table("f", {"ensure_paragraphs": 2}) == Ok(
        OptionDefaults(ensure_paragraphs=2)
    )
    assert parse_options_table("f", {"retranslate_untranslated": True}) == Ok(
        OptionDefaults(retranslate_untranslated=True)
    )
    assert isinstance(parse_options_table("f", {"ascii": "yes"}), Err)
    for bad in ("yes", 0, -1, 1.5, None):
        result = parse_options_table("f", {"ensure_paragraphs": bad})
        assert isinstance(result, Err)
        assert (
            "ensure_paragraphs must be true, false, or a positive integer"
            in describe(result.error)
        )
        result = parse_options_table("f", {"retranslate_untranslated": bad})
        assert isinstance(result, Err)
        assert "retranslate_untranslated must be true or false" in describe(
            result.error
        )
    assert isinstance(parse_options_table("f", {"other": 1}), Err)
    assert isinstance(parse_options_table("f", "x"), Err)


def test_pass_definition_from() -> None:
    result = pass_definition_from("f", "p", {"mode": "weird"}, "inst")
    assert isinstance(result, Err)
    assert (
        "f: [[pass]] p: mode must be analysis, chunk, paragraph, or "
        "update_glossary" in describe(result.error)
    )

    result = pass_definition_from("f", "p", {"mode": "update_glossary"}, "inst")
    assert isinstance(result, Ok)
    assert result.value.mode == "update_glossary"
    assert result.value.target_language is None

    result = pass_definition_from(
        "f", "p", {"mode": "update_glossary", "target_language": "Chinese"}, "inst"
    )
    assert isinstance(result, Ok)
    assert result.value.target_language == "Chinese"

    for bad_target_language in ("", "  ", 3, True):
        result = pass_definition_from(
            "f",
            "p",
            {"mode": "update_glossary", "target_language": bad_target_language},
            "inst",
        )
        assert isinstance(result, Err)
        assert "target_language must be a non-empty string" in describe(result.error)

    result = parse_pass_table(
        "f",
        {
            "name": "p",
            "mode": "chunk",
            "target_language": "French",
            "instruction": "inst",
        },
        ".",
    ).run()
    assert isinstance(result, Ok)
    assert result.value.target_language == "French"

    result = pass_definition_from(
        "f", "p", {"mode": "chunk", "params": {"t": 1}}, "inst"
    )
    assert isinstance(result, Ok)
    assert result.value.mode == "chunk"
    assert result.value.model is None
    assert result.value.ascii is None
    assert result.value.ensure_paragraphs is None
    assert result.value.retranslate_untranslated is None

    result = pass_definition_from(
        "f", "p", {"mode": "chunk", "retranslate_untranslated": True}, "inst"
    )
    assert isinstance(result, Ok)
    assert result.value.retranslate_untranslated is True

    for bad in ("yes", 0, -1, 1.5):
        result = pass_definition_from(
            "f", "p", {"mode": "chunk", "retranslate_untranslated": bad}, "inst"
        )
        assert isinstance(result, Err)
        assert "retranslate_untranslated must be true or false" in describe(
            result.error
        )

    result = pass_definition_from(
        "f", "p", {"mode": "chunk", "ensure_paragraphs": True}, "inst"
    )
    assert isinstance(result, Ok)
    assert result.value.ensure_paragraphs is True

    result = pass_definition_from(
        "f", "p", {"mode": "chunk", "ensure_paragraphs": 3}, "inst"
    )
    assert isinstance(result, Ok)
    assert result.value.ensure_paragraphs == 3

    for bad in ("yes", 0, -1, 1.5):
        result = pass_definition_from(
            "f", "p", {"mode": "chunk", "ensure_paragraphs": bad}, "inst"
        )
        assert isinstance(result, Err)
        assert (
            "ensure_paragraphs must be true, false, or a positive integer"
            in describe(result.error)
        )

    result = parse_pass_table(
        "f", {"name": "p", "mode": "chunk", "strict_fidelity": True}, "."
    ).run()
    assert isinstance(result, Err)
    assert "unknown key(s): strict_fidelity" in describe(result.error)


def test_apply_default_options() -> None:
    passes = (
        PassDefinition("a", "i", "chunk", {}, None, None, None, None),
        PassDefinition("b", "i", "chunk", {}, None, True, False, True),
    )
    applied = apply_default_options(
        passes,
        OptionDefaults(
            ascii=True, ensure_paragraphs=True, retranslate_untranslated=True
        ),
        False,
    )
    assert [
        (
            pass_definition.ascii,
            pass_definition.ensure_paragraphs,
            pass_definition.retranslate_untranslated,
        )
        for pass_definition in applied
    ] == [
        (True, True, True),
        (True, False, True),
    ]

    applied = apply_default_options(
        passes,
        OptionDefaults(),
        True,
    )
    assert [
        (
            pass_definition.ascii,
            pass_definition.ensure_paragraphs,
            pass_definition.retranslate_untranslated,
        )
        for pass_definition in applied
    ] == [
        (False, True, False),
        (True, False, True),
    ]


def test_build_setup_resolves_ensure_paragraphs_precedence() -> None:
    config = Config("u", "k", "m", 1.0, 100, {})

    def setup_with(options: OptionDefaults, flag: bool) -> Setup:
        resolved: Result[tuple[OptionDefaults, tuple[PassDefinition, ...]], Any] = Ok(
            (
                options,
                (
                    PassDefinition("unset", "i", "chunk", {}, None, None, None),
                    PassDefinition("explicit", "i", "chunk", {}, None, None, False),
                    PassDefinition("forced", "i", "chunk", {}, None, None, True),
                ),
            )
        )
        result = build_setup(config, resolved, "English", flag)
        assert isinstance(result, Ok)
        return result.value

    setup = setup_with(OptionDefaults(), True)
    assert setup.ensure_paragraphs
    assert [pass_definition.ensure_paragraphs for pass_definition in setup.passes] == [
        True,
        False,
        True,
    ]

    setup = setup_with(OptionDefaults(), False)
    assert not setup.ensure_paragraphs
    assert [pass_definition.ensure_paragraphs for pass_definition in setup.passes] == [
        False,
        False,
        True,
    ]

    setup = setup_with(OptionDefaults(ensure_paragraphs=2), False)
    assert setup.ensure_paragraphs == 2
    assert [pass_definition.ensure_paragraphs for pass_definition in setup.passes] == [
        2,
        False,
        True,
    ]

    setup = setup_with(OptionDefaults(ensure_paragraphs=2), True)
    assert setup.ensure_paragraphs == 2
    assert [pass_definition.ensure_paragraphs for pass_definition in setup.passes] == [
        2,
        False,
        True,
    ]


def test_resolve_call_settings() -> None:
    config = Config("u", "k", "default-model", 1.0, 100, {"a": 1})
    pass_definition = PassDefinition("p", "i", "chunk", {"b": 2}, "pass-model", False)
    model, parameters = resolve_call_settings(config, pass_definition)
    assert model == "pass-model"
    assert parameters == {"a": 1, "b": 2}


def test_pass_salt_includes_glossary_and_effective_language() -> None:
    pass_definition = PassDefinition("p", "i", "chunk", {}, None, False)
    assert pass_salt(pass_definition) == (CACHE_SALT_VERSION + "\x00p\x00i\x00\x00[]\n")
    entries = (GlossaryEntry("南京", MappingProxyType({"translated": "Nanjing"})),)
    salted = pass_salt(pass_definition, entries)
    assert salted != pass_salt(pass_definition)
    assert '"original": "南京"' in salted
    assert '"translated": "Nanjing"' in salted
    assert pass_salt(pass_definition, entries) == salted

    glossary_pass = PassDefinition("p", "", "update_glossary", {}, None, False)
    assert pass_salt(glossary_pass, entries) == pass_salt(glossary_pass)
    assert pass_salt(glossary_pass, effective_language="Chinese") == pass_salt(
        replace(glossary_pass, target_language="Chinese"),
        effective_language="Chinese",
    )


def test_pass_and_retranslation_salts_normalize_language() -> None:
    for mode in ("analysis", "chunk", "paragraph", "update_glossary"):
        pass_definition = PassDefinition("p", "i", mode, {}, None, False)
        assert pass_salt(pass_definition, effective_language="Chinese") != pass_salt(
            pass_definition, effective_language="French"
        )
    pass_definition = PassDefinition("p", "i", "chunk", {}, None, False)
    assert pass_salt(pass_definition, effective_language="Chinese") == pass_salt(
        pass_definition, effective_language="chinese"
    )
    assert pass_salt(pass_definition, effective_language="Chinese") == pass_salt(
        pass_definition, effective_language="  CHINESE  "
    )
    assert retranslation_salt(pass_definition) == (
        pass_salt(pass_definition) + "\x00retranslate"
    )
    assert retranslation_salt(
        pass_definition, effective_language="Chinese"
    ) != retranslation_salt(pass_definition, effective_language="English")


def test_effective_target_language_prefers_the_pass_override() -> None:
    pass_definition = PassDefinition(
        "p", "i", "chunk", {}, None, False, target_language="Chinese"
    )
    assert effective_target_language("English", pass_definition) == "Chinese"
    assert (
        effective_target_language(
            "English", PassDefinition("p", "i", "chunk", {}, None, False)
        )
        == "English"
    )


def test_cache_key_variants() -> None:
    base = cache_key("text", "model")
    assert base == cache_key("text", "model")
    assert base != cache_key("text", "model", "salt")
    assert base != cache_key("text", "model", "", "work")
    assert base != cache_key("text", "model", "", "", {"temperature": 1})
    assert base != cache_key("text", "model", analysis="brief")
    assert cache_key("text", "model", analysis="brief") == cache_key(
        "text", "model", analysis="brief"
    )


def test_parse_pass_table_and_inline_instruction() -> None:
    parsed = parse_pass_table("f", {"name": "p", "instruction": " do "}, ".").run()
    assert isinstance(parsed, Ok)
    assert parsed.value.instruction == "do"
    missing = parse_pass_table("f", {"name": "p"}, ".").run()
    assert isinstance(missing, Err)
    assert "exactly one of" in describe(missing.error)

    inline = load_instruction_text("f", "p", {"instruction": " inst "}, ".").run()
    assert isinstance(inline, Ok)
    assert inline.value == "inst"
    conflicting = load_instruction_text(
        "f", "p", {"instruction": "", "instruction_file": "x"}, "."
    ).run()
    assert isinstance(conflicting, Err)
    for instruction in ("   ", "", 3, None):
        result = load_instruction_text(
            "f", "p", {"instruction": instruction}, "."
        ).run()
        assert isinstance(result, Err)
        assert "instruction is empty" in describe(result.error)


def test_parse_pass_table_rejects_non_table_and_empty_name() -> None:
    non_table = parse_pass_table("f", "chunk", ".").run()
    assert isinstance(non_table, Err)
    assert "entries must be tables" in describe(non_table.error)
    empty_name = parse_pass_table("f", {"name": "  ", "instruction": "i"}, ".").run()
    assert isinstance(empty_name, Err)
    assert "name must be a non-empty string" in describe(empty_name.error)


def test_document_passes_without_tables_yields_no_passes() -> None:
    assert document_passes("f", {"api": {}}).run() == Ok(())


def test_resolve_passes_requires_pass_table() -> None:
    result = resolve_passes("user", ok_document({"api": {}}), None, Ok({})).run()
    assert isinstance(result, Err)
    assert "no [[pass]] tables" in describe(result.error)


def test_resolve_passes_selection_and_options() -> None:
    user = ok_document(
        {
            "api": {},
            "options": {"ascii": True},
            "pass": [{"name": "p", "instruction": "i"}],
        }
    )
    result = resolve_passes("user", user, None, Ok({})).run()
    assert isinstance(result, Ok)
    options, passes = result.value
    assert options == OptionDefaults(ascii=True)
    assert len(passes) == 1


def test_resolve_config_path_requested_and_environment() -> None:
    assert resolve_config_path("a.toml", {}).run() == "a.toml"
    result = resolve_config_path(None, {"TRANSLATE_CONFIG": " b.toml "}).run()
    assert result == "b.toml"


def test_mask_api_key() -> None:
    assert mask_api_key("secret-key") == "secr...ey"
    assert mask_api_key("short") == "***"
    assert mask_api_key("") == "***"


def test_setup_report_lists_api_and_passes() -> None:
    setup = Setup(
        config=Config(
            base_url="http://endpoint.test/v1",
            api_key="secret-key",
            model="m",
            timeout=30.0,
            maximum_tokens=1000,
            parameters={"temperature": 1},
        ),
        passes=(
            PassDefinition(
                "translate", "T.", "chunk", {"reasoning": "high"}, None, True, False
            ),
        ),
        ensure_paragraphs=False,
        target_language="English",
    )
    report = setup_report(setup, True)
    assert "api.base_url: http://endpoint.test/v1" in report
    assert "api.model: m" in report
    assert "api.api_key: secr...ey" in report
    assert '"temperature": 1' in report
    assert (
        "pass 1/1 [translate]: mode=chunk ascii=True ensure_paragraphs=False"
        " retranslate_untranslated=None model=<default> instruction=2 chars"
        " target_language=English scripts=Latin"
        " uses_target_language_placeholder=False" in report
    )
    assert "options.ensure_paragraphs: True" in report


def test_pass_definition_from_rejects_bad_fields() -> None:
    result = pass_definition_from("f", "p", {"mode": "chunk", "ascii": "yes"}, "i")
    assert isinstance(result, Err)
    assert "ascii must be true or false" in describe(result.error)

    result = pass_definition_from("f", "p", {"mode": "chunk", "model": "  "}, "i")
    assert isinstance(result, Err)
    assert "model must be a non-empty string" in describe(result.error)

    result = pass_definition_from("f", "p", {"mode": "chunk", "params": 3}, "i")
    assert isinstance(result, Err)
    assert "params must be a table" in describe(result.error)


def test_document_passes_rejects_bad_entry_lists() -> None:
    for entries in ("nope", [], [{"name": "p", "instruction": "i"}, 3]):
        result = document_passes("f", {"pass": entries}).run()
        assert isinstance(result, Err)
        assert "[[pass]]" in describe(result.error)


def test_load_instruction_text_rejects_bad_file_settings(
    tmp_path: Path,
) -> None:
    result = load_instruction_text(
        "f", "p", {"instruction_file": 3}, str(tmp_path)
    ).run()
    assert isinstance(result, Err)
    assert "instruction_file must be a path" in describe(result.error)

    result = load_instruction_text(
        "f", "p", {"instruction_file": "missing.txt"}, str(tmp_path)
    ).run()
    assert isinstance(result, Err)
    assert isinstance(result.error, FileError)
    assert "could not read f: [[pass]] p: instruction file" in describe(result.error)


def test_read_document_without_a_file_is_empty() -> None:
    result = read_document("f", "config file", exists=False).run()
    assert isinstance(result, Ok)
    assert result.value == {}


def test_resolve_config_path_discovers_local_then_user(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty"
    local = tmp_path / "l2l.toml"
    user_root = tmp_path / "cfg"
    user_path = user_root / "l2l" / "config.toml"

    def exists_none(path: str) -> bool:
        return False

    assert (
        resolve_config_path(
            None, {"XDG_CONFIG_HOME": str(empty_root)}, exists_none
        ).run()
        is None
    )

    def exists_local(path: str) -> bool:
        return path == str(local)

    assert resolve_config_path(
        None, {}, exists_local, io_pure(str(tmp_path))
    ).run() == str(local)

    def exists_user(path: str) -> bool:
        return path == str(user_path)

    assert resolve_config_path(
        None, {"XDG_CONFIG_HOME": str(user_root)}, exists_user
    ).run() == str(user_path)


def test_load_setup_reports_malformed_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.toml"
    config_path.write_text("[api\nbroken", encoding="utf-8")
    arguments = Arguments(
        config=str(config_path),
        base_url=None,
        api_key=None,
        model=None,
        timeout=None,
        maximum_tokens=None,
        no_cache=False,
        ensure_paragraphs=False,
        verbose=False,
        print_log_path=False,
        cache_directory=None,
    )
    result = load_setup(arguments, {}).run()
    assert isinstance(result, Err)
    assert "cannot parse config file" in describe(result.error)


def test_pass_params_must_contain_json_values() -> None:
    result = pass_definition_from("f", "p", {"params": {"t": [object()]}}, "i")
    assert isinstance(result, Err)
    assert "params must contain only JSON values" in describe(result.error)


SELECTED_CONFIG_TEXT = (
    'target_language = "English"\n'
    'glossary = ["selected-glossary.json"]\n'
    "\n[api]\n"
    'base_url = "http://endpoint.test/v1"\n'
    'api_key = "key"\n'
    'model = "m"\n'
    "\n[[pass]]\n"
    'name = "translate"\n'
    'mode = "chunk"\n'
    'instruction = "Translate."\n'
)


def glossary_arguments(config_path: Path) -> Arguments:
    return Arguments(
        config=str(config_path),
        base_url=None,
        api_key=None,
        model=None,
        timeout=None,
        maximum_tokens=None,
        no_cache=False,
        ensure_paragraphs=False,
        verbose=False,
        print_log_path=False,
        cache_directory=None,
    )


def test_document_target_language_validates() -> None:
    assert document_target_language("f", {}) == Ok(None)
    assert document_target_language("f", {"target_language": " French "}) == Ok(
        "French"
    )
    for bad_target_language in ("", "  ", 3, True):
        result = document_target_language("f", {"target_language": bad_target_language})
        assert isinstance(result, Err)
        assert "target_language must be a non-empty string" in describe(result.error)


def test_load_setup_requires_target_language(tmp_path: Path) -> None:
    config_path = tmp_path / "book.toml"
    config_path.write_text(
        SELECTED_CONFIG_TEXT.replace('target_language = "English"\n', ""),
        encoding="utf-8",
    )
    result = load_setup(
        glossary_arguments(config_path),
        {"XDG_CONFIG_HOME": str(tmp_path / "nouser")},
    ).run()
    assert isinstance(result, Err)
    assert (
        "target language is required; pass --target-language or set "
        "target_language in a config file" in describe(result.error)
    )


def test_load_setup_resolves_target_language_precedence(tmp_path: Path) -> None:
    user_home = tmp_path / "user"
    user_config_directory = user_home / "l2l"
    user_config_directory.mkdir(parents=True)
    (user_config_directory / "config.toml").write_text(
        'target_language = "Japanese"\n', encoding="utf-8"
    )

    selected_directory = tmp_path / "selected"
    selected_directory.mkdir()
    selected_config = selected_directory / "book.toml"
    selected_config.write_text(SELECTED_CONFIG_TEXT, encoding="utf-8")
    (selected_directory / "selected-glossary.json").write_text("[]", encoding="utf-8")

    environment = {"XDG_CONFIG_HOME": str(user_home)}
    configured = load_setup(glossary_arguments(selected_config), environment).run()
    assert isinstance(configured, Ok)
    assert configured.value.target_language == "English"

    from_command_line = load_setup(
        replace(
            glossary_arguments(selected_config),
            target_language="German",
        ),
        environment,
    ).run()
    assert isinstance(from_command_line, Ok)
    assert from_command_line.value.target_language == "German"


def test_load_setup_rejects_blank_command_line_target_language(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "book.toml"
    config_path.write_text(SELECTED_CONFIG_TEXT, encoding="utf-8")
    (tmp_path / "selected-glossary.json").write_text("[]", encoding="utf-8")
    result = load_setup(
        replace(glossary_arguments(config_path), target_language="  "),
        {"XDG_CONFIG_HOME": str(tmp_path / "nouser")},
    ).run()
    assert isinstance(result, Err)
    assert "--target-language must be a non-empty string" in describe(result.error)


def test_validate_document_rejects_bad_glossary_values() -> None:
    for document in (
        {"glossary": "a.json"},
        {"glossary": [1]},
        {"glossary": [""]},
    ):
        result = validate_document("f", document)
        assert isinstance(result, Err)
        assert "f: glossary must be a list of file paths" in describe(result.error)


def test_load_setup_resolves_and_merges_glossaries(tmp_path: Path) -> None:
    user_home = tmp_path / "user"
    user_config_directory = user_home / "l2l"
    user_config_directory.mkdir(parents=True)
    (user_config_directory / "config.toml").write_text(
        'glossary = ["user-glossary.json"]\n', encoding="utf-8"
    )
    (user_config_directory / "user-glossary.json").write_text(
        '[{"original": "甲", "translated": "Alpha", "pronouns": "she/her"}]',
        encoding="utf-8",
    )

    selected_directory = tmp_path / "selected"
    selected_directory.mkdir()
    selected_config = selected_directory / "book.toml"
    selected_config.write_text(SELECTED_CONFIG_TEXT, encoding="utf-8")
    (selected_directory / "selected-glossary.json").write_text(
        '[{"original": "甲", "translated": "Aleph"}]', encoding="utf-8"
    )

    cli_glossary = tmp_path / "cli-glossary.json"
    cli_glossary.write_text(
        '[{"original": "乙", "translated": "Bee"}]', encoding="utf-8"
    )

    arguments = replace(
        glossary_arguments(selected_config), glossary=(str(cli_glossary),)
    )
    result = load_setup(arguments, {"XDG_CONFIG_HOME": str(user_home)}).run()
    assert isinstance(result, Ok)
    setup = result.value
    assert setup.glossary_paths == (
        str(user_config_directory / "user-glossary.json"),
        str(selected_directory / "selected-glossary.json"),
        str(cli_glossary),
    )
    assert [entry.original for entry in setup.glossary_entries] == ["甲", "乙"]
    assert tuple(setup.glossary_entries[0].attributes.items()) == (
        ("translated", "Aleph"),
    )
    for pass_definition in setup.passes:
        assert pass_definition.instruction == "Translate."
    assert [entry.original for entry in setup.glossary_entries] == ["甲", "乙"]
    assert tuple(setup.glossary_entries[0].attributes.items()) == (
        ("translated", "Aleph"),
    )


def test_load_setup_without_glossary_keeps_instructions(tmp_path: Path) -> None:
    config_path = tmp_path / "book.toml"
    config_path.write_text(
        SELECTED_CONFIG_TEXT.replace('glossary = ["selected-glossary.json"]\n', ""),
        encoding="utf-8",
    )
    result = load_setup(
        glossary_arguments(config_path),
        {"XDG_CONFIG_HOME": str(tmp_path / "nouser")},
    ).run()
    assert isinstance(result, Ok)
    setup = result.value
    assert setup.glossary_paths == ()
    assert setup.glossary_entries == ()
    assert [pass_definition.instruction for pass_definition in setup.passes] == [
        "Translate."
    ]


def test_load_setup_rejects_invalid_glossary_entries(tmp_path: Path) -> None:
    config_path = tmp_path / "book.toml"
    config_path.write_text(SELECTED_CONFIG_TEXT, encoding="utf-8")
    (tmp_path / "selected-glossary.json").write_text(
        '[{"translated": "x"}]', encoding="utf-8"
    )
    result = load_setup(
        glossary_arguments(config_path),
        {"XDG_CONFIG_HOME": str(tmp_path / "nouser")},
    ).run()
    assert isinstance(result, Err)
    assert "entry 1" in describe(result.error)


def test_load_setup_accepts_update_glossary_pass_with_cli_glossary(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "book.toml"
    config_path.write_text(
        SELECTED_CONFIG_TEXT.replace(
            'mode = "chunk"', 'mode = "update_glossary"'
        ).replace('name = "translate"', 'name = "glossary"'),
        encoding="utf-8",
    )
    (tmp_path / "selected-glossary.json").write_text(
        '[{"original": "甲", "translated": "Aleph"}]', encoding="utf-8"
    )
    cli_glossary = tmp_path / "cli-glossary.json"
    cli_glossary.write_text("[]", encoding="utf-8")
    arguments = replace(glossary_arguments(config_path), glossary=(str(cli_glossary),))
    result = load_setup(arguments, {"XDG_CONFIG_HOME": str(tmp_path / "nouser")}).run()
    assert isinstance(result, Ok)
    setup = result.value
    assert [item.mode for item in setup.passes] == ["update_glossary"]
    assert [item.original for item in setup.glossary_entries] == ["甲"]


def test_load_setup_rejects_update_glossary_pass_without_cli_glossary(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "book.toml"
    config_path.write_text(
        SELECTED_CONFIG_TEXT.replace('mode = "chunk"', 'mode = "update_glossary"'),
        encoding="utf-8",
    )
    (tmp_path / "selected-glossary.json").write_text(
        '[{"original": "甲", "translated": "Aleph"}]', encoding="utf-8"
    )
    result = load_setup(
        glossary_arguments(config_path),
        {"XDG_CONFIG_HOME": str(tmp_path / "nouser")},
    ).run()
    assert isinstance(result, Err)
    assert (
        "an update_glossary pass requires at least one --glossary file on the "
        "command line" in describe(result.error)
    )


def test_setup_report_lists_glossary_when_configured() -> None:
    setup = Setup(
        config=Config(
            base_url="http://endpoint.test/v1",
            api_key="secret-key",
            model="m",
            timeout=30.0,
            maximum_tokens=1000,
            parameters={},
        ),
        passes=(PassDefinition("translate", "T.", "chunk", {}, None, True, False),),
        ensure_paragraphs=False,
        target_language="English",
        glossary_paths=("generic.json", "book.json"),
        glossary_entries=(
            GlossaryEntry("南京", MappingProxyType({"translated": "Nanjing"})),
        ),
    )
    report = setup_report(setup, False)
    assert "glossary: 2 file(s), 1 entries" in report
    assert "  - generic.json" in report
    assert "  - book.json" in report


def test_setup_report_omits_glossary_when_unconfigured() -> None:
    setup = Setup(
        config=Config(
            base_url="http://endpoint.test/v1",
            api_key="secret-key",
            model="m",
            timeout=30.0,
            maximum_tokens=1000,
            parameters={},
        ),
        passes=(PassDefinition("translate", "T.", "chunk", {}, None, True, False),),
        ensure_paragraphs=False,
        target_language="English",
    )
    assert "glossary" not in setup_report(setup, False)


def make_setup_for_warnings(
    target_language: str = "English",
    passes: tuple[PassDefinition, ...] = (),
    glossary_entries: tuple[GlossaryEntry, ...] = (),
) -> Setup:
    return Setup(
        config=Config("u", "k", "m", 1.0, 1000, {}),
        passes=passes,
        ensure_paragraphs=False,
        target_language=target_language,
        glossary_entries=glossary_entries,
    )


def test_validate_ascii_target_language() -> None:
    ascii_pass = (PassDefinition("p", "i", "chunk", {}, None, True),)
    for language in ("English", "Spanish", "English (US)", "Zorblaxian"):
        assert isinstance(
            validate_ascii_target_language(
                make_setup_for_warnings(language, ascii_pass)
            ),
            Ok,
        )

    rejected = validate_ascii_target_language(
        make_setup_for_warnings("Chinese", ascii_pass)
    )
    assert isinstance(rejected, Err)
    message = describe(rejected.error)
    assert "pass 1 [p]" in message
    assert "Chinese" in message
    assert "Han" in message
    assert "ascii = false" in message

    overridden = validate_ascii_target_language(
        make_setup_for_warnings(
            "English",
            (
                PassDefinition(
                    "p", "i", "chunk", {}, None, True, target_language="Japanese"
                ),
            ),
        )
    )
    assert isinstance(overridden, Err)
    assert "Japanese" in describe(overridden.error)


def test_setup_warnings_report_unknown_languages() -> None:
    warnings = setup_warnings(make_setup_for_warnings("Zorblaxian"))
    assert len(warnings) == 1
    assert "not recognized" in warnings[0]
    assert setup_warnings(make_setup_for_warnings("English")) == ()


def test_setup_warnings_report_instructions_without_the_language() -> None:
    without_language = make_setup_for_warnings(
        "English", (PassDefinition("p", "T.", "chunk", {}, None, False),)
    )
    assert any(
        "does not mention the target language" in warning
        for warning in setup_warnings(without_language)
    )

    with_placeholder = make_setup_for_warnings(
        "English",
        (
            PassDefinition(
                "p", "Translate into {target_language}.", "chunk", {}, None, False
            ),
        ),
    )
    assert setup_warnings(with_placeholder) == ()

    named = make_setup_for_warnings(
        "English",
        (PassDefinition("p", "Translate into English.", "chunk", {}, None, False),),
    )
    assert setup_warnings(named) == ()

    glossary_pass = make_setup_for_warnings(
        "English", (PassDefinition("p", "", "update_glossary", {}, None, False),)
    )
    assert setup_warnings(glossary_pass) == ()


def test_setup_warnings_report_glossary_script_mismatches() -> None:
    mismatched = make_setup_for_warnings(
        "English",
        glossary_entries=(
            GlossaryEntry("南京", MappingProxyType({"translated": "南京"})),
        ),
    )
    warnings = setup_warnings(mismatched)
    assert len(warnings) == 1
    assert "glossary entry" in warnings[0]
    assert "another target language" in warnings[0]

    matched = make_setup_for_warnings(
        "English",
        glossary_entries=(
            GlossaryEntry("南京", MappingProxyType({"translated": "Nanjing"})),
        ),
    )
    assert setup_warnings(matched) == ()

    unknown = make_setup_for_warnings(
        "Zorblaxian",
        glossary_entries=(
            GlossaryEntry("南京", MappingProxyType({"translated": "南京"})),
        ),
    )
    assert all("glossary entry" not in warning for warning in setup_warnings(unknown))

    untranslated_entry = make_setup_for_warnings(
        "English",
        glossary_entries=(GlossaryEntry("南京", MappingProxyType({})),),
    )
    assert setup_warnings(untranslated_entry) == ()
