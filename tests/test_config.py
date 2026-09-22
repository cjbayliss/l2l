from pathlib import Path
from typing import Any

from l2l.config import (
    api_settings_from_arguments,
    api_settings_from_environment,
    apply_default_options,
    build_config,
    build_setup,
    document_api_settings,
    document_passes,
    load_instruction_text,
    load_setup,
    max_tokens_api_setting,
    merge_api_settings,
    merged_api_settings,
    params_api_setting,
    parse_options_table,
    parse_pass_table,
    pass_definition_from,
    read_document,
    resolve_config_path,
    resolve_passes,
    string_api_setting,
    timeout_api_setting,
    validate_document,
)
from l2l.errors import TranslationError, describe
from l2l.monads import Err, Ok, Result
from l2l.plans import mask_api_key, setup_report
from l2l.settings import (
    DEFAULT_API_SETTINGS,
    Arguments,
    Config,
    PartialApiSettings,
    PassDefinition,
    Setup,
    pass_salt,
    resolve_call_settings,
)
from l2l.text import CACHE_SALT_VERSION, cache_key


def ok_document(document: dict[str, Any]) -> Result[dict[str, Any], TranslationError]:
    return Ok(document)


def test_string_api_setting() -> None:
    base = PartialApiSettings()
    result = string_api_setting("f", base, {"base_url": " http://x ", "model": 3})
    assert isinstance(result, Err)
    assert "[api] model" in describe(result.error)

    result = string_api_setting("f", base, {"base_url": " http://x ", "api_key": "k"})
    assert isinstance(result, Ok)
    assert result.value.base_url == "http://x"
    assert result.value.api_key == "k"


def test_timeout_api_setting() -> None:
    base = PartialApiSettings()
    assert isinstance(timeout_api_setting("f", base, {"timeout": 0}), Err)
    assert isinstance(timeout_api_setting("f", base, {"timeout": True}), Err)
    assert isinstance(timeout_api_setting("f", base, {"timeout": "5"}), Err)
    result = timeout_api_setting("f", base, {"timeout": 5})
    assert isinstance(result, Ok)
    assert result.value.timeout == 5.0


def test_max_tokens_api_setting() -> None:
    base = PartialApiSettings()
    assert isinstance(max_tokens_api_setting("f", base, {"max_tokens": -1}), Err)
    assert isinstance(max_tokens_api_setting("f", base, {"max_tokens": 1.5}), Err)
    result = max_tokens_api_setting("f", base, {"max_tokens": 7})
    assert isinstance(result, Ok)
    assert result.value.max_tokens == 7


def test_document_api_settings_unknown_key() -> None:
    result = document_api_settings("f", {"api": {"nope": 1}})
    assert isinstance(result, Err)
    assert "unknown key(s): nope" in describe(result.error)


def test_merge_api_settings_params_deep_merge() -> None:
    merged = merge_api_settings(
        PartialApiSettings(model="a", params={"x": 1, "y": 1}),
        PartialApiSettings(params={"y": 2, "z": 3}, model="b"),
    )
    assert merged == PartialApiSettings(model="b", params={"x": 1, "y": 2, "z": 3})


def test_api_settings_from_environment() -> None:
    result = api_settings_from_environment(
        {"TRANSLATE_MODEL": " m ", "TRANSLATE_TIMEOUT": "bogus"}
    )
    assert isinstance(result, Err)
    assert "TRANSLATE_TIMEOUT" in describe(result.error)

    result = api_settings_from_environment(
        {"TRANSLATE_BASE_URL": "http://e", "TRANSLATE_MAX_TOKENS": "9"}
    )
    assert isinstance(result, Ok)
    assert result.value.base_url == "http://e"
    assert result.value.max_tokens == 9


def test_api_settings_from_arguments() -> None:
    arguments = Arguments(
        config=None,
        base_url=None,
        api_key="k",
        model=None,
        timeout=2.0,
        max_tokens=None,
        no_cache=False,
        ensure_paragraphs=False,
        verbose=False,
        show_log_path=False,
        cache_dir=None,
    )
    assert api_settings_from_arguments(arguments) == PartialApiSettings(
        api_key="k", timeout=2.0
    )


def test_merged_api_settings_precedence() -> None:
    arguments = Arguments(
        config=None,
        base_url=None,
        api_key=None,
        model="arg-model",
        timeout=None,
        max_tokens=None,
        no_cache=False,
        ensure_paragraphs=False,
        verbose=False,
        show_log_path=False,
        cache_dir=None,
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
    assert config.params["a"] == 1


def test_build_config_missing() -> None:
    result = build_config(DEFAULT_API_SETTINGS)
    assert isinstance(result, Err)
    assert "missing required API settings" in describe(result.error)


def test_validate_document() -> None:
    assert isinstance(validate_document("f", {}), Ok)
    result = validate_document("f", {"weird": 1})
    assert isinstance(result, Err)
    assert "weird" in describe(result.error)


def test_parse_options_table() -> None:
    assert parse_options_table("f", {"ascii": True}) == Ok(
        {"ascii": True, "ensure_paragraphs": False}
    )
    assert parse_options_table("f", {"ensure_paragraphs": True}) == Ok(
        {"ascii": False, "ensure_paragraphs": True}
    )
    assert isinstance(parse_options_table("f", {"ascii": "yes"}), Err)
    assert isinstance(parse_options_table("f", {"ensure_paragraphs": "yes"}), Err)
    assert isinstance(parse_options_table("f", {"other": 1}), Err)
    assert isinstance(parse_options_table("f", "x"), Err)


def test_pass_definition_from() -> None:
    result = pass_definition_from("f", "p", {"mode": "weird"}, "inst")
    assert isinstance(result, Err)

    result = pass_definition_from(
        "f", "p", {"mode": "chunk", "params": {"t": 1}}, "inst"
    )
    assert isinstance(result, Ok)
    assert result.value.mode == "chunk"
    assert result.value.model is None
    assert result.value.ascii is None
    assert result.value.ensure_paragraphs is None

    result = pass_definition_from(
        "f", "p", {"mode": "chunk", "ensure_paragraphs": True}, "inst"
    )
    assert isinstance(result, Ok)
    assert result.value.ensure_paragraphs is True

    result = pass_definition_from(
        "f", "p", {"mode": "chunk", "ensure_paragraphs": "yes"}, "inst"
    )
    assert isinstance(result, Err)
    assert "ensure_paragraphs must be true or false" in describe(result.error)

    result = parse_pass_table(
        "f", {"name": "p", "mode": "chunk", "strict_fidelity": True}, "."
    ).run()
    assert isinstance(result, Err)
    assert "unknown key(s): strict_fidelity" in describe(result.error)


def test_apply_default_options() -> None:
    passes = (
        PassDefinition("a", "i", "chunk", {}, None, None, None),
        PassDefinition("b", "i", "chunk", {}, None, True, False),
    )
    applied = apply_default_options(
        passes, {"ascii": True, "ensure_paragraphs": True}, False
    )
    assert [(p.ascii, p.ensure_paragraphs) for p in applied] == [
        (True, True),
        (True, False),
    ]

    applied = apply_default_options(
        passes, {"ascii": False, "ensure_paragraphs": False}, True
    )
    assert [(p.ascii, p.ensure_paragraphs) for p in applied] == [
        (False, True),
        (True, False),
    ]


def test_build_setup_resolves_ensure_paragraphs_precedence() -> None:
    config = Config("u", "k", "m", 1.0, 100, {})

    def setup_with_flag(flag: bool) -> Setup:
        resolved: Result[tuple[dict[str, bool], tuple[PassDefinition, ...]], Any] = Ok(
            (
                {"ascii": False, "ensure_paragraphs": False},
                (
                    PassDefinition("unset", "i", "chunk", {}, None, None, None),
                    PassDefinition("explicit", "i", "chunk", {}, None, None, False),
                    PassDefinition("forced", "i", "chunk", {}, None, None, True),
                ),
            )
        )
        result = build_setup(config, resolved, flag)
        assert isinstance(result, Ok)
        return result.value

    setup = setup_with_flag(True)
    assert setup.ensure_paragraphs
    assert [p.ensure_paragraphs for p in setup.passes] == [True, False, True]

    setup = setup_with_flag(False)
    assert not setup.ensure_paragraphs
    assert [p.ensure_paragraphs for p in setup.passes] == [False, False, True]


def test_resolve_call_settings() -> None:
    config = Config("u", "k", "default-model", 1.0, 100, {"a": 1})
    pass_definition = PassDefinition("p", "i", "chunk", {"b": 2}, "pass-model", False)
    model, params = resolve_call_settings(config, pass_definition)
    assert model == "pass-model"
    assert params == {"a": 1, "b": 2}


def test_pass_salt() -> None:
    pass_definition = PassDefinition("p", "i", "chunk", {}, None, False)
    assert pass_salt(pass_definition) == (CACHE_SALT_VERSION + "\x00p\x00i")


def test_cache_key_variants() -> None:
    base = cache_key("text", "model")
    assert base == cache_key("text", "model")
    assert base != cache_key("text", "model", "salt")
    assert base != cache_key("text", "model", "", "work")
    assert base != cache_key("text", "model", "", "", {"temperature": 1})


def test_parse_pass_table_inline() -> None:
    result = parse_pass_table("f", {"name": "p", "instruction": " do "}, ".").run()
    assert isinstance(result, Ok)
    assert result.value.instruction == "do"

    result = parse_pass_table("f", {"name": "p"}, ".").run()
    assert isinstance(result, Err)
    assert "exactly one of" in describe(result.error)


def test_load_instruction_text_inline() -> None:
    result = load_instruction_text("f", "p", {"instruction": " inst "}, ".").run()
    assert isinstance(result, Ok)
    assert result.value == "inst"

    result = load_instruction_text(
        "f", "p", {"instruction": "", "instruction_file": "x"}, "."
    ).run()
    assert isinstance(result, Err)


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
    assert options == {"ascii": True, "ensure_paragraphs": False}
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
            max_tokens=1000,
            params={"temperature": 1},
        ),
        passes=(
            PassDefinition(
                "translate", "T.", "chunk", {"reasoning": "high"}, None, True, False
            ),
        ),
        ensure_paragraphs=False,
    )
    report = setup_report(setup, True)
    assert "api.base_url: http://endpoint.test/v1" in report
    assert "api.model: m" in report
    assert "api.api_key: secr...ey" in report
    assert '"temperature": 1' in report
    assert (
        "pass 1/1 [translate]: mode=chunk ascii=True ensure_paragraphs=False"
        " model=<default> instruction=2 chars" in report
    )
    assert "options.ensure_paragraphs: True" in report


def test_document_api_settings_requires_a_table() -> None:
    result = document_api_settings("f", {"api": "nope"})
    assert isinstance(result, Err)
    assert "[api] must be a table" in describe(result.error)


def test_document_api_settings_rejects_non_table_params() -> None:
    result = document_api_settings("f", {"api": {"params": ["x"]}})
    assert isinstance(result, Err)
    assert "[api] params must be a table" in describe(result.error)


def test_timeout_api_setting_rejects_negative() -> None:
    result = timeout_api_setting("f", PartialApiSettings(), {"timeout": -3})
    assert isinstance(result, Err)
    assert "timeout must be a positive number" in describe(result.error)


def test_max_tokens_api_setting_rejects_float_with_message() -> None:
    result = max_tokens_api_setting("f", PartialApiSettings(), {"max_tokens": 2.5})
    assert isinstance(result, Err)
    assert "max_tokens must be a positive integer" in describe(result.error)


def test_environment_rejects_bogus_and_negative_numbers() -> None:
    result = api_settings_from_environment({"TRANSLATE_MAX_TOKENS": "bogus"})
    assert isinstance(result, Err)
    assert "TRANSLATE_MAX_TOKENS must be an integer" in describe(result.error)

    result = api_settings_from_environment({"TRANSLATE_TIMEOUT": "-1"})
    assert isinstance(result, Err)
    assert "TRANSLATE_TIMEOUT must be positive" in describe(result.error)

    result = api_settings_from_environment({"TRANSLATE_TIMEOUT": "1.5"})
    assert isinstance(result, Ok)
    assert result.value.timeout == 1.5


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


def test_parse_pass_table_rejects_non_table_and_empty_name() -> None:
    result = parse_pass_table("f", "chunk", ".").run()
    assert isinstance(result, Err)
    assert "entries must be tables" in describe(result.error)

    result = parse_pass_table("f", {"name": "  ", "instruction": "i"}, ".").run()
    assert isinstance(result, Err)
    assert "name must be a non-empty string" in describe(result.error)


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
    assert "instruction file not found" in describe(result.error)


def test_read_document_without_a_file_is_empty() -> None:
    result = read_document("f", "config file", exists=False).run()
    assert isinstance(result, Ok)
    assert result.value == {}


def test_resolve_config_path_discovers_local_then_user(
    tmp_path: Path, monkeypatch: Any
) -> None:
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    monkeypatch.chdir(tmp_path)
    assert resolve_config_path(None, {"XDG_CONFIG_HOME": str(empty_root)}).run() is (
        None
    )

    local = tmp_path / "l2l.toml"
    local.write_text("", encoding="utf-8")
    assert resolve_config_path(None, {}).run() == str(local)

    user_root = tmp_path / "cfg"
    user_path = user_root / "l2l" / "config.toml"
    user_path.parent.mkdir(parents=True)
    user_path.write_text("", encoding="utf-8")
    local.unlink()
    assert resolve_config_path(None, {"XDG_CONFIG_HOME": str(user_root)}).run() == (
        str(user_path)
    )


def test_load_setup_reports_malformed_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.toml"
    config_path.write_text("[api\nbroken", encoding="utf-8")
    arguments = Arguments(
        config=str(config_path),
        base_url=None,
        api_key=None,
        model=None,
        timeout=None,
        max_tokens=None,
        no_cache=False,
        ensure_paragraphs=False,
        verbose=False,
        show_log_path=False,
        cache_dir=None,
    )
    result = load_setup(arguments, {}).run()
    assert isinstance(result, Err)
    assert "cannot parse config file" in describe(result.error)


def test_api_params_must_contain_json_values() -> None:
    result = params_api_setting("f", PartialApiSettings(), {"params": {"t": object()}})
    assert isinstance(result, Err)
    assert "[api] params must contain only JSON values" in describe(result.error)


def test_pass_params_must_contain_json_values() -> None:
    result = pass_definition_from("f", "p", {"params": {"t": [object()]}}, "i")
    assert isinstance(result, Err)
    assert "params must contain only JSON values" in describe(result.error)
