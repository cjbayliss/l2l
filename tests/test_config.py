from typing import Any

import zh2en as z


def ok_document(document: dict[str, Any]) -> z.Result[dict[str, Any], str]:
    return z.Ok(document)


def test_string_api_settings() -> None:
    result = z.string_api_settings("f", {"base_url": " http://x ", "model": 3})
    assert isinstance(result, z.Err)
    assert "[api] model" in result.error

    result = z.string_api_settings("f", {"base_url": " http://x ", "api_key": "k"})
    assert isinstance(result, z.Ok)
    assert result.value["base_url"] == "http://x"


def test_timeout_api_setting() -> None:
    base: dict[str, Any] = {}
    assert isinstance(z.timeout_api_setting("f", base, {"timeout": 0}), z.Err)
    assert isinstance(z.timeout_api_setting("f", base, {"timeout": True}), z.Err)
    assert isinstance(z.timeout_api_setting("f", base, {"timeout": "5"}), z.Err)
    result = z.timeout_api_setting("f", base, {"timeout": 5})
    assert isinstance(result, z.Ok)
    assert result.value["timeout"] == 5.0


def test_max_tokens_api_setting() -> None:
    base: dict[str, Any] = {}
    assert isinstance(z.max_tokens_api_setting("f", base, {"max_tokens": -1}), z.Err)
    assert isinstance(z.max_tokens_api_setting("f", base, {"max_tokens": 1.5}), z.Err)
    result = z.max_tokens_api_setting("f", base, {"max_tokens": 7})
    assert isinstance(result, z.Ok)
    assert result.value["max_tokens"] == 7


def test_document_api_settings_unknown_key() -> None:
    result = z.document_api_settings("f", {"api": {"nope": 1}})
    assert isinstance(result, z.Err)
    assert "unknown key(s): nope" in result.error


def test_merge_api_settings_params_deep_merge() -> None:
    merged = z.merge_api_settings(
        {"model": "a", "params": {"x": 1, "y": 1}},
        {"params": {"y": 2, "z": 3}, "model": "b"},
    )
    assert merged == {"model": "b", "params": {"x": 1, "y": 2, "z": 3}}


def test_api_settings_from_environment() -> None:
    result = z.api_settings_from_environment(
        {"TRANSLATE_MODEL": " m ", "TRANSLATE_TIMEOUT": "bogus"}
    )
    assert isinstance(result, z.Err)
    assert "TRANSLATE_TIMEOUT" in result.error

    result = z.api_settings_from_environment(
        {"TRANSLATE_BASE_URL": "http://e", "TRANSLATE_MAX_TOKENS": "9"}
    )
    assert isinstance(result, z.Ok)
    assert result.value["base_url"] == "http://e"
    assert result.value["max_tokens"] == 9


def test_api_settings_from_arguments() -> None:
    arguments = z.Arguments(
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
    assert z.api_settings_from_arguments(arguments) == {
        "api_key": "k",
        "timeout": 2.0,
    }


def test_merged_api_settings_precedence() -> None:
    arguments = z.Arguments(
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
    selected: z.Result[dict[str, Any], str] = z.Ok({})
    result = z.merged_api_settings(arguments, environment, "user", user, None, selected)
    assert isinstance(result, z.Ok)
    config = result.value
    assert config.base_url == "http://env"
    assert config.api_key == "env-key"
    assert config.model == "arg-model"
    assert config.timeout == 5.0
    assert config.params["a"] == 1


def test_build_config_missing() -> None:
    result = z.build_config(z.default_api_settings())
    assert isinstance(result, z.Err)
    assert "missing required API settings" in result.error


def test_validate_document() -> None:
    assert isinstance(z.validate_document("f", {}), z.Ok)
    result = z.validate_document("f", {"weird": 1})
    assert isinstance(result, z.Err)
    assert "weird" in result.error


def test_parse_options_table() -> None:
    assert z.parse_options_table("f", {"ascii": True}) == z.Ok(
        {"ascii": True, "ensure_paragraphs": False}
    )
    assert z.parse_options_table("f", {"ensure_paragraphs": True}) == z.Ok(
        {"ascii": False, "ensure_paragraphs": True}
    )
    assert isinstance(z.parse_options_table("f", {"ascii": "yes"}), z.Err)
    assert isinstance(z.parse_options_table("f", {"ensure_paragraphs": "yes"}), z.Err)
    assert isinstance(z.parse_options_table("f", {"other": 1}), z.Err)
    assert isinstance(z.parse_options_table("f", "x"), z.Err)


def test_pass_definition_from() -> None:
    result = z.pass_definition_from("f", "p", {"mode": "weird"}, "inst")
    assert isinstance(result, z.Err)

    result = z.pass_definition_from(
        "f", "p", {"mode": "chunk", "params": {"t": 1}}, "inst"
    )
    assert isinstance(result, z.Ok)
    assert result.value.mode == "chunk"
    assert result.value.model is None
    assert result.value.ascii is None

    result = z.parse_pass_table(
        "f", {"name": "p", "mode": "chunk", "strict_fidelity": True}, "."
    ).run()
    assert isinstance(result, z.Err)
    assert "unknown key(s): strict_fidelity" in result.error


def test_apply_default_ascii() -> None:
    passes = (
        z.PassDefinition("a", "i", "chunk", {}, None, None),
        z.PassDefinition("b", "i", "chunk", {}, None, True),
    )
    applied = z.apply_default_ascii(passes, {"ascii": True})
    assert [p.ascii for p in applied] == [True, True]


def test_resolve_call_settings() -> None:
    config = z.Config("u", "k", "default-model", 1.0, 100, {"a": 1})
    pass_definition = z.PassDefinition("p", "i", "chunk", {"b": 2}, "pass-model", False)
    model, params = z.resolve_call_settings(config, pass_definition)
    assert model == "pass-model"
    assert params == {"a": 1, "b": 2}


def test_pass_salt() -> None:
    pass_definition = z.PassDefinition("p", "i", "chunk", {}, None, False)
    assert z.pass_salt(pass_definition) == (z.CACHE_SALT_VERSION + "\x00p\x00i")


def test_cache_key_variants() -> None:
    base = z.cache_key("text", "model")
    assert base == z.cache_key("text", "model")
    assert base != z.cache_key("text", "model", "salt")
    assert base != z.cache_key("text", "model", "", "work")
    assert base != z.cache_key("text", "model", "", "", {"temperature": 1})


def test_parse_pass_table_inline() -> None:
    result = z.parse_pass_table("f", {"name": "p", "instruction": " do "}, ".").run()
    assert isinstance(result, z.Ok)
    assert result.value.instruction == "do"

    result = z.parse_pass_table("f", {"name": "p"}, ".").run()
    assert isinstance(result, z.Err)
    assert "exactly one of" in result.error


def test_load_instruction_text_inline() -> None:
    result = z.load_instruction_text("f", "p", {"instruction": " inst "}, ".").run()
    assert isinstance(result, z.Ok)
    assert result.value == "inst"

    result = z.load_instruction_text(
        "f", "p", {"instruction": "", "instruction_file": "x"}, "."
    ).run()
    assert isinstance(result, z.Err)


def test_resolve_passes_requires_pass_table() -> None:
    result = z.resolve_passes("user", ok_document({"api": {}}), None, z.Ok({})).run()
    assert isinstance(result, z.Err)
    assert "no [[pass]] tables" in result.error


def test_resolve_passes_selection_and_options() -> None:
    user = ok_document(
        {
            "api": {},
            "options": {"ascii": True},
            "pass": [{"name": "p", "instruction": "i"}],
        }
    )
    result = z.resolve_passes("user", user, None, z.Ok({})).run()
    assert isinstance(result, z.Ok)
    options, passes = result.value
    assert options == {"ascii": True, "ensure_paragraphs": False}
    assert len(passes) == 1


def test_resolve_config_path_requested_and_environment() -> None:
    assert z.resolve_config_path("a.toml", {}).run() == "a.toml"
    result = z.resolve_config_path(None, {"TRANSLATE_CONFIG": " b.toml "}).run()
    assert result == "b.toml"
