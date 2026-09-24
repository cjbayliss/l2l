import io
import time

from fakes import make_console

from l2l.console import (
    Console,
    LogEvent,
    StatusLine,
    StatusView,
    draw_render,
    erase_rows_render,
    event_line,
    event_visible,
    finish_render,
    interrupt_render,
    progress_render,
    raw_begin_render,
    raw_end_render,
    raw_write_render,
    replay_rows,
    replay_text,
    row_count,
    start_render,
    status_erase_text,
    status_line_text,
    stop_render,
    terminal_size,
    toggle_verbose,
)
from l2l.monads import cons_to_tuple, write_ref


def test_status_line_text_renders_prefix_and_values() -> None:
    view = StatusView(label="Working", started=10.0, tokens=5, prefix="p: ", drawn="x")
    assert status_line_text(view, 12.5) == (
        "p: Working: time elapsed: 2.50s, tokens received: 5"
    )
    plain = StatusView(label="Working", started=10.0, tokens=5)
    assert status_line_text(plain, 10.0).startswith("Working: time elapsed: 0.00s")


def test_status_erase_text_clears_drawn_width() -> None:
    assert status_erase_text(StatusView(drawn="abc")) == "\r" + " " * 3 + "\r"
    assert status_erase_text(StatusView(drawn="")) == ""


def test_status_line_live_draws_progress_and_erases() -> None:
    stream = io.StringIO()
    ticks = iter((1.0, 1.5, 2.0, 2.5, 3.0))
    status = StatusLine(stream, live=True, monotonic=lambda: next(ticks, 9.0))
    status.start("Working").run()
    status.progress("Working", 3).run()
    status.stop().run()
    output = stream.getvalue()
    assert "Working" in output
    assert "\r" in output


def test_status_line_finish_writes_text_and_newline() -> None:
    stream = io.StringIO()
    ticks = iter((1.0, 1.5))
    status = StatusLine(stream, live=True, monotonic=lambda: next(ticks, 9.0))
    status.start("Working").run()
    status.finish("Done: 2.0s").run()
    output = stream.getvalue()
    assert output.endswith("Done: 2.0s\n")


def test_status_line_without_live_is_quiet() -> None:
    stream = io.StringIO()
    status = StatusLine(stream, live=False)
    status.start("Working").run()
    status.progress("Working", 1).run()
    status.stop().run()
    assert stream.getvalue() == ""


def test_draw_render_writes_line_and_records_drawn() -> None:
    view, text = draw_render(StatusView(started=1.0), 3.0)
    assert text == "\rWorking: time elapsed: 2.00s, tokens received: 0"
    assert view.drawn == "Working: time elapsed: 2.00s, tokens received: 0"


def test_draw_render_pads_over_longer_previous_line() -> None:
    long_view = StatusView(drawn="a much longer previous line")
    _, text = draw_render(long_view, 1.0)
    assert text.endswith(
        " "
        * (
            len("a much longer previous line")
            - len("Working: time elapsed: 0.00s, tokens received: 0")
        )
    )


def test_start_render_resets_tokens_and_label() -> None:
    view, text = start_render(
        StatusView(label="Old", tokens=99, drawn="stale"), "Working", 5.0
    )
    assert view.label == "Working"
    assert view.tokens == 0
    assert view.started == 5.0
    assert "Working" in text
    assert text.startswith("\r")


def test_progress_render_updates_label_and_accumulates() -> None:
    view = progress_render(StatusView(label="Working", tokens=2), "Thinking", 3)
    assert view == StatusView(label="Thinking", tokens=5)


def test_stop_render_erases_and_restores_prefix() -> None:
    view, text = stop_render(StatusView(prefix="p: ", drawn="line"))
    assert text == "\r" + " " * 4 + "\r" + "p: "
    assert view == StatusView(prefix="p: ", drawn="p: ")
    clean, text = stop_render(StatusView(drawn="line"))
    assert clean == StatusView()
    assert text == "\r" + " " * 4 + "\r"
    empty, text = stop_render(StatusView())
    assert empty == StatusView()
    assert text == ""


def test_interrupt_render_newlines_when_nothing_drawn() -> None:
    view, text = interrupt_render(StatusView(prefix="p: "), live=True)
    assert view == StatusView()
    assert text == "\n"

    view, text = interrupt_render(StatusView(prefix="p: ", drawn="line"), live=True)
    assert view == StatusView()
    assert text == "\r" + " " * 4 + "\r"

    view, text = interrupt_render(StatusView(), live=False)
    assert view == StatusView()
    assert text == ""


def test_finish_render_writes_prefix_then_text() -> None:
    view, line = finish_render(StatusView(prefix="p: ", drawn="line"), "Done")
    assert view == StatusView()
    assert line == "\r" + " " * 4 + "\r" + "p: " + "Done\n"

    view, line = finish_render(StatusView(), "Done")
    assert view == StatusView()
    assert line == "Done\n"


def test_raw_begin_erases_drawn_line_and_is_idempotent() -> None:
    view, text = raw_begin_render(StatusView(drawn="status line", prefix="p: "))
    assert text == "\r" + " " * 11 + "\r"
    assert view == StatusView(raw=True)

    again, text = raw_begin_render(view)
    assert again == view
    assert text == ""


def test_raw_write_tracks_open_line_state() -> None:
    view = raw_write_render(StatusView(raw=True), "thought ")
    assert view == StatusView(raw=True, raw_open=True)

    view = raw_write_render(view, "more\n")
    assert view == StatusView(raw=True)

    assert raw_write_render(view, "") == view


def test_raw_end_terminates_open_line_once_and_resets() -> None:
    view, text = raw_end_render(StatusView(raw=True, raw_open=True))
    assert text == "\n"
    assert view == StatusView()

    again, text = raw_end_render(view)
    assert again == view
    assert text == ""


def test_raw_end_after_trailing_newline_writes_nothing() -> None:
    view, text = raw_end_render(StatusView(raw=True))
    assert text == ""
    assert view == StatusView()


def test_interrupt_and_stop_during_raw_terminate_the_block() -> None:
    view, text = interrupt_render(StatusView(raw=True, raw_open=True), live=True)
    assert text == "\n"
    assert view == StatusView()

    view, text = stop_render(StatusView(raw=True, raw_open=True))
    assert text == "\n"
    assert view == StatusView()


def test_finish_render_terminates_raw_block_before_text() -> None:
    view, line = finish_render(StatusView(raw=True, raw_open=True), "Done.")
    assert line == "\nDone.\n"
    assert view == StatusView()

    view, line = finish_render(StatusView(raw=True), "Done.")
    assert line == "Done.\n"
    assert view == StatusView()


def test_start_render_leaves_raw_mode() -> None:
    view, _ = start_render(StatusView(raw=True, drawn="x"), "Working", 5.0)
    assert not view.raw


def test_status_line_raw_streams_deltas_to_the_stream() -> None:
    stream = io.StringIO()
    status = StatusLine(stream, live=True, monotonic=lambda: 1.0)
    status.begin_raw().run()
    status.begin_raw().run()
    status.write_raw("thought ").run()
    status.write_raw("more").run()
    status.end_raw().run()
    status.end_raw().run()
    assert stream.getvalue() == "thought more\n"


def test_status_line_tick_suppressed_while_raw() -> None:
    stream = io.StringIO()
    status = StatusLine(stream, live=True, monotonic=lambda: 1.0)
    status.start("Working").run()
    status.begin_raw().run()
    status.write_raw("thinking...").run()
    time.sleep(0.25)
    assert stream.getvalue().count("Working") == 1
    status.stop().run()
    assert stream.getvalue().endswith("thinking...\n")


def test_status_line_raw_without_live_still_writes() -> None:
    stream = io.StringIO()
    status = StatusLine(stream, live=False)
    status.begin_raw().run()
    status.write_raw("thought\n").run()
    status.end_raw().run()
    assert stream.getvalue() == "thought\n"


def make_live_console() -> tuple[Console, io.StringIO]:
    stream = io.StringIO()
    console = Console(stream, StatusLine(stream, live=True), term_size=lambda: (80, 24))
    return console, stream


def test_row_count_accounts_for_wrapping_and_blank_lines() -> None:
    assert row_count("one", 80) == 1
    assert row_count("a\nb", 80) == 2
    assert row_count("a\n\n", 80) == 2
    assert row_count("", 80) == 0
    assert row_count("x" * 100, 80) == 2
    assert row_count("x" * 10, 4) == 3
    assert row_count("word", 0) == 4


def test_row_count_weights_wide_characters_by_display_width() -> None:
    assert row_count("阿" * 40, 80) == 1
    assert row_count("阿" * 60, 80) == 2
    assert row_count("阿\nb", 80) == 2
    assert row_count("ok 阿爹", 80) == 1


def test_event_line_appends_newlines() -> None:
    assert event_line(LogEvent("done", False, False, 1)) == "done\n"
    assert event_line(LogEvent("thought", True, True, 1)) == "thought\n"
    assert event_line(LogEvent("thought\n", True, True, 1)) == "thought\n"


def test_event_visible_follows_verbose_mode() -> None:
    assert event_visible(LogEvent("a", False, False, 1), False)
    assert not event_visible(LogEvent("a", True, False, 1), False)
    assert event_visible(LogEvent("a", True, False, 1), True)


def test_replay_text_filters_events_and_keeps_pending_open() -> None:
    events = (
        LogEvent("one", False, False, 1),
        LogEvent("two", True, False, 1),
    )
    assert replay_text(events, "", False) == "one\n"
    assert replay_text(events, "", True) == "one\ntwo\n"
    assert replay_text(events, "think", True) == "one\ntwo\nthink"
    assert replay_text(events, "think", False) == "one\n"


def test_replay_rows_counts_visible_rows_only() -> None:
    events = (
        LogEvent("one", False, False, 1),
        LogEvent("two", True, False, 2),
    )
    assert replay_rows(events, "think", False, 80) == 1
    assert replay_rows(events, "think", True, 80) == 4
    assert replay_rows((), "", False, 80) == 0


def test_erase_rows_render_clamps_to_screen_height() -> None:
    assert erase_rows_render(0, 24) == "\x1b[J"
    assert erase_rows_render(3, 24) == "\x1b[3A\x1b[J"
    assert erase_rows_render(50, 24) == "\x1b[23A\x1b[J"
    assert erase_rows_render(3, 1) == "\x1b[J"
    assert erase_rows_render(-1, 24) == ""


def test_console_records_log_and_verbose_events() -> None:
    console, _ = make_live_console()
    console.log("one").run()
    console.log_verbose("two").run()
    events = cons_to_tuple(console.events.value)
    assert [event.text for event in events] == ["one", "two"]
    assert [event.verbose_only for event in events] == [False, True]
    assert [event.rows for event in events] == [1, 1]
    assert console.displayed_rows.value == 1


def test_log_verbose_hides_until_verbose_is_enabled() -> None:
    console, stream = make_live_console()
    console.log_verbose("hidden").run()
    assert stream.getvalue() == ""
    write_ref(console.verbose, True).run()
    console.log_verbose("shown").run()
    assert stream.getvalue() == "shown\n"
    assert console.displayed_rows.value == 1


def test_toggle_verbose_replays_visible_history() -> None:
    console, stream = make_live_console()
    console.log("one").run()
    console.log_verbose("two").run()
    assert stream.getvalue() == "one\n"

    assert toggle_verbose(console).run() is True
    assert stream.getvalue() == "one\n" + "\x1b[1A\x1b[J" + "one\ntwo\n"
    assert console.verbose.value is True

    assert toggle_verbose(console).run() is False
    assert stream.getvalue() == (
        "one\n" + "\x1b[1A\x1b[J" + "one\ntwo\n" + "\x1b[2A\x1b[J" + "one\n"
    )
    assert console.verbose.value is False


def test_console_replay_without_live_is_a_noop() -> None:
    console, stream = make_console()
    console.log("one").run()
    assert toggle_verbose(console).run() is True
    assert stream.getvalue() == "one\n"


def test_replay_clamps_erase_to_visible_rows() -> None:
    console, stream = make_live_console()
    console.log("first").run()
    console.log("last").run()
    write_ref(console.displayed_rows, 50).run()
    toggle_verbose(console).run()
    assert stream.getvalue() == ("first\nlast\n" + "\x1b[23A\x1b[J" + "first\nlast\n")


def test_stream_reasoning_captures_while_hidden_then_replays() -> None:
    console, stream = make_live_console()
    console.stream_reasoning("secret ").run()
    console.stream_reasoning("thought").run()
    assert stream.getvalue() == ""
    assert console.pending_raw.value == "secret thought"

    toggle_verbose(console).run()
    assert stream.getvalue() == "secret thought"
    assert console.status.view.value.raw is True

    console.stream_reasoning(" more").run()
    assert stream.getvalue() == "secret thought more"

    console.end_raw().run()
    assert stream.getvalue() == "secret thought more\n"
    assert console.pending_raw.value == ""
    assert cons_to_tuple(console.events.value)[-1] == LogEvent(
        "secret thought more", True, True, 1
    )


def test_toggle_off_mid_raw_clears_only_the_open_row() -> None:
    console, stream = make_live_console()
    write_ref(console.verbose, True).run()
    console.stream_reasoning("thinking").run()
    assert stream.getvalue() == "thinking"

    toggle_verbose(console).run()
    assert stream.getvalue() == "thinking" + "\r" + "\x1b[J"
    assert console.verbose.value is False
    assert console.status.view.value.raw is False

    toggle_verbose(console).run()
    assert stream.getvalue() == "thinking" + "\r" + "\x1b[J" + "thinking"
    # Read through locals: the Ref was mutated by the IO action, and mypy's
    # narrowing of the attribute chain cannot see that.
    raw_now: bool = console.status.view.value.raw
    raw_open_now: bool = console.status.view.value.raw_open
    assert raw_now is True
    assert raw_open_now is True


def test_toggle_off_erases_shown_reasoning_and_replays_from_events() -> None:
    console, stream = make_live_console()
    write_ref(console.verbose, True).run()
    console.stream_reasoning("thoughts\n").run()
    console.end_raw().run()
    assert stream.getvalue() == "thoughts\n"

    toggle_verbose(console).run()
    assert stream.getvalue() == "thoughts\n" + "\x1b[1A\x1b[J"

    toggle_verbose(console).run()
    assert stream.getvalue() == ("thoughts\n" + "\x1b[1A\x1b[J" + "thoughts\n")


def test_toggle_off_erases_all_rows_of_wide_character_reasoning() -> None:
    console, stream = make_live_console()
    write_ref(console.verbose, True).run()
    console.stream_reasoning("阿" * 120 + "\n").run()
    console.end_raw().run()
    assert stream.getvalue() == "阿" * 120 + "\n"

    toggle_verbose(console).run()
    assert stream.getvalue() == "阿" * 120 + "\n" + "\x1b[3A\x1b[J"


def test_log_commits_open_reasoning_block_before_the_message() -> None:
    console, stream = make_live_console()
    write_ref(console.verbose, True).run()
    console.stream_reasoning("thinking").run()
    console.log("msg").run()
    assert stream.getvalue() == "thinking" + "\n" + "msg\n"
    events = cons_to_tuple(console.events.value)
    assert [event.text for event in events] == ["thinking", "msg"]
    assert [event.raw for event in events] == [True, False]
    assert console.pending_raw.value == ""


def test_finish_records_the_prefixed_line() -> None:
    console, stream = make_live_console()
    console.write_partial("Enforcing ASCII... ").run()
    console.finish("0.50s, 1, 2, $0.01").run()
    assert stream.getvalue() == "Enforcing ASCII... 0.50s, 1, 2, $0.01\n"
    assert (
        cons_to_tuple(console.events.value)[-1].text
        == "Enforcing ASCII... 0.50s, 1, 2, $0.01"
    )
    assert console.displayed_rows.value == 1


def test_replay_erases_drawn_status_line() -> None:
    console, stream = make_live_console()
    console.status.start("Working").run()
    console.log("update").run()
    console.status.start("Working").run()
    console.interrupt().run()
    toggle_verbose(console).run()
    output = stream.getvalue()
    assert output.endswith("\x1b[1A\x1b[J" + "update\n")
    assert "\r" in output


def test_status_line_interrupt_erases_and_never_loops() -> None:
    stream = io.StringIO()
    status = StatusLine(stream, live=True, monotonic=lambda: 1.0)
    status.start("Working").run()
    time.sleep(0.25)
    status.interrupt().run()
    status.interrupt().run()
    status.stop().run()
    assert stream.getvalue().endswith("\r")


def test_terminal_size_reports_positive_dimensions() -> None:
    width, height = terminal_size()
    assert width >= 1
    assert height >= 1
