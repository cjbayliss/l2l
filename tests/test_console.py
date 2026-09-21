import io
import time

from zh2en.console import (
    StatusLine,
    StatusView,
    draw_render,
    finish_render,
    interrupt_render,
    progress_render,
    raw_begin_render,
    raw_end_render,
    raw_write_render,
    start_render,
    status_erase_text,
    status_line_text,
    stop_render,
)


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
