import io

from zh2en.console import StatusLine, StatusView, status_erase_text, status_line_text


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
