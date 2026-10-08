from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

type HttpResponder = Callable[
    [bytes, dict[str, str]], tuple[int, tuple[tuple[str, str], ...], bytes]
]
type JsonReply = tuple[int, tuple[tuple[str, str], ...], bytes]


@dataclass(frozen=True)
class ReceivedRequest:
    path: str
    body: bytes
    headers: dict[str, str]

    def payload(self) -> dict[str, object]:
        return dict(json.loads(self.body))


def json_reply(payload: object, status: int = 200) -> JsonReply:
    return (status, (), json.dumps(payload).encode("utf-8"))


def raw_reply(status: int, body: bytes) -> JsonReply:
    return (status, (), body)


class LocalEndpoint:
    def __init__(self, responder: HttpResponder) -> None:
        endpoint = self

        class RecordingHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return None

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                headers = dict(self.headers.items())
                endpoint.requests.append(ReceivedRequest(self.path, body, headers))
                status, reply_headers, payload = responder(body, headers)
                self.send_response(status)
                for name, value in reply_headers:
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.requests: list[ReceivedRequest] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), RecordingHandler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return "http://%s:%d/v1" % (cast("str", host), port)

    def __enter__(self) -> LocalEndpoint:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)


def local_endpoint(responder: HttpResponder) -> LocalEndpoint:
    return LocalEndpoint(responder)
