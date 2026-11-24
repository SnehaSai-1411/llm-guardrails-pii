"""FastAPI/Starlette guardrail middleware.

:class:`GuardrailMiddleware` sits in front of any FastAPI app and applies
the guardrails to traffic in both directions:

* **Request side** — for JSON request bodies: redact PII from every
  string value (in place, so the endpoint only ever sees the redacted
  text) and run the prompt-injection check over the body's text. A
  high-severity injection match is rejected with HTTP 400 before the
  endpoint runs.
* **Response side** — for configured ``/ask``-style routes: run the
  grounding check over the endpoint's JSON response (``answer`` field
  against ``contexts``/``context``) and annotate the response with a
  ``grounding`` object plus an ``X-Grounding-Verdict`` header. By
  default a ``block`` verdict only annotates; set
  ``grounding_block_on_fail=True`` to reject such responses with 502.

Per-route behaviour is set through the constructor: ``request_paths``
limits which routes get request-side checks (default: all JSON routes
except ``exclude_paths``), ``grounding_paths`` lists the routes whose
responses get the grounding check (default: ``("/ask",)``).

Mount it like any Starlette middleware::

    from fastapi import FastAPI
    from guardrails.middleware import GuardrailMiddleware

    app = FastAPI()
    app.add_middleware(GuardrailMiddleware, grounding_paths=["/ask"])
"""

from __future__ import annotations

import json
from typing import Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from guardrails import grounding as grounding_mod
from guardrails import injection as injection_mod
from guardrails import pii as pii_mod
from guardrails.config import GuardrailConfig

_DEFAULT_EXCLUDE = {"/health", "/docs", "/openapi.json", "/redoc", "/favicon.ico"}
_CONTEXT_KEYS = ("contexts", "context", "retrieved_context", "sources")


def _redact_json(value: Any, config: GuardrailConfig) -> tuple[Any, bool]:
    """Recursively redact PII in JSON string values. Returns (new, changed)."""
    if isinstance(value, str):
        result = pii_mod.redact(value, config)
        return result["redacted_text"], bool(result["entities_found"])
    if isinstance(value, list):
        changed = False
        out = []
        for item in value:
            new_item, item_changed = _redact_json(item, config)
            changed = changed or item_changed
            out.append(new_item)
        return out, changed
    if isinstance(value, dict):
        changed = False
        out = {}
        for key, item in value.items():
            new_item, item_changed = _redact_json(item, config)
            changed = changed or item_changed
            out[key] = new_item
        return out, changed
    return value, False


def _collect_strings(value: Any, acc: list[str]) -> None:
    """Gather every string in a JSON value for the injection scan."""
    if isinstance(value, str):
        acc.append(value)
    elif isinstance(value, list):
        for item in value:
            _collect_strings(item, acc)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_strings(item, acc)


class GuardrailMiddleware(BaseHTTPMiddleware):
    """PII redaction + injection blocking (request) + grounding (response)."""

    def __init__(
        self,
        app: Callable[..., Any],
        config: GuardrailConfig | None = None,
        *,
        request_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        grounding_paths: list[str] | None = None,
    ) -> None:
        super().__init__(app)
        self.config = config or GuardrailConfig()
        self.request_paths = set(request_paths) if request_paths else None
        self.exclude_paths = set(exclude_paths) if exclude_paths is not None else _DEFAULT_EXCLUDE
        self.grounding_paths = set(grounding_paths) if grounding_paths is not None else {"/ask"}

    # -- routing helpers -------------------------------------------------

    def _request_checked(self, path: str) -> bool:
        if path in self.exclude_paths:
            return False
        if self.request_paths is not None:
            return path in self.request_paths
        return True

    def _response_checked(self, path: str) -> bool:
        return path in self.grounding_paths

    # -- dispatch --------------------------------------------------------

    async def dispatch(self, request: Request, call_next: Callable[..., Any]) -> Response:
        path = request.url.path

        if self._request_checked(path) and request.method in {"POST", "PUT", "PATCH"}:
            blocked = await self._guard_request(request)
            if blocked is not None:
                return blocked

        response = await call_next(request)

        if (
            self._response_checked(path)
            and self.config.grounding_enabled
            and "application/json" in (response.headers.get("content-type") or "")
        ):
            response = await self._guard_response(response)
        return response

    # -- request side ----------------------------------------------------

    async def _guard_request(self, request: Request) -> Response | None:
        """Redact PII / block injections. Returns a 400 response, or None."""
        if "application/json" not in (request.headers.get("content-type") or ""):
            return None
        raw_body = await request.body()
        if not raw_body:
            return None
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            return None  # not our problem — let the endpoint validate it

        # Injection check runs on the text as sent (before redaction, so a
        # PII placeholder can never mask an injection phrase).
        if self.config.injection_enabled:
            strings: list[str] = []
            _collect_strings(payload, strings)
            result = injection_mod.check("\n".join(strings), self.config)
            if injection_mod.is_blocked(result, self.config):
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "blocked_by_guardrail",
                        "detail": (
                            "Request blocked: prompt-injection pattern detected. "
                            "Rephrase without instruction-override language."
                        ),
                        "matches": result["matches"],
                    },
                )

        # PII redaction rewrites the body the endpoint will see.
        new_body = raw_body
        if self.config.pii_enabled:
            redacted_payload, changed = _redact_json(payload, self.config)
            if changed:
                new_body = json.dumps(redacted_payload).encode("utf-8")
                self._set_content_length(request, len(new_body))

        # Hand the downstream app the (possibly redacted) bytes. Two
        # mechanisms, because Starlette changed how BaseHTTPMiddleware
        # replays the body across versions:
        #
        # * Starlette >= 0.36ish caches the body in `request._body` and
        #   replays THAT to the downstream app (via `_CachedRequest`
        #   in starlette/middleware/base.py) — so overwrite the cache.
        # * Older versions replayed `request._receive` instead, and
        #   `request.body()` above already consumed the original stream —
        #   so also replace the receive channel with a fresh one.
        # Setting both keeps this working on either behaviour.
        request._body = new_body  # noqa: SLF001 — required by Starlette
        sent = False

        async def receive() -> dict[str, Any]:
            nonlocal sent
            if sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            sent = True
            return {"type": "http.request", "body": new_body, "more_body": False}

        request._receive = receive  # noqa: SLF001 — the supported pattern
        return None

    @staticmethod
    def _set_content_length(request: Request, length: int) -> None:
        headers = [(k, v) for k, v in request.scope["headers"] if k.lower() != b"content-length"]
        headers.append((b"content-length", str(length).encode()))
        request.scope["headers"] = headers

    # -- response side ---------------------------------------------------

    async def _guard_response(self, response: Response) -> Response:
        """Annotate (or reject) an /ask-style JSON response with grounding."""
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return Response(content=body, status_code=response.status_code, headers=dict(response.headers))
        if not isinstance(payload, dict) or "answer" not in payload:
            return Response(content=body, status_code=response.status_code, headers=dict(response.headers))

        contexts: list[str] = []
        for key in _CONTEXT_KEYS:
            value = payload.get(key)
            if isinstance(value, str):
                contexts = [value]
                break
            if isinstance(value, list):
                contexts = [str(v) for v in value]
                break

        result = grounding_mod.check(str(payload["answer"]), contexts, self.config)

        if result["verdict"] == "block" and self.config.grounding_block_on_fail:
            return JSONResponse(
                status_code=502,
                content={
                    "error": "blocked_by_guardrail",
                    "detail": (
                        "Response blocked: the answer is not grounded in the "
                        "retrieved context."
                    ),
                    "grounding": result,
                },
            )

        payload["grounding"] = result
        headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
        headers["X-Grounding-Verdict"] = result["verdict"]
        return JSONResponse(content=payload, status_code=response.status_code, headers=headers)
