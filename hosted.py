"""Password-protected WSGI entry point for the hosted Deal Structure demo."""

import base64
import hashlib
import hmac
import json
import os
import re
from threading import BoundedSemaphore
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

from deal_structure import DEFAULT_MODEL, HTML, interpret, validate_input

DEMO_USER = "professor"
MAX_BODY = 65536
AI_SLOT = BoundedSemaphore(1)


def configure():
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    password = os.environ.get("DEMO_PASSWORD", "")
    origin = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
    url = urlsplit(origin)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
        raise RuntimeError("Set OPENROUTER_API_KEY in the Render environment.")
    if len(password) < 16 or not password.isascii() or not password.isprintable():
        raise RuntimeError("Set DEMO_PASSWORD to at least 16 printable ASCII characters.")
    if url.scheme != "https" or not url.hostname or url.path or url.query or url.fragment or url.username:
        raise RuntimeError("Expected Render to set RENDER_EXTERNAL_URL to the HTTPS service origin.")
    return key, f"{DEMO_USER}:{password}".encode("ascii"), origin, url.netloc


KEY, AUTH_CREDENTIALS, ORIGIN, HOSTNAME = configure()
MODEL = os.environ.get("OPENROUTER_MODEL", "").strip() or DEFAULT_MODEL
PAGE = HTML.read_text(encoding="utf-8").replace(
    "<head>", '<head>\n<meta name="deal-ai" content="enabled">', 1
)
SCRIPT = PAGE.split("<script>", 1)[1].split("</script>", 1)[0]
SCRIPT_HASH = base64.b64encode(hashlib.sha256(SCRIPT.encode()).digest()).decode("ascii")
PAGE_BYTES = PAGE.encode("utf-8")
CSP = (
    "default-src 'none'; "
    f"script-src 'sha256-{SCRIPT_HASH}'; "
    "style-src 'unsafe-inline'; connect-src 'self'; "
    "img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


def authenticated(header):
    scheme, _, encoded = header.partition(" ")
    if scheme.lower() != "basic" or len(encoded) > 2048:
        return False
    try:
        credentials = base64.b64decode(encoded, validate=True)
    except ValueError:
        return False
    return hmac.compare_digest(credentials, AUTH_CREDENTIALS)


def send(start_response, status, body, *, content_type="application/json", extra=()):
    data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    headers = [
        ("Content-Type", content_type),
        ("Content-Length", str(len(data))),
        ("Cache-Control", "no-store"),
        ("Content-Security-Policy", CSP),
        ("X-Content-Type-Options", "nosniff"),
        ("Referrer-Policy", "no-referrer"),
        ("X-Robots-Tag", "noindex, nofollow"),
    ]
    start_response(status, [*headers, *extra])
    return [data]


def respond_to_interpretation(environ, start_response):
    if environ.get("HTTP_ORIGIN") != ORIGIN:
        return send(start_response, "403 Forbidden", {"error": "Send requests from the demo page."})
    if environ.get("CONTENT_TYPE", "").partition(";")[0].strip().lower() != "application/json":
        return send(start_response, "415 Unsupported Media Type", {"error": "Expected a JSON request."})
    try:
        size = int(environ.get("CONTENT_LENGTH") or "0")
        if not 0 < size <= MAX_BODY:
            raise ValueError("The request is empty or too large.")
        payload = validate_input(json.loads(environ["wsgi.input"].read(size)))
    except (ValueError, UnicodeError) as error:
        message = "The request contained invalid JSON." if isinstance(error, (json.JSONDecodeError, UnicodeError)) else str(error)
        return send(start_response, "400 Bad Request", {"error": message})

    if not AI_SLOT.acquire(blocking=False):
        return send(start_response, "429 Too Many Requests", {"error": "Another AI interpretation is running. Try again shortly."})
    try:
        result = interpret(payload, KEY, MODEL)
    except HTTPError as error:
        message = {
            401: "The demo's OpenRouter key needs attention. Contact the demo owner.",
            402: "This demo has reached its OpenRouter credit limit.",
            429: "OpenRouter is rate limited. Try again shortly.",
        }.get(error.code, f"OpenRouter returned HTTP {error.code}. Try again or use the chart editor.")
        return send(start_response, "502 Bad Gateway", {"error": message})
    except (URLError, TimeoutError):
        return send(start_response, "502 Bad Gateway", {"error": "Could not reach OpenRouter. Try again shortly."})
    except ValueError as error:
        return send(start_response, "502 Bad Gateway", {"error": str(error)})
    finally:
        AI_SLOT.release()
    return send(start_response, "200 OK", result)


def application(environ, start_response):
    method = environ.get("REQUEST_METHOD")
    path = environ.get("PATH_INFO")
    if method == "GET" and path == "/healthz":
        return send(start_response, "200 OK", {"status": "ok"})
    if environ.get("HTTP_HOST") != HOSTNAME:
        return send(start_response, "403 Forbidden", {"error": "Open the demo from its Render address."})
    if not authenticated(environ.get("HTTP_AUTHORIZATION", "")):
        return send(
            start_response, "401 Unauthorized", {"error": "A demo password is required."},
            extra=(("WWW-Authenticate", 'Basic realm="Deal Structure Demo", charset="UTF-8"'),),
        )
    if method == "GET" and path in ("/", "/Deal_Structure.html"):
        return send(start_response, "200 OK", PAGE_BYTES, content_type="text/html; charset=utf-8")
    if method == "POST" and path == "/api/interpret":
        return respond_to_interpretation(environ, start_response)
    return send(start_response, "404 Not Found", {"error": "Page unavailable."})
