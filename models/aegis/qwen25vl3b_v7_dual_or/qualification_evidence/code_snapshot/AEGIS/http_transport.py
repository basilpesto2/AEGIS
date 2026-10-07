from __future__ import annotations

import base64
from dataclasses import dataclass
from http.client import (
    HTTPConnection,
    HTTPException,
    HTTPSConnection,
    HTTPMessage,
    HTTPResponse,
)
import ipaddress
import math
from queue import Empty, Queue
import socket
import ssl
from threading import BoundedSemaphore, Thread, Timer
import time
from urllib.parse import unquote, urlsplit
from urllib.request import getproxies, proxy_bypass


_DNS_RESOLVER_CAPACITY = BoundedSemaphore(8)
_PROXY_DISCOVERY_CAPACITY = BoundedSemaphore(8)
_TLS_CONTEXT_CAPACITY = BoundedSemaphore(8)
_DEFAULT_PROXY_DISCOVERY_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class BoundedHTTPResponse:
    status: int
    headers: HTTPMessage
    body: bytes


def is_loopback_hostname(hostname: str) -> bool:
    normalized = str(hostname).strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def environment_proxy_for_origin(origin: str) -> str | None:
    """Return the permitted environment proxy for an origin, if any."""

    return _environment_proxy_for_origin(
        origin,
        time.monotonic() + _DEFAULT_PROXY_DISCOVERY_TIMEOUT_SECONDS,
    )


def _environment_proxy_for_origin(origin: str, deadline: float) -> str | None:
    """Return an HTTPS proxy without letting OS proxy discovery outlive a deadline."""

    parsed = urlsplit(origin)
    hostname = parsed.hostname or ""
    if parsed.scheme.lower() != "https" or is_loopback_hostname(hostname):
        return None
    return _discover_environment_proxy(hostname, deadline)


def _discover_environment_proxy(hostname: str, deadline: float) -> str | None:
    if not _PROXY_DISCOVERY_CAPACITY.acquire(timeout=_remaining(deadline)):
        raise TimeoutError("HTTP proxy discovery capacity is currently full.")
    result: Queue[tuple[str, str | None]] = Queue(maxsize=1)

    def discover() -> None:
        try:
            if proxy_bypass(hostname):
                result.put(("ok", None))
                return
            proxy = getproxies().get("https")
            result.put(("ok", None if not proxy else str(proxy)))
        except Exception:
            # Proxy-discovery exceptions can interpolate configuration URLs and
            # credentials. Return only a sentinel to the requesting thread.
            result.put(("error", None))
        finally:
            _PROXY_DISCOVERY_CAPACITY.release()

    worker = Thread(target=discover, name="aegis-http-proxy", daemon=True)
    try:
        worker.start()
    except BaseException:
        _PROXY_DISCOVERY_CAPACITY.release()
        raise
    try:
        outcome, proxy = result.get(timeout=_remaining(deadline))
    except Empty:
        raise TimeoutError("HTTP operation exceeded its absolute deadline.") from None
    if outcome == "error":
        raise OSError("Environment proxy discovery failed.") from None
    return proxy


def _create_default_tls_context(deadline: float) -> ssl.SSLContext:
    """Initialize the system-verifying TLS context within the request deadline."""

    if not _TLS_CONTEXT_CAPACITY.acquire(timeout=_remaining(deadline)):
        raise TimeoutError("HTTP TLS context initialization capacity is currently full.")
    result: Queue[tuple[str, ssl.SSLContext | None]] = Queue(maxsize=1)

    def create() -> None:
        try:
            result.put(("ok", ssl.create_default_context()))
        except Exception:
            # Trust-store failures can contain local paths or platform details.
            # Return only a sentinel to the requesting thread.
            result.put(("error", None))
        finally:
            _TLS_CONTEXT_CAPACITY.release()

    worker = Thread(target=create, name="aegis-http-tls-context", daemon=True)
    try:
        worker.start()
    except BaseException:
        _TLS_CONTEXT_CAPACITY.release()
        raise
    try:
        outcome, context = result.get(timeout=_remaining(deadline))
    except Empty:
        raise TimeoutError("HTTP operation exceeded its absolute deadline.") from None
    if outcome == "error" or context is None:
        raise OSError("TLS context initialization failed.") from None
    return context


def request_with_deadline(
    origin: str,
    path: str,
    *,
    method: str,
    headers: dict[str, str],
    timeout_seconds: float,
    max_response_bytes: int,
    data: bytes | None = None,
) -> BoundedHTTPResponse:
    """Perform one HTTP request under a single connect/header/body deadline."""

    if (
        type(timeout_seconds) not in {int, float}
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be positive and finite.")
    if (
        isinstance(max_response_bytes, bool)
        or not isinstance(max_response_bytes, int)
        or max_response_bytes <= 0
    ):
        raise ValueError("max_response_bytes must be positive.")
    if not path.startswith("/"):
        raise ValueError("path must be origin-relative and start with '/'.")

    parsed = urlsplit(origin)
    hostname = parsed.hostname
    if parsed.scheme not in {"http", "https"} or hostname is None:
        raise ValueError("origin must be a valid HTTP(S) origin.")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    deadline = time.monotonic() + float(timeout_seconds)
    connection = _connection_for_origin(parsed, port, deadline)
    response_holder: dict[str, HTTPResponse | None] = {"response": None}

    def abort_request() -> None:
        response = response_holder["response"]
        if response is not None:
            try:
                response.close()
            except OSError:
                pass
        try:
            connection.close()
        except OSError:
            pass

    timer = Timer(_remaining(deadline), abort_request)
    timer.daemon = True
    timer.start()
    response: HTTPResponse | None = None
    try:
        connection.timeout = _remaining(deadline)
        connection.request(method, path, body=data, headers=dict(headers))
        _set_connection_timeout(connection, deadline)
        response = connection.getresponse()
        response_holder["response"] = response
        body = _read_bounded_body(
            response,
            connection,
            deadline=deadline,
            limit=max_response_bytes + 1,
        )
        _remaining(deadline)
        return BoundedHTTPResponse(
            status=int(response.status),
            headers=response.headers,
            body=body,
        )
    except (socket.timeout, TimeoutError):
        raise TimeoutError("HTTP operation exceeded its absolute deadline.") from None
    except HTTPException as exc:
        if time.monotonic() >= deadline:
            raise TimeoutError("HTTP operation exceeded its absolute deadline.") from None
        raise OSError("The peer returned an invalid or incomplete HTTP response.") from exc
    except (OSError, ValueError):
        if time.monotonic() >= deadline:
            raise TimeoutError("HTTP operation exceeded its absolute deadline.") from None
        raise
    finally:
        timer.cancel()
        response_holder["response"] = None
        if response is not None:
            try:
                response.close()
            except OSError:
                pass
        connection.close()


def _connection_for_origin(parsed, port: int, deadline: float):
    hostname = parsed.hostname
    assert hostname is not None
    if parsed.scheme == "http":
        timeout = _remaining(deadline)
        connection = HTTPConnection(hostname, port, timeout=timeout)
        return _bind_resolved_connection(connection, hostname, port, deadline)

    proxy = _environment_proxy_for_origin(parsed.geturl(), deadline)
    context = _create_default_tls_context(deadline)
    timeout = _remaining(deadline)
    if proxy is None:
        connection = HTTPSConnection(
            hostname,
            port,
            timeout=timeout,
            context=context,
        )
        return _bind_resolved_connection(connection, hostname, port, deadline)

    normalized_proxy = proxy if "://" in proxy else f"http://{proxy}"
    proxy_parsed = urlsplit(normalized_proxy)
    if proxy_parsed.scheme.lower() != "http" or proxy_parsed.hostname is None:
        raise ValueError("Only HTTP CONNECT proxies are supported for remote HTTPS.")
    proxy_port = proxy_parsed.port or 80
    connection = HTTPSConnection(
        proxy_parsed.hostname,
        proxy_port,
        timeout=timeout,
        context=context,
    )
    _bind_resolved_connection(
        connection,
        proxy_parsed.hostname,
        proxy_port,
        deadline,
    )
    tunnel_headers: dict[str, str] = {}
    if proxy_parsed.username is not None:
        username = unquote(proxy_parsed.username)
        password = unquote(proxy_parsed.password or "")
        credentials = base64.b64encode(
            f"{username}:{password}".encode("utf-8")
        ).decode("ascii")
        tunnel_headers["Proxy-Authorization"] = f"Basic {credentials}"
    connection.set_tunnel(hostname, port=port, headers=tunnel_headers)
    return connection


def _bind_resolved_connection(
    connection,
    hostname: str,
    port: int,
    deadline: float,
):
    """Resolve once under the request deadline without changing Host or TLS SNI."""

    addresses = _resolve_host(hostname, port, deadline)

    def create_connection(_address, _timeout, source_address):
        return _connect_resolved(addresses, deadline, source_address)

    # http.client deliberately exposes this as an instance hook. Keeping
    # `connection.host` as the original hostname preserves the HTTP Host header and,
    # for HTTPS (including CONNECT), certificate verification and TLS SNI.
    connection._create_connection = create_connection
    return connection


def _resolve_host(hostname: str, port: int, deadline: float):
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if isinstance(literal, ipaddress.IPv4Address):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (str(literal), port))]
    if isinstance(literal, ipaddress.IPv6Address):
        return [
            (socket.AF_INET6, socket.SOCK_STREAM, 0, "", (str(literal), port, 0, 0))
        ]

    if not _DNS_RESOLVER_CAPACITY.acquire(timeout=_remaining(deadline)):
        raise TimeoutError("HTTP DNS resolution capacity is currently full.")
    result: Queue[tuple[str, object]] = Queue(maxsize=1)

    def resolve() -> None:
        try:
            addresses = socket.getaddrinfo(
                hostname,
                port,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
            )
            result.put(("ok", addresses))
        except Exception as exc:
            result.put(("error", exc))
        finally:
            _DNS_RESOLVER_CAPACITY.release()

    resolver = Thread(target=resolve, name="aegis-http-dns", daemon=True)
    try:
        resolver.start()
    except BaseException:
        _DNS_RESOLVER_CAPACITY.release()
        raise
    try:
        outcome, value = result.get(timeout=_remaining(deadline))
    except Empty:
        raise TimeoutError("HTTP operation exceeded its absolute deadline.") from None
    if outcome == "error":
        assert isinstance(value, Exception)
        raise value
    addresses = list(value)
    if not addresses:
        raise OSError(f"No network address was found for {hostname!r}.")
    return addresses


def _connect_resolved(addresses, deadline: float, source_address):
    last_error: OSError | None = None
    for index, (
        family,
        socktype,
        protocol,
        _canonical_name,
        socket_address,
    ) in enumerate(addresses):
        candidate = socket.socket(family, socktype, protocol)
        try:
            # Do not let an unusable first AAAA/A record consume the global
            # deadline. Each remaining address receives an equal share; the last
            # candidate receives all time still available.
            remaining = _remaining(deadline)
            addresses_left = len(addresses) - index
            candidate.settimeout(remaining / addresses_left)
            if source_address:
                candidate.bind(source_address)
            candidate.connect(socket_address)
            # Address attempts receive only a slice so one unreachable record
            # cannot starve fallbacks. Once connected, CONNECT and TLS must use
            # all time still available under the original absolute deadline.
            candidate.settimeout(_remaining(deadline))
            return candidate
        except OSError as exc:
            last_error = exc
            candidate.close()
    if last_error is not None:
        raise last_error
    raise OSError("No usable network address was returned by the resolver.")


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("HTTP operation exceeded its absolute deadline.")
    return remaining


def _set_connection_timeout(connection, deadline: float) -> None:
    remaining = _remaining(deadline)
    connection.timeout = remaining
    sock = getattr(connection, "sock", None)
    if sock is not None:
        sock.settimeout(remaining)


def _read_bounded_body(
    response: HTTPResponse,
    connection,
    *,
    deadline: float,
    limit: int,
) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while size < limit:
        _set_connection_timeout(connection, deadline)
        amount = min(64 * 1024, limit - size)
        reader = getattr(response, "read1", None)
        chunk = reader(amount) if callable(reader) else response.read(amount)
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
    return b"".join(chunks)


__all__ = [
    "BoundedHTTPResponse",
    "environment_proxy_for_origin",
    "is_loopback_hostname",
    "request_with_deadline",
]
