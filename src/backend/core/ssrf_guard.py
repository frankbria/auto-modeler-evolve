"""SSRF protection for outbound requests driven by user-supplied URLs.

Two ingestion surfaces let a user influence where the server makes an HTTP
request: dataset import-from-URL (``POST /api/data/upload-url``) and webhook
dispatch (registered URLs POSTed to when deployment events fire). Without a
guard, either can be pointed at ``169.254.169.254`` (cloud metadata),
``localhost``, or an RFC1918 host to probe / exfiltrate from the internal
network.

``assert_safe_url`` is the single chokepoint. It rejects non-http(s) schemes,
IP-literal hosts in any private/loopback/link-local/reserved range, and
hostnames that resolve to such ranges. Because it re-resolves at call time,
calling it immediately before each request also blunts DNS-rebinding (a host
that resolved public at registration but flips to private at dispatch is caught
on dispatch).

Pure, dependency-free (stdlib only) so it is trivially unit-testable.
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.request
from urllib.parse import urlparse


class UnsafeURLError(ValueError):
    """Raised when a URL is rejected as unsafe for an outbound request."""


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True if ``ip`` is anything other than a globally-routable public address.

    We allowlist by routability rather than enumerating bad ranges: ``is_global``
    is False for loopback (127/8, ::1), link-local (169.254/16 incl. the cloud
    metadata endpoint, fe80::/10), private/RFC1918, unique-local IPv6 (fc00::/7),
    carrier-grade NAT (100.64/10), reserved, and benchmarking ranges in one
    check. Multicast and the unspecified address (0.0.0.0, ::) are added
    explicitly for belt-and-braces. IPv4-mapped IPv6 is unwrapped first so
    ``::ffff:127.0.0.1`` cannot slip through.
    """
    if getattr(ip, "ipv4_mapped", None):
        ip = ip.ipv4_mapped  # type: ignore[assignment]
    return (not ip.is_global) or ip.is_multicast or ip.is_unspecified


def _resolve_host(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Resolve ``host`` to every address it maps to. Empty list on failure."""
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return []
    addrs: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        sockaddr = info[4]
        try:
            addrs.append(ipaddress.ip_address(sockaddr[0]))
        except ValueError:
            continue
    return addrs


def assert_safe_url(url: str, *, allow_unresolved: bool = False) -> None:
    """Raise ``UnsafeURLError`` unless ``url`` is safe to request server-side.

    Args:
        url: the absolute http(s) URL about to be fetched / POSTed.
        allow_unresolved: when a hostname cannot be resolved, allow it (used at
            **registration** time, where a webhook host may not yet exist) vs.
            block it (used at **fetch/dispatch** time, where we are about to
            connect and an unresolvable host is useless anyway).

    A host given as an IP literal in a blocked range is always rejected,
    regardless of ``allow_unresolved``.
    """
    try:
        parsed = urlparse(url)
        scheme, host = parsed.scheme, parsed.hostname
    except ValueError as exc:
        # Malformed URL (e.g. bad IPv6 brackets) — treat as unsafe, not a 500.
        raise UnsafeURLError(f"Malformed URL: {exc}") from exc

    if scheme not in ("http", "https"):
        raise UnsafeURLError("URL must start with http:// or https://")
    if not host:
        raise UnsafeURLError("URL has no host")

    # IP literal — validate directly, no DNS.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_blocked_ip(literal):
            raise UnsafeURLError(f"Refusing to connect to non-public address {host}")
        return

    # Hostname — resolve and validate every address it points at.
    addrs = _resolve_host(host)
    if not addrs:
        if allow_unresolved:
            return
        raise UnsafeURLError(f"Could not resolve host {host}")
    for ip in addrs:
        if _is_blocked_ip(ip):
            raise UnsafeURLError(f"Host {host} resolves to non-public address {ip}")


class _ValidatingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-runs the SSRF guard on every redirect hop.

    ``urlopen`` follows 3xx automatically, so a public first hop that returns
    ``Location: http://169.254.169.254/...`` would otherwise reach the metadata
    endpoint. Validating ``newurl`` here closes that bypass.
    """

    def __init__(self, allow_unresolved: bool) -> None:
        super().__init__()
        self._allow_unresolved = allow_unresolved

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        # Raises UnsafeURLError (propagated out of urlopen) for an unsafe hop.
        assert_safe_url(newurl, allow_unresolved=self._allow_unresolved)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def safe_urlopen(req, *, timeout: float, allow_unresolved: bool = False):
    """Like ``urllib.request.urlopen`` but SSRF-validates the request.

    Validates the initial URL itself (so this is safe to use as a standalone
    primitive — callers need not pre-check) and re-validates every redirect hop,
    which the default opener does not do.
    """
    url = req.full_url if hasattr(req, "full_url") else req
    assert_safe_url(url, allow_unresolved=allow_unresolved)
    opener = urllib.request.build_opener(_ValidatingRedirectHandler(allow_unresolved))
    return opener.open(req, timeout=timeout)
