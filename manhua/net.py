"""HTTPS trust setup.

Python's `requests` verifies certificates against the bundled `certifi` roots,
which do NOT include the root certificates that TLS-inspecting antivirus
software installs (Avast, AVG, Kaspersky, ESET, BitDefender) or that corporate
MITM proxies use. On those machines every HTTPS call from Python fails with:

    SSLError: certificate verify failed: unable to get local issuer certificate

...while curl and the browser work fine, because they use the OS trust store,
which *does* contain the AV's root. This is common enough on Windows that it
would otherwise look like the app is broken.

`truststore` redirects Python's TLS verification to the OS trust store, which
fixes it without disabling verification and without asking anyone to add
antivirus exceptions.
"""
from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def enable_system_certs() -> bool:
    """Verify TLS against the OS trust store. Safe to call repeatedly.

    Returns True if injection succeeded. Failure is non-fatal: on a machine
    without TLS interception the default certifi roots work fine.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
        return True
    except Exception:
        return False


def diagnose(url: str = "https://pypi.org") -> str:
    """Human-readable TLS reachability check, used by `manhua doctor`."""
    import requests

    enable_system_certs()
    try:
        requests.get(url, timeout=15)
        return "ok"
    except Exception as exc:
        name = type(exc).__name__
        if "SSL" in name or "SSL" in str(exc):
            return (
                "TLS verification failed even against the OS trust store.\n"
                "  If you use antivirus HTTPS scanning or a corporate proxy, "
                "export its root certificate and set REQUESTS_CA_BUNDLE to it."
            )
        return f"{name}: {str(exc)[:120]}"
