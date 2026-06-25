"""Single place that builds the Anthropic client (SYN-105 — fuel-proxy seam).

Centralised on purpose: the closed-beta "fuel proxy" (where I lend testers my
Anthropic credits) is meant to be *disposable*. The entire integration lives in
this one file, so unplugging it = stop issuing `syn-fuel-` tokens (or delete the
fuel branch below). Nothing else in the backend knows the proxy exists.

  - normal key  (`sk-ant-…`)   → direct client, the key is used as-is.
  - fuel token  (`syn-fuel-…`) → client pointed at the fuel proxy. The token is
    carried in the `x-synapse-token` header; the real Anthropic key lives only on
    the proxy. `api_key` is a placeholder the proxy ignores.

The proxy endpoint is `SYNAPSE_FUEL_BASE_URL` (baked into the beta build). The
SDK appends `/v1/messages`, so this must be the origin without a path.
"""
import os

import anthropic

from config_store import get_anthropic_key

_FUEL_PREFIX = "syn-fuel-"

# Closed-beta proxy endpoint, baked in so a tester only has to paste the token
# (the URL is not a secret — only the token is). `SYNAPSE_FUEL_BASE_URL` overrides,
# and setting it empty disables the fuel path. Only consulted for syn-fuel- tokens,
# so a normal sk-ant- key (e.g. the Mac mini) is unaffected.
_DEFAULT_FUEL_BASE_URL = "https://synapse-fuel-proxy.alexis-raitano.workers.dev"


def is_fuel_token(key: str | None) -> bool:
    return bool(key) and key.startswith(_FUEL_PREFIX)


def _fuel_base_url() -> str:
    return os.environ.get("SYNAPSE_FUEL_BASE_URL", _DEFAULT_FUEL_BASE_URL).rstrip("/")


def get_client() -> anthropic.Anthropic:
    """Build the Anthropic client from the configured key. Raises if no key."""
    key = get_anthropic_key()
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY manquante — exporte-la (export ANTHROPIC_API_KEY=sk-ant-...), "
            "mets-la dans .env, ou règle-la depuis l'app (Réglages → Clé Anthropic API)."
        )
    if is_fuel_token(key):
        base_url = _fuel_base_url()
        if not base_url:
            raise EnvironmentError(
                "Token fuel détecté mais SYNAPSE_FUEL_BASE_URL n'est pas configurée."
            )
        return anthropic.Anthropic(
            api_key="placeholder-real-key-lives-on-the-proxy",
            base_url=base_url,
            default_headers={"x-synapse-token": key},
        )
    return anthropic.Anthropic(api_key=key)


def get_client_or_none() -> "anthropic.Anthropic | None":
    """For callers that treat 'no key' as 'feature disabled' rather than an error."""
    try:
        return get_client()
    except EnvironmentError:
        return None
