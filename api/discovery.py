"""
mDNS / Bonjour advertising + browsing — phones on the same LAN find this
server without having to type an IP, and (SYN-112 T3) sibling Macs find each
other for P2P sync. Publishes a `_synapse._tcp.local.` service from the
FastAPI lifespan; mobile clients (NsdManager on Android, NetServiceBrowser on
iOS) discover it and offer it as a one-tap pick in Settings. The browser
keeps a live registry of OTHER Synapse instances (filtered by the sync
device_id we advertise) that the peer-sync loop pulls from.
"""

import asyncio
import logging
import os
import socket

from zeroconf import IPVersion, ServiceStateChange
from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf

SERVICE_TYPE = "_synapse._tcp.local."
APP_VERSION = "0.1"

log = logging.getLogger(__name__)

# Live registry of sibling instances, maintained by the browser.
_PEERS: dict[str, dict] = {}
_SELF_DEVICE_ID: str | None = None


def _local_ip() -> str:
    """Best-effort primary LAN IPv4 — UDP socket trick, no packet sent."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _self_device_id() -> str | None:
    """The core sync engine's device id — advertised so peers can tell us
    apart from themselves (mDNS echoes our own service back to us)."""
    global _SELF_DEVICE_ID
    if _SELF_DEVICE_ID is None:
        try:
            from core_store import get_store
            _SELF_DEVICE_ID = get_store().sync_device_id()
        except Exception:  # noqa: BLE001 — never block advertising on this
            _SELF_DEVICE_ID = ""
    return _SELF_DEVICE_ID or None


async def start_advertising(port: int | None = None) -> AsyncZeroconf | None:
    port = port or int(os.environ.get("SYNAPSE_API_PORT", "8000"))
    hostname = socket.gethostname().split(".")[0]
    ip = _local_ip()
    props = {"version": APP_VERSION, "host": hostname}
    dev = _self_device_id()
    if dev:
        props["device_id"] = dev
    info = AsyncServiceInfo(
        type_=SERVICE_TYPE,
        name=f"Synapse on {hostname}.{SERVICE_TYPE}",
        addresses=[socket.inet_aton(ip)],
        port=port,
        properties=props,
        server=f"{hostname}.local.",
    )
    try:
        azc = AsyncZeroconf(ip_version=IPVersion.V4Only)
        await azc.async_register_service(info)
        log.info("mDNS advertising %s on %s:%d", SERVICE_TYPE, ip, port)
        return azc
    except OSError as e:
        # Port 5353 may be taken (rare on macOS where mDNSResponder owns it via SO_REUSEPORT,
        # but possible in containers). Keep the API working anyway.
        log.warning("mDNS advertising disabled: %s", e)
        return None


async def stop_advertising(azc: AsyncZeroconf | None) -> None:
    if azc is None:
        return
    await azc.async_unregister_all_services()
    await azc.async_close()


# ── Peer browsing (SYN-112 T3) ───────────────────────────────────────────────

async def _resolve_peer(zc, service_type: str, name: str) -> None:
    info = AsyncServiceInfo(service_type, name)
    if not await info.async_request(zc, 3000):
        return
    props: dict[str, str] = {}
    for k, v in (info.properties or {}).items():
        try:
            props[k.decode()] = v.decode() if v is not None else ""
        except Exception:  # noqa: BLE001
            continue
    addresses = info.parsed_addresses(IPVersion.V4Only)
    if not addresses or not info.port:
        return
    dev = props.get("device_id")
    if dev and dev == _self_device_id():
        return  # our own advertisement echoed back
    _PEERS[name] = {
        "name": name,
        "url": f"http://{addresses[0]}:{info.port}",
        "device_id": dev,
        "host": props.get("host"),
    }
    log.info("mDNS peer discovered: %s → %s", name, _PEERS[name]["url"])


def _on_service_state_change(zeroconf, service_type, name, state_change) -> None:
    if state_change is ServiceStateChange.Removed:
        if _PEERS.pop(name, None) is not None:
            log.info("mDNS peer left: %s", name)
        return
    asyncio.ensure_future(_resolve_peer(zeroconf, service_type, name))


async def start_browsing(azc: AsyncZeroconf | None) -> AsyncServiceBrowser | None:
    """Watch the LAN for sibling Synapse instances (peer-sync candidates)."""
    if azc is None:
        return None
    try:
        return AsyncServiceBrowser(
            azc.zeroconf, SERVICE_TYPE, handlers=[_on_service_state_change]
        )
    except Exception as e:  # noqa: BLE001
        log.warning("mDNS browsing disabled: %s", e)
        return None


async def stop_browsing(browser: AsyncServiceBrowser | None) -> None:
    if browser is None:
        return
    await browser.async_cancel()


def discovered_peers() -> list[dict]:
    """Instances found on the LAN that are not us (device_id-filtered when we
    could not resolve theirs, the sync client's self-check still catches it)."""
    return [dict(p) for p in _PEERS.values()]
