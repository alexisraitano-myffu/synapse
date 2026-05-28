"""
mDNS / Bonjour advertising — phones on the same LAN find this server without
having to type an IP. Publishes a `_synapse._tcp.local.` service from the
FastAPI lifespan; mobile clients (NsdManager on Android, NetServiceBrowser on
iOS) discover it and offer it as a one-tap pick in Settings.
"""

import logging
import os
import socket

from zeroconf import IPVersion
from zeroconf.asyncio import AsyncServiceInfo, AsyncZeroconf

SERVICE_TYPE = "_synapse._tcp.local."
APP_VERSION = "0.1"

log = logging.getLogger(__name__)


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


async def start_advertising(port: int | None = None) -> AsyncZeroconf | None:
    port = port or int(os.environ.get("SYNAPSE_API_PORT", "8000"))
    hostname = socket.gethostname().split(".")[0]
    ip = _local_ip()
    info = AsyncServiceInfo(
        type_=SERVICE_TYPE,
        name=f"Synapse on {hostname}.{SERVICE_TYPE}",
        addresses=[socket.inet_aton(ip)],
        port=port,
        properties={"version": APP_VERSION, "host": hostname},
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
