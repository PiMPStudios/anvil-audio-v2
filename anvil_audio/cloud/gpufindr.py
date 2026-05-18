"""GPUFindr catalog helpers for cloud preflight searches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

GPUFINDR_BASE_URL = "https://gpufindr.com"


@dataclass(frozen=True, slots=True)
class GPUOffer:
    """Normalized GPUFindr offer."""

    id: str
    source: str
    location: str
    name: str
    num_gpus: int
    vram_gb: float
    total_cost_ph: float
    reliability: float
    flops_per_dollar_ph: float
    gpu_mem_bw_gbps: float
    url: str


@dataclass(frozen=True, slots=True)
class GPUFindrSearch:
    """Search filters for GPUFindr offers."""

    gpu: str = ""
    source: str = ""
    location: str = ""
    max_price: float | None = None
    min_vram_gb: float = 0.0
    min_gpus: int = 1
    min_reliability: float = 0.0
    sort: str = "total_cost_ph.asc"
    limit: int = 25
    api_limit: int = 1000
    max_pages: int = 5
    base_url: str = GPUFINDR_BASE_URL


def build_gpufindr_url(search: GPUFindrSearch, *, offset: int = 0) -> str:
    """Build the GPUFindr `/gpus` URL for remote-side filters."""
    params: dict[str, str | int | float] = {
        "sort": search.sort,
        "limit": max(1, min(search.api_limit, 1000)),
    }
    if offset:
        params["offset"] = offset
    if search.source:
        params["source"] = search.source
    if search.location:
        params["location"] = search.location
    if search.max_price is not None:
        params["max_price"] = search.max_price
    query = urlencode(params)
    return f"{search.base_url.rstrip('/')}/gpus?{query}"


def fetch_gpufindr_offers(search: GPUFindrSearch) -> list[GPUOffer]:
    """Fetch, normalize, and locally filter GPUFindr offers."""
    all_offers: list[GPUOffer] = []
    page_size = max(1, min(search.api_limit, 1000))
    for page in range(max(1, search.max_pages)):
        payload = _fetch_gpufindr_page(search, offset=page * page_size)
        all_offers.extend(
            _offer_from_payload(item) for item in payload if isinstance(item, dict)
        )
        if len(payload) < page_size:
            break
    return filter_gpufindr_offers(all_offers, search)


def _fetch_gpufindr_page(search: GPUFindrSearch, *, offset: int) -> list[Any]:
    url = build_gpufindr_url(search, offset=offset)
    request = Request(url, headers={"User-Agent": "anvil-audio/1.0"})
    try:
        with urlopen(request, timeout=20) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"GPUFindr request failed: HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"GPUFindr request failed: {exc.reason}") from exc

    if not isinstance(payload, list):
        raise RuntimeError("GPUFindr returned an unexpected payload.")
    return payload


def filter_gpufindr_offers(
    offers: list[GPUOffer], search: GPUFindrSearch
) -> list[GPUOffer]:
    """Apply Anvil-side filters and return the display-limited offer list."""
    gpu_query = search.gpu.strip().lower()
    filtered = []
    for offer in offers:
        if gpu_query and gpu_query not in offer.name.lower():
            continue
        if offer.vram_gb < search.min_vram_gb:
            continue
        if offer.num_gpus < search.min_gpus:
            continue
        if offer.reliability < search.min_reliability:
            continue
        filtered.append(offer)
    return filtered[: max(1, search.limit)]


def _offer_from_payload(payload: dict[str, Any]) -> GPUOffer:
    return GPUOffer(
        id=str(payload.get("id") or ""),
        source=str(payload.get("source") or ""),
        location=str(payload.get("location") or ""),
        name=str(payload.get("name") or ""),
        num_gpus=_as_int(payload.get("num_gpus"), default=1),
        vram_gb=_as_float(payload.get("vram_mb"), default=0.0) / 1024.0,
        total_cost_ph=_as_float(payload.get("total_cost_ph"), default=0.0),
        reliability=_as_float(payload.get("reliability"), default=0.0),
        flops_per_dollar_ph=_as_float(
            payload.get("flops_per_dollar_ph"), default=0.0
        ),
        gpu_mem_bw_gbps=_as_float(payload.get("gpu_mem_bw_gbps"), default=0.0),
        url=str(payload.get("url") or ""),
    )


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
