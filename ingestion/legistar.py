"""Client for the Legistar Web API (https://webapi.legistar.com/v1/{client}/...).

Legistar hosts agendas/minutes/matters for many cities under a per-city "client" slug
(e.g. seattle, oakland). Responses are JSON lists; queries use OData params. Some
clients reject filters with a 400, so errors must be loud and non-retryable.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

BASE_URL = "https://webapi.legistar.com/v1"
REQUEST_TIMEOUT = 10.0


class LegistarError(Exception):
    """Non-retryable Legistar API failure: a 4xx response or a malformed payload."""


@dataclass
class MeetingInfo:
    """A Legistar event flattened to the fields the ingestion pipeline needs."""

    client: str
    event_id: int
    body_name: str
    date: date | None
    time: str | None
    location: str | None
    agenda_url: str | None
    minutes_url: str | None
    video_url: str | None
    insite_url: str | None
    agenda_status: str | None


def _clean_str(value: object) -> str | None:
    """Return a stripped string, or None for null/blank/non-string values."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _parse_event_date(value: object) -> date | None:
    """Parse an ISO timestamp like '2026-08-05T00:00:00' to a date; None if absent/bad."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        logger.warning("unparseable Legistar EventDate: %r", value)
        return None


def parse_event(client: str, event: dict) -> MeetingInfo:
    """Map a raw Legistar event JSON object to MeetingInfo; every field may be null."""
    return MeetingInfo(
        client=client,
        event_id=int(event.get("EventId") or 0),
        body_name=_clean_str(event.get("EventBodyName")) or "",
        date=_parse_event_date(event.get("EventDate")),
        time=_clean_str(event.get("EventTime")),
        location=_clean_str(event.get("EventLocation")),
        agenda_url=_clean_str(event.get("EventAgendaFile")),
        minutes_url=_clean_str(event.get("EventMinutesFile")),
        video_url=_clean_str(event.get("EventVideoPath")),
        insite_url=_clean_str(event.get("EventInSiteURL")),
        agenda_status=_clean_str(event.get("EventAgendaStatusName")),
    )


class LegistarClient:
    """Thin HTTP client for the Legistar Web API with timeouts and retries."""

    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        self._http = httpx.Client(base_url=BASE_URL, timeout=REQUEST_TIMEOUT, transport=transport)

    def close(self) -> None:
        self._http.close()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.2, max=2),
        reraise=True,
    )
    def _get(self, path: str, params: dict[str, str] | None = None) -> list[dict]:
        response = self._http.get(path, params=params)
        if response.status_code >= 500:
            response.raise_for_status()  # httpx.HTTPStatusError -> retried by tenacity
        if response.status_code >= 400:
            raise LegistarError(
                f"Legistar returned {response.status_code} for {response.request.url}:"
                f" {response.text[:200]}"
            )
        payload = response.json()
        if not isinstance(payload, list):
            raise LegistarError(f"expected a JSON list from {path}, got {type(payload).__name__}")
        return payload

    def recent_events(
        self, client: str, top: int = 50, before: date | None = None
    ) -> list[MeetingInfo]:
        """Most recent events for a client, newest first, optionally before a date."""
        params: dict[str, str] = {"$top": str(top), "$orderby": "EventDate desc"}
        if before is not None:
            params["$filter"] = f"EventDate lt datetime'{before.isoformat()}'"
        events = self._get(f"/{client}/events", params=params)
        return [parse_event(client, event) for event in events]

    def event_items(self, client: str, event_id: int) -> list[dict]:
        """Agenda items for an event, with agenda/minutes note text included."""
        return self._get(
            f"/{client}/events/{event_id}/eventitems",
            params={"AgendaNote": "1", "MinutesNote": "1"},
        )

    def matter_attachments(self, client: str, matter_id: int) -> list[dict]:
        """Attachments (PDFs etc.) for a legislative matter."""
        return self._get(f"/{client}/matters/{matter_id}/attachments")


def pick_responsive_clients(
    candidates: list[str], client: LegistarClient | None = None, need: int = 2
) -> list[str]:
    """Probe candidates with a 1-event query; return the first `need` that respond."""
    api = client or LegistarClient()
    responsive: list[str] = []
    for candidate in candidates:
        if len(responsive) >= need:
            break
        try:
            api.recent_events(candidate, top=1)
        except (LegistarError, httpx.HTTPError) as exc:
            logger.info("legistar client %r not responsive: %s", candidate, exc)
            continue
        responsive.append(candidate)
    return responsive
