"""Unit tests for the Legistar client. No network: httpx.MockTransport fakes the API."""

from datetime import date

import httpx
import pytest

from ingestion.legistar import (
    LegistarClient,
    LegistarError,
    MeetingInfo,
    parse_event,
    pick_responsive_clients,
)

FULL_EVENT = {
    "EventId": 601234,
    "EventGuid": "5E4A1C0B-9C7D-4E7B-8F1A-2D3C4B5A6978",
    "EventBodyId": 136,
    "EventBodyName": "City Council",
    "EventDate": "2026-08-05T00:00:00",
    "EventTime": "2:00 PM",
    "EventLocation": "Council Chamber, City Hall\n600 4th Avenue, Seattle ",
    "EventAgendaFile": "https://legistar2.granicus.com/seattle/meetings/2026/8/agenda.pdf",
    "EventMinutesFile": "https://legistar2.granicus.com/seattle/meetings/2026/8/minutes.pdf",
    "EventVideoPath": "https://www.seattlechannel.org/FullCouncil?videoid=x123456",
    "EventInSiteURL": "https://seattle.legistar.com/MeetingDetail.aspx?ID=601234",
    "EventAgendaStatusName": "Final",
}

NULL_EVENT = {
    "EventId": 700001,
    "EventGuid": "A1B2C3D4-0000-1111-2222-333344445555",
    "EventBodyId": 246,
    "EventBodyName": "Select Budget Committee",
    "EventDate": None,
    "EventTime": None,
    "EventLocation": None,
    "EventAgendaFile": None,
    "EventMinutesFile": None,
    "EventVideoPath": None,
    "EventInSiteURL": None,
    "EventAgendaStatusName": None,
}

SPARSE_EVENT = {"EventId": 700002}


def test_parse_event_full_mapping() -> None:
    info = parse_event("seattle", FULL_EVENT)
    assert info == MeetingInfo(
        client="seattle",
        event_id=601234,
        body_name="City Council",
        date=date(2026, 8, 5),
        time="2:00 PM",
        location="Council Chamber, City Hall\n600 4th Avenue, Seattle",
        agenda_url="https://legistar2.granicus.com/seattle/meetings/2026/8/agenda.pdf",
        minutes_url="https://legistar2.granicus.com/seattle/meetings/2026/8/minutes.pdf",
        video_url="https://www.seattlechannel.org/FullCouncil?videoid=x123456",
        insite_url="https://seattle.legistar.com/MeetingDetail.aspx?ID=601234",
        agenda_status="Final",
    )


def test_parse_event_all_nulls() -> None:
    info = parse_event("seattle", NULL_EVENT)
    assert info.event_id == 700001
    assert info.body_name == "Select Budget Committee"
    assert info.date is None
    for field in (
        "time",
        "location",
        "agenda_url",
        "minutes_url",
        "video_url",
        "insite_url",
        "agenda_status",
    ):
        assert getattr(info, field) is None, field


def test_parse_event_missing_keys_and_bad_date() -> None:
    info = parse_event("oakland", SPARSE_EVENT)
    assert info.client == "oakland"
    assert info.event_id == 700002
    assert info.body_name == ""
    assert info.date is None
    assert parse_event("oakland", {"EventDate": "not-a-date"}).date is None


def _client_with(handler) -> LegistarClient:
    return LegistarClient(transport=httpx.MockTransport(handler))


def test_recent_events_request_and_parsing() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[FULL_EVENT, NULL_EVENT])

    events = _client_with(handler).recent_events("seattle", top=25, before=date(2026, 7, 1))
    assert [e.event_id for e in events] == [601234, 700001]
    assert events[0].date == date(2026, 8, 5)

    (request,) = seen
    assert request.url.path == "/v1/seattle/events"
    assert request.url.params["$top"] == "25"
    assert request.url.params["$orderby"] == "EventDate desc"
    assert request.url.params["$filter"] == "EventDate lt datetime'2026-07-01'"
    raw_query = request.url.query.decode()
    assert " " not in raw_query  # OData params are URL-encoded on the wire
    assert "%24orderby=EventDate+desc" in raw_query
    assert "%24filter=EventDate+lt+datetime%272026-07-01%27" in raw_query


def test_recent_events_omits_filter_without_before() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[])

    assert _client_with(handler).recent_events("seattle") == []
    assert seen[0].url.params["$top"] == "50"
    assert "$filter" not in seen[0].url.params


def test_400_raises_legistar_error_without_retry() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="Unsupported filter")

    with pytest.raises(LegistarError, match="400"):
        _client_with(handler).recent_events("gotham")
    assert calls["n"] == 1


def test_500_then_200_is_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, text="Internal Server Error")
        return httpx.Response(200, json=[FULL_EVENT])

    events = _client_with(handler).recent_events("seattle", top=1)
    assert calls["n"] == 2
    assert len(events) == 1


def test_transport_error_exhausts_retries() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(httpx.ConnectError):
        _client_with(handler).recent_events("seattle", top=1)
    assert calls["n"] == 3


def test_event_items_path_and_note_params() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[{"EventItemId": 1, "EventItemTitle": "CB 121001"}])

    items = _client_with(handler).event_items("seattle", 601234)
    assert items == [{"EventItemId": 1, "EventItemTitle": "CB 121001"}]
    assert seen[0].url.path == "/v1/seattle/events/601234/eventitems"
    assert seen[0].url.params["AgendaNote"] == "1"
    assert seen[0].url.params["MinutesNote"] == "1"


def test_matter_attachments_path() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[{"MatterAttachmentId": 7, "MatterAttachmentName": "A"}])

    attachments = _client_with(handler).matter_attachments("seattle", 98765)
    assert attachments[0]["MatterAttachmentId"] == 7
    assert seen[0].url.path == "/v1/seattle/matters/98765/attachments"


def test_pick_responsive_clients_skips_failures_and_stops_at_need() -> None:
    probed: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        slug = request.url.path.split("/")[2]
        probed.append(slug)
        if slug == "gotham":
            return httpx.Response(400, text="Bad Request")
        return httpx.Response(200, json=[FULL_EVENT])

    chosen = pick_responsive_clients(
        ["gotham", "seattle", "oakland", "springfield"],
        client=_client_with(handler),
        need=2,
    )
    assert chosen == ["seattle", "oakland"]
    assert probed == ["gotham", "seattle", "oakland"]  # springfield never probed
