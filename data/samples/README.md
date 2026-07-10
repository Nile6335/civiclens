# Bundled sample corpus

A tiny, fully public corpus so `make demo`, unit tests, and CI never depend on network
access. One real meeting of the Mesa (AZ) City Council:

| File | What | Provenance (fetched 2026-07-10) |
|---|---|---|
| `mesa_council_2026-04-06.en.vtt` | YouTube auto-captions for the 23-minute City Council meeting of 2026-04-06 | `yt-dlp --write-subs --write-auto-subs --sub-langs en --sub-format vtt --skip-download https://www.youtube.com/watch?v=4Ey7MKj_n7Y` (channel: City of Mesa) |
| `mesa_council_2026-04-06_agenda.pdf` | Official 6-page meeting agenda | Legistar event 4474 `EventAgendaFile`: <https://legistar1.granicus.com/Mesa/meetings/2026/4/4474_A_City_Council_26-04-06_Meeting_Agenda.pdf> |
| `mesa_council_2026-04-06_agenda_items.csv` | The meeting's 33 agenda items (number, sequence, matter file, type, title) | Legistar Web API: `https://webapi.legistar.com/v1/mesa/events/4474/eventitems` |

Meeting metadata: city=`mesa`, meeting_id=`4474`, date=`2026-04-06`,
video=<https://www.youtube.com/watch?v=4Ey7MKj_n7Y>.

All content is public government record published by the City of Mesa via Granicus/Legistar
and YouTube. The VTT deliberately keeps YouTube's raw auto-caption artifacts (rolling
duplicate lines, inline `<c>` timing tags, HTML entities) — the parser is tested against
the real thing, not a sanitized fixture.
