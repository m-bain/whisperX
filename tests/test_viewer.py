import csv
import json
import threading
import urllib.error
import urllib.request
from urllib.parse import urlencode
from pathlib import Path

from whisperx.viewer import ViewerApp, ViewerStore, make_server


def _write_library(tmp_path: Path) -> tuple[Path, Path]:
    library = tmp_path / "_ai_library"
    transcript_dir = library / "transcripts" / "Day 1" / "C0001"
    transcript_dir.mkdir(parents=True)
    media = tmp_path / "Day 1" / "C0001.MP4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"0123456789abcdef")
    transcript = transcript_dir / "C0001.json"
    transcript.write_text(
        json.dumps(
            {
                "language": "zh",
                "segments": [
                    {
                        "start": 1.25,
                        "end": 4.5,
                        "speaker": "SPEAKER_00",
                        "text": "AI 可以帮我做基础工作",
                        "avg_logprob": -0.2,
                    },
                    {
                        "start": 5.0,
                        "end": 8.75,
                        "text": "但是味道还是人来判断",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with (library / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file", "relative_file", "output_dir", "json", "status"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file": str(media),
                "relative_file": "Day 1/C0001.MP4",
                "output_dir": str(transcript_dir),
                "json": str(transcript),
                "status": "done",
            }
        )
        writer.writerow(
            {
                "file": str(tmp_path / "Day 1" / "C0002.MP4"),
                "relative_file": "Day 1/C0002.MP4",
                "output_dir": str(library / "transcripts" / "Day 1" / "C0002"),
                "json": str(library / "transcripts" / "Day 1" / "C0002" / "C0002.json"),
                "status": "pending",
            }
        )
    return library, media


def _json_request(url: str, method: str = "GET", payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def test_store_indexes_manifest_and_segments(tmp_path):
    library, media = _write_library(tmp_path)
    store = ViewerStore(library)

    summary = store.rescan()
    assert summary["counts"]["files"] == 2
    assert summary["counts"]["segments"] == 2
    assert summary["counts"]["statuses"] == {"done": 1, "pending": 1}

    segments = store.list_segments({"q": "基础"})["segments"]
    assert len(segments) == 1
    assert segments[0]["source_path"] == str(media)
    assert segments[0]["speaker"] == "SPEAKER_00"
    assert segments[0]["language"] == "zh"


def test_selects_are_persisted_and_exported(tmp_path):
    library, _media = _write_library(tmp_path)
    store = ViewerStore(library)
    store.rescan()
    segment = store.list_segments({"q": "味道"})["segments"][0]

    select = store.upsert_select(
        {
            "segment_id": segment["id"],
            "adjusted_start": 5.25,
            "adjusted_end": 8.5,
            "status": "good",
            "tags": ["food", "human judgment"],
            "theme": "AI lacks taste",
            "hook_strength": 5,
            "notes": "Strong contrast line",
        }
    )
    assert select["status"] == "good"
    assert select["tags"] == ["food", "human judgment"]

    export_path = store.export_csv()
    with export_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["relative_file"] == "Day 1/C0001.MP4"
    assert rows[0]["start"] == "5.250"
    assert rows[0]["end"] == "8.500"
    assert rows[0]["theme"] == "AI lacks taste"
    assert rows[0]["hook_strength"] == "5"
    assert rows[0]["tags"] == "food,human judgment"


def test_rescan_removes_stale_segments(tmp_path):
    library, _media = _write_library(tmp_path)
    store = ViewerStore(library)
    store.rescan()
    transcript = library / "transcripts" / "Day 1" / "C0001" / "C0001.json"
    transcript.write_text(
        json.dumps({"language": "zh", "segments": [{"start": 0, "end": 1, "text": "only one"}]}),
        encoding="utf-8",
    )

    summary = store.rescan()
    assert summary["counts"]["segments"] == 1
    assert len(store.list_segments({})["segments"]) == 1


def test_http_api_and_range_video_stream(tmp_path):
    library, _media = _write_library(tmp_path)
    app = ViewerApp(ViewerStore(library), rescan_interval=0)
    server = make_server("127.0.0.1", 0, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        library_payload = _json_request(f"{base}/api/library")
        assert library_payload["counts"]["files"] == 2

        segments_payload = _json_request(f"{base}/api/segments?{urlencode({'q': '基础'})}")
        segment = segments_payload["segments"][0]
        created = _json_request(
            f"{base}/api/selects",
            method="POST",
            payload={"segment_id": segment["id"], "status": "maybe", "hook_strength": 3},
        )
        assert created["select"]["status"] == "maybe"

        export_response = urllib.request.urlopen(f"{base}/api/export/selects.csv", timeout=5)
        assert export_response.status == 200
        assert b"Day 1/C0001.MP4" in export_response.read()

        request = urllib.request.Request(
            f"{base}/api/video/{segment['file_id']}",
            headers={"Range": "bytes=2-5"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == "bytes 2-5/16"
            assert response.read() == b"2345"

        try:
            urllib.request.urlopen(f"{base}/api/video/not-a-file", timeout=5)
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("expected non-manifest media request to fail")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
