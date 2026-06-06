"""Unit tests for the SSE transport broker (``app.sse.Broker``) and the
``sse_format`` framing helper.

The broker is the in-process pub/sub that bridges background worker threads to
open ``EventSource`` requests. These tests pin its contract: per-channel fan-out
and isolation, subscriber cleanup (no leaks), the drop-not-block behavior on a
full/slow client, and concurrent publish safety. No Flask needed — the broker is
deliberately framework-free (only ``sse_response`` touches Flask, lazily).
"""

from __future__ import annotations

import json
import queue
import threading

from app.sse import Broker, sse_format


# --- fan-out & delivery ------------------------------------------------------

def test_publish_delivers_to_subscriber():
    b = Broker()
    q = b.subscribe("ch")
    b.publish("ch", {"n": 1})
    assert q.get_nowait() == {"n": 1}


def test_publish_fans_out_to_all_subscribers_on_channel():
    b = Broker()
    q1, q2 = b.subscribe("ch"), b.subscribe("ch")
    b.publish("ch", {"hello": "world"})
    assert q1.get_nowait() == {"hello": "world"}
    assert q2.get_nowait() == {"hello": "world"}


def test_channels_are_isolated():
    b = Broker()
    qa = b.subscribe("a")
    qb = b.subscribe("b")
    b.publish("a", {"for": "a"})
    assert qa.get_nowait() == {"for": "a"}
    assert qb.empty()  # the "b" subscriber must not see "a" events


def test_publish_with_no_subscribers_is_a_noop():
    b = Broker()
    b.publish("nobody", {"x": 1})  # must not raise
    # a later subscriber does not receive the earlier (undelivered) event
    q = b.subscribe("nobody")
    assert q.empty()


def test_publish_preserves_order_for_a_subscriber():
    b = Broker()
    q = b.subscribe("ch")
    for i in range(5):
        b.publish("ch", {"i": i})
    assert [q.get_nowait()["i"] for _ in range(5)] == [0, 1, 2, 3, 4]


# --- unsubscribe / cleanup ---------------------------------------------------

def test_unsubscribe_stops_delivery():
    b = Broker()
    q = b.subscribe("ch")
    b.unsubscribe("ch", q)
    b.publish("ch", {"n": 1})
    assert q.empty()


def test_unsubscribe_one_leaves_others_working():
    b = Broker()
    q1, q2 = b.subscribe("ch"), b.subscribe("ch")
    b.unsubscribe("ch", q1)
    b.publish("ch", {"n": 1})
    assert q1.empty()
    assert q2.get_nowait() == {"n": 1}


def test_channel_entry_is_removed_when_last_subscriber_leaves():
    b = Broker()
    q = b.subscribe("ch")
    assert "ch" in b._subs
    b.unsubscribe("ch", q)
    assert "ch" not in b._subs  # no leak of empty channel sets


def test_unsubscribe_unknown_channel_or_queue_is_safe():
    b = Broker()
    b.unsubscribe("missing", queue.Queue())  # never subscribed — must not raise
    q = b.subscribe("ch")
    b.unsubscribe("ch", q)
    b.unsubscribe("ch", q)  # double unsubscribe — must not raise


# --- backpressure: drop, never block the worker ------------------------------

def test_full_queue_drops_without_blocking():
    b = Broker()
    q = b.subscribe("ch")
    # Queue maxsize is 64; overfill it and assert publish neither blocks nor raises.
    for i in range(100):
        b.publish("ch", {"i": i})
    assert q.qsize() == 64
    # The retained events are the first 64 (later ones dropped on a full queue).
    assert q.get_nowait() == {"i": 0}


def test_full_subscriber_does_not_starve_others():
    # The reason publish wraps each put in its own try/except: one dead/slow
    # client (full queue) must not block delivery to healthy subscribers.
    b = Broker()
    slow, fast = b.subscribe("ch"), b.subscribe("ch")
    for _ in range(64):
        slow.put_nowait({"fill": True})  # saturate the slow client directly
    b.publish("ch", {"n": 1})
    assert slow.qsize() == 64            # dropped for the full one
    assert fast.get_nowait() == {"n": 1}  # still delivered to the healthy one


def test_queue_recovers_after_draining():
    # A slow client that catches up (drains) receives subsequent events again.
    b = Broker()
    q = b.subscribe("ch")
    for i in range(64):
        b.publish("ch", {"i": i})       # full
    b.publish("ch", {"i": 64})          # dropped
    for _ in range(64):
        q.get_nowait()                  # client catches up
    b.publish("ch", {"i": 65})
    assert q.get_nowait() == {"i": 65}  # delivery resumes


def test_unsubscribe_with_mismatched_channel_is_safe():
    # Unsubscribing a queue against the wrong channel discards nothing and leaves
    # the real subscription intact.
    b = Broker()
    qa = b.subscribe("a")
    b.subscribe("b")  # keep "b" non-empty so the branch runs
    b.unsubscribe("b", qa)              # qa belongs to "a" — no-op on "b"
    b.publish("a", {"n": 1})
    assert qa.get_nowait() == {"n": 1}  # "a" subscription untouched


def test_event_is_delivered_by_reference_to_all_subscribers():
    # Documents a footgun: subscribers share the same event object (no copy), so a
    # consumer must treat events as read-only. (sse_response only json.dumps them.)
    b = Broker()
    q1, q2 = b.subscribe("ch"), b.subscribe("ch")
    event = {"n": 1}
    b.publish("ch", event)
    assert q1.get_nowait() is q2.get_nowait() is event


# --- concurrency -------------------------------------------------------------

def test_concurrent_publish_delivers_every_event():
    b = Broker()
    q = b.subscribe("ch")
    threads = [
        threading.Thread(target=lambda base=t * 10: [b.publish("ch", {"v": base + k}) for k in range(10)])
        for t in range(5)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    got = sorted(q.get_nowait()["v"] for _ in range(50))  # 5 threads x 10 = 50, under the 64 cap
    assert got == list(range(50))


def test_concurrent_subscribe_unsubscribe_is_safe():
    b = Broker()

    def churn():
        for _ in range(50):
            q = b.subscribe("ch")
            b.publish("ch", {"x": 1})
            b.unsubscribe("ch", q)

    threads = [threading.Thread(target=churn) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert "ch" not in b._subs  # every subscriber cleaned up; no leak


# --- framing -----------------------------------------------------------------

def test_sse_format_shape():
    out = sse_format({"a": 1, "b": "x"})
    assert out.endswith("\n\n")
    assert out.startswith("data: ")
    # the payload between "data: " and the blank line is valid JSON round-tripping the event
    assert json.loads(out[len("data: "):].strip()) == {"a": 1, "b": "x"}
