// Shared Server-Sent Events helpers for the Manuscript web app.
//
// Loaded as a plain classic <script> (no bun bundle, no new dependency) by
// base.html *and* the standalone onboarding page, so both share one
// implementation instead of copy-pasting the EventSource plumbing. Every SSE
// consumer in the app goes through these: framing (JSON parse + guard, browser
// auto-reconnect) and the common "swap rendered HTML into a target" tail live
// here in one place.

// Open an EventSource and hand each parsed message to onData(data, es).
// Malformed frames are dropped; the browser auto-reconnects on transient
// errors. Returns the EventSource so callers can close() it on a terminal.
function openSSE(url, onData) {
  const es = new EventSource(url);
  es.onmessage = (e) => { let d; try { d = JSON.parse(e.data); } catch { return; } onData(d, es); };
  return es;
}

// Stream whose events carry a rendered fragment in `d.html` to swap into a
// target element. `terminal(d)` (optional) marks the last event and closes the
// stream; omit it for a persistent stream that swaps on every event. `target`
// is a selector (re-queried after the swap so outerHTML works); `swap` is
// 'outer' (default) or 'inner'. htmx.process re-binds the new DOM (a no-op on
// pages without htmx, e.g. onboarding).
function sseSwap(url, { target, swap = 'outer', terminal } = {}) {
  return openSSE(url, (d, es) => {
    if (terminal) {
      if (!terminal(d)) return;
      es.close();  // close before swapping, matching the original watchers
    }
    if (!d.html) return;
    const dst = document.querySelector(target);
    if (!dst) return;
    if (swap === 'outer') dst.outerHTML = d.html; else dst.innerHTML = d.html;
    window.htmx?.process(document.querySelector(target) || document.body);
  });
}

// Cloud-backup OAuth: the consent flow runs server-side and is non-blocking, so
// the "Connect" action swaps in a [data-backup-connecting] fragment. This opens
// the /backup/events stream and, on the terminal event, swaps the rendered card
// into the per-page target (outer for Settings, inner for onboarding). Idempotent
// via data-backup-watching so a re-fire (htmx:load) won't open a second stream.
function watchBackupConnect(root) {
  const el = root.matches?.('[data-backup-connecting]') ? root
    : root.querySelector?.('[data-backup-connecting]');
  if (!el || el.dataset.backupWatching) return;
  el.dataset.backupWatching = '1';
  sseSwap('/backup/events', {
    target: el.dataset.backupTarget,
    swap: el.dataset.backupSwap,
    terminal: (d) => d.status === 'linked' || d.status === 'error',
  });
}
