// Unit tests for the client-side SSE primitives in static/sse.js:
//   openSSE          — EventSource + JSON-parse guard + auto-reconnect
//   sseSwap          — swap a rendered `d.html` fragment into a target + htmx.process
//   watchBackupConnect — discover [data-backup-connecting] and stream the OAuth result
//
// Runs against happy-dom (real querySelector / outerHTML / innerHTML / dataset)
// and a controllable FakeEventSource (see tests/setup.ts + tests/fake-eventsource.ts).

import { beforeEach, describe, expect, mock, test } from "bun:test";

import { FakeEventSource } from "./fake-eventsource";
import { openSSE, sseSwap, watchBackupConnect } from "./load-sse";

let processSpy: ReturnType<typeof mock>;

beforeEach(() => {
  FakeEventSource.reset();
  document.body.innerHTML = "";
  processSpy = mock(() => {});
  // htmx is optional on the page; give it a spy so we can assert re-binding.
  (window as any).htmx = { process: processSpy };
});

// --- openSSE -----------------------------------------------------------------

describe("openSSE", () => {
  test("opens an EventSource to the url and returns it", () => {
    const es = openSSE("/x/events", () => {});
    expect(FakeEventSource.instances).toHaveLength(1);
    expect(FakeEventSource.last().url).toBe("/x/events");
    expect(es).toBe(FakeEventSource.last());
  });

  test("delivers parsed JSON and the EventSource to onData", () => {
    const seen: any[] = [];
    const es = openSSE("/x/events", (d, src) => seen.push([d, src]));
    FakeEventSource.last().emitJSON({ stage: "aligning", eta: 3 });
    expect(seen).toHaveLength(1);
    expect(seen[0][0]).toEqual({ stage: "aligning", eta: 3 });
    expect(seen[0][1]).toBe(es);
  });

  test("drops a malformed frame without calling onData or throwing", () => {
    const onData = mock(() => {});
    openSSE("/x/events", onData);
    expect(() => FakeEventSource.last().emit("not json{")).not.toThrow();
    expect(onData).not.toHaveBeenCalled();
  });

  test("delivers multiple frames in order", () => {
    const order: number[] = [];
    openSSE("/x/events", (d) => order.push(d.i));
    for (let i = 0; i < 3; i++) FakeEventSource.last().emitJSON({ i });
    expect(order).toEqual([0, 1, 2]);
  });
});

// --- sseSwap (persistent: no terminal) ---------------------------------------

describe("sseSwap (persistent)", () => {
  test("swaps d.html into the target via outerHTML and re-binds with htmx.process", () => {
    document.body.innerHTML = `<div id="card">old</div>`;
    sseSwap("/backup/status/events", { target: "#card" });
    FakeEventSource.last().emitJSON({ html: `<div id="card">new</div>` });
    expect(document.querySelector("#card")!.textContent).toBe("new");
    expect(processSpy).toHaveBeenCalledTimes(1);
  });

  test("swap:'inner' sets innerHTML, keeping the target element", () => {
    document.body.innerHTML = `<div id="card">old</div>`;
    const before = document.querySelector("#card");
    sseSwap("/x", { target: "#card", swap: "inner" });
    FakeEventSource.last().emitJSON({ html: "<span>inner</span>" });
    expect(document.querySelector("#card")).toBe(before); // same node, content replaced
    expect(document.querySelector("#card span")!.textContent).toBe("inner");
  });

  test("ignores an event with no html", () => {
    document.body.innerHTML = `<div id="card">old</div>`;
    sseSwap("/x", { target: "#card" });
    FakeEventSource.last().emitJSON({ state: "idle" });
    expect(document.querySelector("#card")!.textContent).toBe("old");
    expect(processSpy).not.toHaveBeenCalled();
  });

  test("is a no-op when the target is missing (no throw)", () => {
    sseSwap("/x", { target: "#nope" });
    expect(() => FakeEventSource.last().emitJSON({ html: "<div>x</div>" })).not.toThrow();
    expect(processSpy).not.toHaveBeenCalled();
  });

  test("does not close the stream (persistent) and swaps on every event", () => {
    document.body.innerHTML = `<div id="card">old</div>`;
    sseSwap("/x", { target: "#card" });
    const es = FakeEventSource.last();
    es.emitJSON({ html: `<div id="card">one</div>` });
    es.emitJSON({ html: `<div id="card">two</div>` });
    expect(es.closed).toBe(false);
    expect(document.querySelector("#card")!.textContent).toBe("two");
  });
});

// --- sseSwap (terminal) ------------------------------------------------------

describe("sseSwap (terminal)", () => {
  const terminal = (d: any) => d.status === "linked" || d.status === "error";

  test("ignores a non-terminal event (no swap, stream stays open)", () => {
    document.body.innerHTML = `<div id="card">old</div>`;
    sseSwap("/backup/events", { target: "#card", terminal });
    const es = FakeEventSource.last();
    es.emitJSON({ status: "pending" });
    expect(es.closed).toBe(false);
    expect(document.querySelector("#card")!.textContent).toBe("old");
  });

  test("closes the stream and swaps on the terminal event", () => {
    document.body.innerHTML = `<div id="card">old</div>`;
    sseSwap("/backup/events", { target: "#card", terminal });
    const es = FakeEventSource.last();
    es.emitJSON({ status: "linked", html: `<div id="card">linked</div>` });
    expect(es.closed).toBe(true);
    expect(document.querySelector("#card")!.textContent).toBe("linked");
  });

  test("closes even when the terminal event carries no html", () => {
    document.body.innerHTML = `<div id="card">old</div>`;
    sseSwap("/backup/events", { target: "#card", terminal });
    const es = FakeEventSource.last();
    es.emitJSON({ status: "error" });
    expect(es.closed).toBe(true);
    expect(document.querySelector("#card")!.textContent).toBe("old");
  });
});

// --- watchBackupConnect ------------------------------------------------------

describe("watchBackupConnect", () => {
  test("streams /backup/events and swaps the card on a terminal event", () => {
    document.body.innerHTML = `
      <div id="root">
        <div id="conn" data-backup-connecting data-backup-target="#card" data-backup-swap="inner"></div>
      </div>
      <div id="card">old</div>`;
    watchBackupConnect(document.getElementById("root"));
    expect(FakeEventSource.last().url).toBe("/backup/events");
    FakeEventSource.last().emitJSON({ status: "linked", html: "<span>done</span>" });
    expect(document.querySelector("#card span")!.textContent).toBe("done");
  });

  test("marks the element watched and is idempotent (no second stream)", () => {
    document.body.innerHTML = `
      <div id="conn" data-backup-connecting data-backup-target="#card" data-backup-swap="outer"></div>
      <div id="card">old</div>`;
    const root = document.body;
    watchBackupConnect(root);
    expect(document.getElementById("conn")!.dataset.backupWatching).toBe("1");
    watchBackupConnect(root); // already watching
    expect(FakeEventSource.instances).toHaveLength(1);
  });

  test("accepts the connecting element itself as root (matches)", () => {
    document.body.innerHTML = `
      <div id="conn" data-backup-connecting data-backup-target="#card" data-backup-swap="outer"></div>
      <div id="card">old</div>`;
    watchBackupConnect(document.getElementById("conn"));
    expect(FakeEventSource.instances).toHaveLength(1);
    expect(FakeEventSource.last().url).toBe("/backup/events");
  });

  test("is a no-op when there is no connecting element", () => {
    document.body.innerHTML = `<div id="root"></div>`;
    watchBackupConnect(document.getElementById("root"));
    expect(FakeEventSource.instances).toHaveLength(0);
  });
});
