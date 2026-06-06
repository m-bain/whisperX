// Controllable stand-in for the browser EventSource, for unit-testing the SSE
// consumers in static/sse.js. Records every instance so a test can grab the one
// a function just opened, drive messages into it (`emit`), and assert it was
// closed. Registered as the global `EventSource` by tests/setup.ts.

export class FakeEventSource {
  static instances: FakeEventSource[] = [];
  static reset(): void {
    FakeEventSource.instances = [];
  }
  /** The most recently opened stream (what the function under test just created). */
  static last(): FakeEventSource {
    const all = FakeEventSource.instances;
    if (all.length === 0) throw new Error("no EventSource was opened");
    return all[all.length - 1];
  }

  url: string;
  onmessage: ((e: { data: string }) => void) | null = null;
  closed = false;

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  close(): void {
    this.closed = true;
  }

  /** Deliver a raw SSE `data:` payload (a string, as the browser would). */
  emit(data: string): void {
    this.onmessage?.({ data });
  }

  /** Convenience: JSON-encode an object and deliver it. */
  emitJSON(obj: unknown): void {
    this.emit(JSON.stringify(obj));
  }
}
