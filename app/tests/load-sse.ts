// static/sse.js is a classic browser script that defines globals (no exports).
// Load its source and hand back the function declarations so tests can call
// them. The functions' free references to window/document/EventSource resolve to
// the globals the setup preload installed (happy-dom + FakeEventSource).
import { readFileSync } from "node:fs";
import { join } from "node:path";

const src = readFileSync(join(import.meta.dir, "../static/sse.js"), "utf8");
const factory = new Function(
  src + "\nreturn { openSSE, sseSwap, watchBackupConnect };",
);

type OnData = (d: any, es: any) => void;
export const { openSSE, sseSwap, watchBackupConnect } = factory() as {
  openSSE: (url: string, onData: OnData) => any;
  sseSwap: (
    url: string,
    opts: { target: string; swap?: string; terminal?: (d: any) => boolean },
  ) => any;
  watchBackupConnect: (root: any) => void;
};
