// Preloaded before the JS test suite (see bunfig.toml). Installs a real DOM
// (happy-dom: document/window/Element with working querySelector + outerHTML/
// innerHTML) and registers the controllable FakeEventSource as the global
// EventSource, so the static/sse.js consumers run against a faithful browser
// environment without a real browser.

import { GlobalRegistrator } from "@happy-dom/global-registrator";

GlobalRegistrator.register();

import { FakeEventSource } from "./fake-eventsource";

// @ts-expect-error — override the (absent in happy-dom) global EventSource.
globalThis.EventSource = FakeEventSource;
