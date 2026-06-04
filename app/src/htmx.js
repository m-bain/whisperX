// htmx, bundled as a classic (IIFE) script so it stays a *blocking* <script>
// in base.html — preserving the parse-time availability of window.htmx that the
// inline template scripts (and htmx's own auto-processing) rely on. ES modules
// defer, which would change that timing; see the htmx gotchas in CLAUDE.md.
import htmx from "htmx.org";

window.htmx = htmx;
