// Builds the vendored frontend bundle into static/vendor/ with Bun.
//   cd app && bun install && bun run build
// Output is gitignored; the Dockerfile runs this in a dedicated `assets` stage.
import { cp, rm, mkdir } from "node:fs/promises";

const out = "static/vendor";

await rm(out, { recursive: true, force: true });
await mkdir(out, { recursive: true });

// One Bun.build per entrypoint. NB: mixing JS and CSS entrypoints in a single
// Bun.build call segfaults Bun 1.3.11, so keep them separate.
const builds = await Promise.all([
  // htmx -> classic IIFE (blocking <script>, sets window.htmx).
  Bun.build({ entrypoints: ["src/htmx.js"], outdir: out, format: "iife", minify: true, naming: "[name].js" }),
  // Shoelace components -> ESM module.
  Bun.build({ entrypoints: ["src/shoelace.js"], outdir: out, format: "esm", minify: true, naming: "[name].js" }),
  // Theme + font CSS. Bun bundles the @imports and copies the woff2 files into out.
  Bun.build({ entrypoints: ["src/vendor.css"], outdir: out, minify: true, naming: "[name].[ext]" }),
]);

for (const result of builds) {
  if (!result.success) {
    for (const log of result.logs) console.error(log);
    process.exit(1);
  }
}

// sl-icon loads SVGs by name at runtime (not via import), so the bundler can't
// see them — copy Shoelace's assets dir to <basePath> (see setBasePath in shoelace.js).
await cp(
  "node_modules/@shoelace-style/shoelace/dist/assets",
  `${out}/shoelace/assets`,
  { recursive: true },
);

console.log(`Built vendor bundle -> app/${out}`);
