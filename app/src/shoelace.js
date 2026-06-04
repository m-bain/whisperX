// Shoelace, cherry-picked to the components actually used in templates/**.
// Loaded as an ES module (web components self-register; load order is not
// critical). Replaces the CDN autoloader from base.html.
import { setBasePath } from "@shoelace-style/shoelace/dist/utilities/base-path.js";

// sl-icon fetches Bootstrap Icons by name at runtime from <basePath>/assets/icons/.
// build.ts copies Shoelace's assets dir to static/vendor/shoelace/ so this resolves locally.
setBasePath("/static/vendor/shoelace");

import "@shoelace-style/shoelace/dist/components/icon/icon.js";
import "@shoelace-style/shoelace/dist/components/button/button.js";
import "@shoelace-style/shoelace/dist/components/input/input.js";
import "@shoelace-style/shoelace/dist/components/select/select.js";
import "@shoelace-style/shoelace/dist/components/option/option.js";
import "@shoelace-style/shoelace/dist/components/dialog/dialog.js";
import "@shoelace-style/shoelace/dist/components/alert/alert.js";
import "@shoelace-style/shoelace/dist/components/spinner/spinner.js";
import "@shoelace-style/shoelace/dist/components/range/range.js";
