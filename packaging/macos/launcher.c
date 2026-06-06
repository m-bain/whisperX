// Minimal native launcher for the Manuscript (WhisperX web app) .app bundle.
//
// The .app's CFBundleExecutable must be a signed Mach-O (a shell script can't
// carry the hardened runtime / be notarized cleanly), so this tiny binary just
// sets up the environment and exec()s the embedded Python interpreter to run the
// Flask server. This is the *browser-open PoC* launcher (design (B), step 2 of
// the verification order in MACOS_INSTALLER.md); the Tauri WKWebView shell
// replaces it later.
//
// Layout it assumes (relative to this executable):
//   Contents/MacOS/Manuscript            <- this binary
//   Contents/Resources/python/bin/python3                 (relocatable base interp)
//   Contents/Resources/runtime/lib/<PYTAG>/site-packages  (uv-installed deps)
//   Contents/Resources/app/server.py     (import path: `app.server`)
//   Contents/Resources/bin/ffmpeg
//
// We run the *base* python-build-standalone interpreter (it relocates correctly)
// and put the venv's site-packages on PYTHONPATH. The venv itself can't relocate
// — its pyvenv.cfg `home` is absolute and CPython won't honour a relative one —
// so we don't depend on it at runtime. PYTAG (e.g. "python3.12") is baked in at
// compile time from PYTHON_VERSION (see build.py cmd_launcher).

#ifndef PYTAG
#define PYTAG "python3.12"
#endif

#include <libgen.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  // Resolve this executable's real path (follows symlinks).
  char buf[PATH_MAX];
  uint32_t size = sizeof(buf);
  if (_NSGetExecutablePath(buf, &size) != 0) {
    fprintf(stderr, "launcher: executable path too long\n");
    return 1;
  }
  char exe[PATH_MAX];
  if (realpath(buf, exe) == NULL) {
    perror("launcher: realpath");
    return 1;
  }

  // exe = .../Contents/MacOS/Manuscript  ->  Contents = dirname(dirname(exe))
  char tmp[PATH_MAX];
  strncpy(tmp, exe, sizeof(tmp));
  char *macos_dir = dirname(tmp);          // .../Contents/MacOS
  char macos_copy[PATH_MAX];
  strncpy(macos_copy, macos_dir, sizeof(macos_copy));
  char *contents = dirname(macos_copy);    // .../Contents

  char res[PATH_MAX];
  snprintf(res, sizeof(res), "%s/Resources", contents);

  char python[PATH_MAX];
  snprintf(python, sizeof(python), "%s/python/bin/python3", res);

  char ffbin[PATH_MAX];
  snprintf(ffbin, sizeof(ffbin), "%s/bin", res);

  // PYTHONPATH = venv site-packages + Resources (so `import app` resolves).
  char pythonpath[PATH_MAX * 2];
  snprintf(pythonpath, sizeof(pythonpath),
           "%s/runtime/lib/" PYTAG "/site-packages:%s", res, res);

  // Environment for the server process.
  setenv("WHISPERX_OPEN_BROWSER", "1", 1);  // PoC: open the default browser
  setenv("PYTHONPATH", pythonpath, 1);
  setenv("PYTHONUNBUFFERED", "1", 1);

  // Prepend the bundled ffmpeg to PATH (whisperx shells out to it).
  const char *old_path = getenv("PATH");
  if (old_path == NULL) old_path = "/usr/bin:/bin";
  char new_path[PATH_MAX * 2];
  snprintf(new_path, sizeof(new_path), "%s:%s", ffbin, old_path);
  setenv("PATH", new_path, 1);

  char *const child_argv[] = {python, "-m", "app.server", NULL};
  execv(python, child_argv);

  perror("launcher: execv");  // only reached on failure
  return 127;
}
