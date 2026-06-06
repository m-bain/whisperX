// Native Tauri (WKWebView) shell for the Manuscript (WhisperX web app) .app bundle.
//
// This replaces the browser-open PoC launcher (../launcher.c). It folds in all of
// that launcher's responsibilities — resolve the bundle, set up the child env,
// prepend the bundled ffmpeg to PATH, redirect logs, spawn the embedded Python
// interpreter — and adds a native desktop window pointed at the local Flask server.
//
// Design (B)+(D) from ../../MACOS_INSTALLER.md. The interpreter lives in
// Contents/Resources (NOT a Tauri `externalBin` — see landmine 6 / tauri#11992);
// we spawn it ourselves with std::process::Command. build.py compiles this with a
// plain `cargo build --release` and copies the binary to Contents/MacOS/Manuscript,
// so build.py remains the single owner of the .app tree and the deep-sign.
//
// Bundle layout assumed (relative to this executable):
//   Contents/MacOS/Manuscript                              <- this binary
//   Contents/Resources/python/bin/python3                  (relocatable base interp)
//   Contents/Resources/runtime/lib/python3.X/site-packages (uv-installed deps)
//   Contents/Resources/app/server.py                       (import path: app.server)
//   Contents/Resources/bin/ffmpeg
//
// We run the *base* python-build-standalone interpreter (it relocates correctly)
// with the venv's site-packages on PYTHONPATH — the venv itself can't relocate (its
// pyvenv.cfg `home` is absolute). Mirrors the launcher.c rationale.

use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tauri::{RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent};

/// Resolve `Contents/Resources` from the running executable
/// (`Contents/MacOS/Manuscript` -> `Contents/Resources`).
fn resources_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe");
    let exe = fs::canonicalize(&exe).unwrap_or(exe);
    // exe = .../Contents/MacOS/Manuscript
    let macos = exe.parent().expect("MacOS dir");
    let contents = macos.parent().expect("Contents dir");
    contents.join("Resources")
}

/// The `python3.X` tag for the site-packages path, discovered by globbing
/// `runtime/lib/python3.*` (avoids baking PYTHON_VERSION into this binary).
fn pytag(resources: &Path) -> String {
    let libdir = resources.join("runtime").join("lib");
    if let Ok(entries) = fs::read_dir(&libdir) {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with("python3") && e.path().join("site-packages").is_dir() {
                return name;
            }
        }
    }
    // Fallback; should never hit if the runtime phase ran.
    "python3.12".to_string()
}

/// Writable data dir for logs: WHISPERX_DATA_DIR, else the macOS app-support path.
/// Mirrors launcher.c so logs land in the same place the app already uses.
fn data_dir() -> PathBuf {
    if let Ok(d) = std::env::var("WHISPERX_DATA_DIR") {
        if !d.is_empty() {
            return PathBuf::from(d);
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home).join("Library/Application Support/WhisperX")
}

/// Bind an ephemeral localhost port, then release it so Python can grab it via
/// PORT. Deterministic handoff (no runtime-port file race); `_choose_port()` in
/// server.py still falls back gracefully if it's taken in the meantime.
fn pick_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .expect("pick free port")
}

/// Open `<data dir>/manuscript.log` (append) for the child's stdout/stderr.
fn open_log() -> Option<File> {
    let dir = data_dir();
    let _ = fs::create_dir_all(&dir);
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("manuscript.log"))
        .ok()
}

/// Spawn `python3 -m app.server` from the in-bundle interpreter with the env the
/// server needs. Returns the child handle (kept so we can SIGTERM it on quit).
fn spawn_server(resources: &Path, port: u16) -> std::io::Result<Child> {
    let python = resources.join("python/bin/python3");
    let tag = pytag(resources);
    let site = resources.join("runtime/lib").join(&tag).join("site-packages");

    // PYTHONPATH = venv site-packages + Resources (so `import app` resolves).
    let pythonpath = format!("{}:{}", site.display(), resources.display());

    // Prepend the bundled ffmpeg dir to PATH (whisperx shells out to it).
    let ffbin = resources.join("bin");
    let old_path = std::env::var("PATH").unwrap_or_else(|_| "/usr/bin:/bin".into());
    let new_path = format!("{}:{}", ffbin.display(), old_path);

    let mut cmd = Command::new(&python);
    cmd.args(["-m", "app.server"])
        .env("PYTHONPATH", pythonpath)
        .env("PYTHONUNBUFFERED", "1")
        .env("PATH", new_path)
        .env("PORT", port.to_string());
    // Deliberately NOT setting WHISPERX_OPEN_BROWSER — the native window replaces
    // the browser tab.

    if let Some(log) = open_log() {
        match log.try_clone() {
            Ok(err) => {
                cmd.stdout(Stdio::from(log)).stderr(Stdio::from(err));
            }
            Err(_) => {
                cmd.stdout(Stdio::from(log));
            }
        }
    }

    cmd.spawn()
}

/// Poll http://127.0.0.1:<port>/healthz until it returns 200 (Flask is serving) or
/// we give up (~30s). models_ready in the body is ignored — model warm is async and
/// must NOT gate the window.
fn wait_healthz(port: u16) -> bool {
    let url = format!("http://127.0.0.1:{port}/healthz");
    for _ in 0..120 {
        if let Ok(resp) = ureq::get(&url).timeout(Duration::from_millis(800)).call() {
            if resp.status() == 200 {
                return true;
            }
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    false
}

/// SIGTERM the child (graceful `_shutdown()` flushes the WAL + job executor), wait
/// briefly, then SIGKILL as a fallback. Idempotent via the Option::take.
fn terminate(child: &Arc<Mutex<Option<Child>>>) {
    let Some(mut c) = child.lock().unwrap().take() else {
        return;
    };
    unsafe {
        libc::kill(c.id() as i32, libc::SIGTERM);
    }
    for _ in 0..50 {
        match c.try_wait() {
            Ok(Some(_)) => return,
            _ => std::thread::sleep(Duration::from_millis(100)),
        }
    }
    let _ = c.kill();
    let _ = c.wait();
}

fn main() {
    let resources = resources_dir();
    let port = pick_port();

    let child: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(
        match spawn_server(&resources, port) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("manuscript: failed to spawn server: {e}");
                None
            }
        },
    ));

    let child_for_exit = child.clone();
    let target = format!("http://127.0.0.1:{port}/");

    let app = tauri::Builder::default()
        .setup(move |app| {
            // Create the window on the bundled splash page (main thread, in setup),
            // then navigate to the Flask UI once healthz passes — building the
            // webview off the main thread would panic on macOS.
            let win = WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("Manuscript")
                .inner_size(1180.0, 820.0)
                .min_inner_size(820.0, 600.0)
                .build()?;

            let target = target.clone();
            std::thread::spawn(move || {
                if wait_healthz(port) {
                    if let Ok(url) = target.parse() {
                        let _ = win.navigate(url);
                    }
                } else {
                    eprintln!("manuscript: server did not become healthy in time");
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error building tauri app");

    app.run(move |app_handle, event| match event {
        // Closing the (only) window quits the app on macOS rather than lingering in
        // the dock — drives the Exit event below, which SIGTERMs the server.
        RunEvent::WindowEvent {
            event: WindowEvent::CloseRequested { .. },
            ..
        } => {
            app_handle.exit(0);
        }
        RunEvent::Exit => terminate(&child_for_exit),
        _ => {}
    });
}
