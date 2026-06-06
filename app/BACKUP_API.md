# Backup / Restore — Frontend Brief

How to drive the cloud-backup backend from the UI. Backend mirrors the local
data (DB + per-session artifacts) to a remote (Google Drive). All endpoints
return JSON.

> **Heads-up on timing.** `/backup/now`, `/backup/restore`, `/backup/link` and the
> bootstrap endpoints are **synchronous** — the HTTP request stays open until the
> operation finishes (a backup uploads every changed file). Show a spinner / a
> persistent label, don't expect an instant response, and poll `/backup/status`
> (or read its `state`) for live progress from a *separate* request.

---

## The one thing to render: `state`

Everything keys off a single state string from `GET /backup/status`:

| `state`       | Meaning                                  | What the UI should show |
|---------------|------------------------------------------|-------------------------|
| `disabled`    | Backup not configured on the server      | Hide the feature, or "not available" |
| `unlinked`    | Configured but no account linked         | **"Link Google Drive"** button |
| `idle`        | Linked, everything backed up             | "Up to date ✓", offer Back up now / Restore |
| `dirty`       | Linked, local changes not yet pushed     | "Changes not backed up", highlight **Back up now** |
| `backing_up`  | A backup is running                      | Spinner "Backing up…" |
| `restoring`   | A restore is running                     | Spinner "Restoring…" |
| `error`       | Last op failed (see `last_error`)        | Error banner with `last_error`, allow retry |

`backing_up` / `restoring` are observable by polling `/backup/status` *while* a
`/backup/now` or `/backup/restore` request is in flight.

---

## Endpoints

### `GET /backup/status`
Cheap; poll this (e.g. every 2–3 s while an op runs).
```jsonc
{
  "state": "idle",            // see table above — drive all UI from this
  "linked": true,
  "backend": "gdrive",        // or "local", or null when disabled
  "dirty": false,             // null when not linked
  "last_root": "9af3…",       // merkle root of the last successful backup, or null
  "last_error": null,         // string when state == "error"
  "interval": 900             // periodic auto-backup seconds (0 = off)
}
```

### `POST /backup/link`
Starts the Google OAuth consent flow, then reports what's already on the remote.
> ⚠️ This opens a browser **on the server host** (loopback OAuth) — it suits a
> local/desktop deployment. Confirm with backend before using on a remote server.

Success `200`:
```jsonc
{
  "linked": true,
  "remote": {                 // what was found in the Drive folder
    "exists": false,          // false = nothing there yet (fresh)
    "generation": 0,
    "entries": 0,
    "total_size": 0,
    "created_at": ""
  }
}
```
Failure `400`: `{ "error": "GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET are not set …" }`

**After linking, branch on `remote.exists`:**
- `false` → fresh remote. Just call `POST /backup/now` to seed the first backup.
- `true` → the remote already has a backup. **Ask the user to choose** (this is a
  destructive fork — see below), then call one of the bootstrap endpoints.

### `POST /backup/bootstrap/adopt`  — "Load my existing backup"
Pulls the remote down, **overwriting local** data. Returns `{ "restored": <fileCount> }`.

### `POST /backup/bootstrap/overwrite` — "Start fresh from this device"
Pushes local up, **replacing the remote** backup. Returns the same shape as
`/backup/now`. Use when the user wants this machine to win.

> Present this as an explicit either/or dialog when `remote.exists === true`. One
> wipes local, the other wipes the remote — make the consequence clear.

### `POST /backup/now`
Manual backup (snapshot + upload changed files).
Success `200`:
```jsonc
{
  "uploaded": 3,     // blobs newly pushed this run
  "skipped": 0,      // changed files whose content already existed remotely (dedup)
  "generation": 4,   // increments each successful backup
  "root": "1b7c…"    // new merkle root
}
```
`409`: `{ "error": "backup backend is not linked" }` — link first.

### `POST /backup/restore`
Pulls the remote down to local (mirror). Destructive to local — confirm first.
Success `200`: `{ "restored": <fileCount> }`.
`409`: `{ "error": "cannot restore while transcription jobs are active" }` — a
transcription is running; tell the user to wait. Also `409` if not linked.

> After a restore the in-memory session list changes; refresh the dashboard
> (re-fetch the sessions view) once it returns.

### `POST /backup/unlink`
Drops stored credentials. Returns `{ "linked": false }`. State → `unlinked`.

---

## Recommended flows

**Link (first time)**
1. State is `unlinked` → user clicks "Link Google Drive".
2. `POST /backup/link`. On `400`, show `error`.
3. On success: if `remote.exists` → show the adopt/overwrite choice dialog;
   else → `POST /backup/now`.
4. Poll `/backup/status` until `state` is `idle`/`error`.

**Manual backup**
1. Available when `state` is `idle` or `dirty`.
2. `POST /backup/now`; while waiting, poll `/backup/status` (shows `backing_up`).
3. On return, refresh status. On `409`, surface the message.

**Restore**
1. Confirm (destructive). 2. `POST /backup/restore`; poll status (`restoring`).
3. On `409` "jobs are active", ask the user to retry after transcriptions finish.
4. On success, re-fetch the dashboard.

## Error handling (all endpoints)
- `400` → bad config (only `/backup/link`): show `error`, keep `unlinked`.
- `409` → not linked, or restore blocked by active jobs: show `error`, no state change.
- After any failed op the next `GET /backup/status` returns `state: "error"` with
  `last_error`; a subsequent successful op clears it back to `idle`.

## Notes
- There is **no auto-polling SSE** for backup yet (unlike the transcription job
  progress stream). Use periodic `GET /backup/status` while an op is in flight.
- Periodic auto-backup runs server-side when `interval > 0` and linked; the UI
  doesn't trigger it, but `state` will flip `dirty → backing_up → idle` on its own.
