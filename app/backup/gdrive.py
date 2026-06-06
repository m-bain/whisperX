"""Google Drive implementation of :class:`StorageBackend`.

Maps the abstract object-store contract onto Drive:

    <backup folder>/manifest.json
    <backup folder>/objects/<sha256>

Uses the ``drive.file`` scope, so ``files().list`` only ever returns files this
app created — the folder lookup can't see (or clobber) the user's other data.

``google.*`` / ``googleapiclient`` imports are lazy and confined here, so the
package imports fine without the ``gdrive`` extra; only constructing/using this
backend needs it.
"""

from __future__ import annotations

import io
import logging
from typing import BinaryIO

from app.backup import oauth
from app.backup.backend import StorageBackend
from app.backup.manifest import Manifest

logger = logging.getLogger(__name__)

_FOLDER_MIME = "application/vnd.google-apps.folder"
_MANIFEST_NAME = "manifest.json"


def _escape(name: str) -> str:
    return name.replace("\\", "\\\\").replace("'", "\\'")


class GDriveBackend(StorageBackend):
    name = "gdrive"

    def __init__(self, folder_name: str = "Manuscript Backup"):
        self._folder_name = folder_name
        self._service = None
        self._root_id: str | None = None
        self._objects_id: str | None = None

    def set_folder(self, name: str) -> None:
        """Point at a different root folder; drop the cached folder-id lookups so
        the next op re-resolves (or creates) under the new name."""
        name = (name or "").strip()
        if not name or name == self._folder_name:
            return
        self._folder_name = name
        self._root_id = None
        self._objects_id = None

    # --- linkage / lazy service -------------------------------------------
    def is_linked(self) -> bool:
        return oauth.is_linked()

    def _svc(self):
        if self._service is None:
            from googleapiclient.discovery import build
            creds = oauth.load_credentials()
            if creds is None:
                raise RuntimeError("Google Drive is not linked")
            self._service = build("drive", "v3", credentials=creds,
                                  cache_discovery=False)
        return self._service

    def _find_child(self, name: str, parent_id: str | None, *, folder: bool):
        q = [f"name = '{_escape(name)}'", "trashed = false"]
        if folder:
            q.append(f"mimeType = '{_FOLDER_MIME}'")
        if parent_id:
            q.append(f"'{parent_id}' in parents")
        res = self._svc().files().list(
            q=" and ".join(q), spaces="drive",
            fields="files(id, name)", pageSize=1,
        ).execute()
        files = res.get("files", [])
        return files[0]["id"] if files else None

    def _ensure_folder(self, name: str, parent_id: str | None) -> str:
        existing = self._find_child(name, parent_id, folder=True)
        if existing:
            return existing
        meta = {"name": name, "mimeType": _FOLDER_MIME}
        if parent_id:
            meta["parents"] = [parent_id]
        created = self._svc().files().create(body=meta, fields="id").execute()
        return created["id"]

    def _root(self) -> str:
        if self._root_id is None:
            self._root_id = self._ensure_folder(self._folder_name, None)
        return self._root_id

    def _objects(self) -> str:
        if self._objects_id is None:
            self._objects_id = self._ensure_folder("objects", self._root())
        return self._objects_id

    # --- manifest pointer --------------------------------------------------
    def read_manifest(self) -> Manifest | None:
        fid = self._find_child(_MANIFEST_NAME, self._root(), folder=False)
        if not fid:
            return None
        data = self._download(fid)
        return Manifest.from_json(data)

    def write_manifest(self, manifest: Manifest) -> None:
        from googleapiclient.http import MediaIoBaseUpload
        body = io.BytesIO(manifest.to_json().encode("utf-8"))
        media = MediaIoBaseUpload(body, mimetype="application/json", resumable=False)
        existing = self._find_child(_MANIFEST_NAME, self._root(), folder=False)
        if existing:  # single-request content replace == atomic server-side
            self._svc().files().update(fileId=existing, media_body=media).execute()
        else:
            meta = {"name": _MANIFEST_NAME, "parents": [self._root()]}
            self._svc().files().create(body=meta, media_body=media,
                                       fields="id").execute()

    # --- object store ------------------------------------------------------
    def has_object(self, key: str) -> bool:
        return self._find_child(key, self._objects(), folder=False) is not None

    def put_object(self, key: str, data: BinaryIO) -> None:
        if self.has_object(key):
            return  # content-addressed => already identical
        from googleapiclient.http import MediaIoBaseUpload
        media = MediaIoBaseUpload(data, mimetype="application/octet-stream",
                                  resumable=True)
        meta = {"name": key, "parents": [self._objects()]}
        self._svc().files().create(body=meta, media_body=media,
                                   fields="id").execute()

    def get_object(self, key: str) -> bytes:
        fid = self._find_child(key, self._objects(), folder=False)
        if not fid:
            raise KeyError(key)
        return self._download(fid)

    def delete_object(self, key: str) -> None:
        fid = self._find_child(key, self._objects(), folder=False)
        if fid:
            self._svc().files().delete(fileId=fid).execute()

    def list_objects(self) -> set[str]:
        keys: set[str] = set()
        page_token = None
        while True:
            res = self._svc().files().list(
                q=f"'{self._objects()}' in parents and trashed = false",
                spaces="drive", fields="nextPageToken, files(name)",
                pageToken=page_token, pageSize=1000,
            ).execute()
            keys.update(f["name"] for f in res.get("files", []))
            page_token = res.get("nextPageToken")
            if not page_token:
                break
        return keys

    # --- helpers -----------------------------------------------------------
    def _download(self, file_id: str) -> bytes:
        from googleapiclient.http import MediaIoBaseDownload
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, self._svc().files().get_media(fileId=file_id))
        done = False
        while not done:
            _status, done = downloader.next_chunk()
        return buf.getvalue()
