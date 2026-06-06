"""Filesystem reference backend.

Implements :class:`StorageBackend` against a local directory. Drives every unit
test (no network, no Google libs) and doubles as a "backup to another disk /
mounted share" option. The on-disk layout is exactly the remote contract:

    <root>/manifest.json
    <root>/objects/<sha256>
"""

from __future__ import annotations

import os
import shutil
from typing import BinaryIO

from app.backup.backend import StorageBackend
from app.backup.manifest import Manifest


class LocalFsBackend(StorageBackend):
    name = "local"

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.objects_dir = os.path.join(self.root, "objects")
        self.manifest_path = os.path.join(self.root, "manifest.json")
        os.makedirs(self.objects_dir, exist_ok=True)

    def is_linked(self) -> bool:
        return os.path.isdir(self.root)

    # --- manifest pointer --------------------------------------------------
    def read_manifest(self) -> Manifest | None:
        if not os.path.exists(self.manifest_path):
            return None
        with open(self.manifest_path, encoding="utf-8") as f:
            return Manifest.from_json(f.read())

    def write_manifest(self, manifest: Manifest) -> None:
        tmp = f"{self.manifest_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(manifest.to_json())
        os.replace(tmp, self.manifest_path)  # atomic commit

    # --- object store ------------------------------------------------------
    def _object_path(self, key: str) -> str:
        return os.path.join(self.objects_dir, key)

    def has_object(self, key: str) -> bool:
        return os.path.exists(self._object_path(key))

    def put_object(self, key: str, data: BinaryIO) -> None:
        dest = self._object_path(key)
        if os.path.exists(dest):  # content-addressed => identical, skip
            return
        tmp = f"{dest}.tmp"
        with open(tmp, "wb") as f:
            shutil.copyfileobj(data, f)
        os.replace(tmp, dest)

    def get_object(self, key: str) -> bytes:
        try:
            with open(self._object_path(key), "rb") as f:
                return f.read()
        except FileNotFoundError as exc:
            raise KeyError(key) from exc

    def delete_object(self, key: str) -> None:
        try:
            os.remove(self._object_path(key))
        except FileNotFoundError:
            pass

    def list_objects(self) -> set[str]:
        try:
            return {n for n in os.listdir(self.objects_dir)
                    if not n.endswith(".tmp")}
        except FileNotFoundError:
            return set()
