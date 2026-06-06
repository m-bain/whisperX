"""The pluggable storage backend interface.

A backend is a dumb content-addressed object store plus one mutable pointer (the
manifest). All sync intelligence — snapshotting, diffing, GC — lives in
:class:`app.backup.service.BackupService`; a backend only has to move bytes. This
keeps Google Drive swappable for S3 / WebDAV / a local folder later.

Remote layout a backend must present:
    manifest.json        # the merkle root; written LAST as the commit point
    objects/<sha256>     # immutable content-addressed blobs

``write_manifest`` MUST be atomic (write-then-rename) so a reader never sees a
half-written manifest — the manifest swap is what makes a backup "land".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import BinaryIO

from app.backup.manifest import Manifest


class StorageBackend(ABC):
    #: Short identifier for logs / status (e.g. "gdrive", "local").
    name: str = "backend"

    @abstractmethod
    def is_linked(self) -> bool:
        """Whether usable credentials / a reachable target exist right now."""

    @abstractmethod
    def read_manifest(self) -> Manifest | None:
        """The current remote manifest, or None if the target is fresh/empty."""

    def probe(self) -> Manifest | None:
        """Alias for :meth:`read_manifest` used during bootstrap-on-link."""
        return self.read_manifest()

    @abstractmethod
    def write_manifest(self, manifest: Manifest) -> None:
        """Atomically replace the remote manifest (write-then-rename)."""

    @abstractmethod
    def has_object(self, key: str) -> bool:
        """Cheap existence check so an unchanged blob is never re-uploaded."""

    @abstractmethod
    def put_object(self, key: str, data: BinaryIO) -> None:
        """Store a blob under its content hash ``key`` (idempotent)."""

    @abstractmethod
    def get_object(self, key: str) -> bytes:
        """Fetch a blob by content hash. Raises KeyError if absent."""

    @abstractmethod
    def delete_object(self, key: str) -> None:
        """Remove a blob (no-op if absent). Used by GC."""

    @abstractmethod
    def list_objects(self) -> set[str]:
        """Every stored blob key — so GC can find orphans the manifest dropped."""

    def set_folder(self, name: str) -> None:
        """Re-target the backend at a different destination folder/name.

        No-op by default; backends with a user-chosen destination (Drive) override
        this. Lets the UI switch destinations on a live backend without a restart.
        """
        return None
