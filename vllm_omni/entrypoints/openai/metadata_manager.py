"""
Metadata manager for voice samples and cache information.

Provides a unified interface for managing metadata.json with
concurrency safety and data consistency across multiple processes.
"""

import fcntl
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MetadataManager:
    """
    Manages metadata for uploaded speakers and cache information.

    Features:
    1. Single source of truth for metadata
    2. Concurrency safety with threading locks
    3. Atomic read-modify-write operations
    4. Merge updates to preserve fields from different components
    """

    def __init__(self, metadata_file: Path):
        self.metadata_file = metadata_file
        self._lock = threading.Lock()
        self._metadata = self._load_from_disk()

        self.lock_file = metadata_file.with_suffix(".lock")
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_from_disk(self) -> dict[str, Any]:
        if not self.metadata_file.exists():
            return {"uploaded_speakers": {}}

        try:
            with open(self.metadata_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata from {self.metadata_file}: {e}")
            return {"uploaded_speakers": {}}

    def _save_to_disk(self, metadata: dict[str, Any]) -> bool:
        try:
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.metadata_file.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(metadata, f, indent=2)
            tmp.replace(self.metadata_file)
            return True
        except Exception as e:
            logger.error(f"Failed to save metadata to {self.metadata_file}: {e}")
            return False

    def _update_with_file_lock(
        self, update_fn: Callable[[dict[str, Any]], dict[str, Any] | None]
    ) -> dict[str, Any] | None:
        lock_fd = os.open(self.lock_file, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            metadata = self._load_from_disk()
            result = update_fn(metadata)
            if result is None:
                return None

            if not self._save_to_disk(metadata):
                return None

            self._metadata = metadata
            return result
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def get_uploaded_speakers(self) -> dict[str, dict[str, Any]]:
        metadata = self._load_from_disk()
        return metadata.get("uploaded_speakers", {}).copy()

    def get_speaker(self, speaker_key: str) -> dict[str, Any] | None:
        metadata = self._load_from_disk()
        speakers = metadata.get("uploaded_speakers", {})
        return speakers.get(speaker_key, {}).copy() if speaker_key in speakers else None

    def update_speaker(self, speaker_key: str, updates: dict[str, Any]) -> bool:
        with self._lock:

            def _update(metadata: dict[str, Any]):
                speakers = metadata.setdefault("uploaded_speakers", {})
                entry = speakers.get(speaker_key, {})
                entry.update(updates)
                speakers[speaker_key] = entry
                return True

            return self._update_with_file_lock(_update) is not None

    def create_speaker(self, speaker_key: str, speaker_data: dict[str, Any]) -> bool:
        with self._lock:

            def _create(metadata: dict[str, Any]):
                speakers = metadata.setdefault("uploaded_speakers", {})
                if speaker_key in speakers:
                    logger.warning(f"Speaker {speaker_key} already exists")
                    return None
                speakers[speaker_key] = speaker_data
                return True

            return self._update_with_file_lock(_create) is not None

    def update_cache_info(self, speaker_key: str, cache_file_path: Path, status: str = "ready") -> bool:
        updates = {
            "cache_status": status,
            "cache_file": str(cache_file_path),
            "cache_generated_at": time.time(),
        }
        return self.update_speaker(speaker_key, updates)

    def delete_speaker(self, speaker_key: str) -> dict[str, Any] | None:
        with self._lock:

            def _delete(metadata: dict[str, Any]):
                speakers = metadata.get("uploaded_speakers", {})
                if speaker_key not in speakers:
                    logger.warning(f"Speaker {speaker_key} not found in metadata")
                    return None

                speaker_info = speakers.pop(speaker_key)

                deleted_files = self._cleanup_speaker_files(speaker_info)
                if deleted_files:
                    logger.info(f"Deleted {len(deleted_files)} files for speaker {speaker_key}: {deleted_files}")

                return speaker_info

            return self._update_with_file_lock(_delete)

    def _cleanup_speaker_files(self, speaker_info: dict[str, Any]) -> list[str]:
        deleted_files = []

        def safe_delete(file_path_str: str, description: str) -> bool:
            if not file_path_str:
                return False

            try:
                file_path = Path(file_path_str)

                if not file_path.exists():
                    logger.debug(f"{description} not found: {file_path}")
                    return False

                file_path.unlink()
                logger.info(f"Deleted {description}: {file_path}")
                deleted_files.append(str(file_path))
                return True

            except Exception as e:
                logger.error(f"Failed to delete {description} {file_path_str}: {e}")
                return False

        audio_file = speaker_info.get("file_path")
        if audio_file:
            safe_delete(audio_file, "audio file")

        cache_file = speaker_info.get("cache_file")
        if cache_file:
            safe_delete(cache_file, "cache file")

        return deleted_files

    def reload_from_disk(self) -> bool:
        with self._lock:
            try:
                self._metadata = self._load_from_disk()
                return True
            except Exception as e:
                logger.error(f"Failed to reload metadata from disk: {e}")
                return False
