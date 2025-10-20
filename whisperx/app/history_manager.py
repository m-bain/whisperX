"""
Database manager for transcription history.
Handles SQLite3 operations for storing and retrieving transcription records.
"""
import sqlite3
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class TranscriptionRecord:
    """Data class representing a single transcription history record."""

    # Required fields
    timestamp: str  # ISO 8601 format
    filepath: str
    file_size: int  # bytes
    transcription_time: float  # seconds
    model_name: str
    status: str  # 'success' or 'error'

    # Optional fields
    id: Optional[int] = None
    duration: Optional[float] = None  # audio duration in seconds
    language: Optional[str] = None
    alignment_enabled: bool = False
    diarization_enabled: bool = False
    result_raw: Optional[str] = None
    result_timestamped: Optional[str] = None
    result_speakers: Optional[str] = None
    result_full: Optional[str] = None
    error_message: Optional[str] = None

    def get_filename(self) -> str:
        """Extract filename from filepath."""
        return os.path.basename(self.filepath)

    def get_file_size_mb(self) -> float:
        """Get file size in MB."""
        return self.file_size / (1024 * 1024)

    def get_formatted_timestamp(self) -> str:
        """Get human-readable timestamp."""
        try:
            dt = datetime.fromisoformat(self.timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            return self.timestamp


class HistoryManager:
    """Manages transcription history database operations."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize history manager.

        Args:
            db_path: Path to SQLite database file. If None, uses default location.
        """
        if db_path is None:
            # Use same directory as app config
            config_dir = Path.home() / '.whisperx_app'
            config_dir.mkdir(exist_ok=True)
            db_path = str(config_dir / 'transcription_history.db')

        self.db_path = db_path
        self.connection = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Establish database connection."""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise

    def _create_tables(self):
        """Create database tables if they don't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS transcription_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            filepath TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            duration REAL,
            transcription_time REAL NOT NULL,
            model_name TEXT NOT NULL,
            language TEXT,
            alignment_enabled INTEGER NOT NULL DEFAULT 0,
            diarization_enabled INTEGER NOT NULL DEFAULT 0,
            result_raw TEXT,
            result_timestamped TEXT,
            result_speakers TEXT,
            result_full TEXT,
            error_message TEXT,
            status TEXT NOT NULL
        );
        """

        create_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_timestamp 
        ON transcription_history(timestamp DESC);
        """

        try:
            cursor = self.connection.cursor()
            cursor.execute(create_table_sql)
            cursor.execute(create_index_sql)
            self.connection.commit()
        except Exception as e:
            print(f"Error creating tables: {e}")
            raise

    def add_transcription(self, record: TranscriptionRecord) -> int:
        """
        Add a new transcription record to the database.

        Args:
            record: TranscriptionRecord to add

        Returns:
            int: ID of the inserted record
        """
        insert_sql = """
        INSERT INTO transcription_history (
            timestamp, filepath, file_size, duration, transcription_time,
            model_name, language, alignment_enabled, diarization_enabled,
            result_raw, result_timestamped, result_speakers, result_full,
            error_message, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        try:
            cursor = self.connection.cursor()
            cursor.execute(insert_sql, (
                record.timestamp,
                record.filepath,
                record.file_size,
                record.duration,
                record.transcription_time,
                record.model_name,
                record.language,
                1 if record.alignment_enabled else 0,
                1 if record.diarization_enabled else 0,
                record.result_raw,
                record.result_timestamped,
                record.result_speakers,
                record.result_full,
                record.error_message,
                record.status
            ))
            self.connection.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"Error adding transcription record: {e}")
            self.connection.rollback()
            raise

    def get_all_transcriptions(self, limit: int = 100) -> List[TranscriptionRecord]:
        """
        Get all transcription records, sorted by timestamp (newest first).

        Args:
            limit: Maximum number of records to return

        Returns:
            List of TranscriptionRecord objects
        """
        select_sql = """
        SELECT * FROM transcription_history
        ORDER BY timestamp DESC
        LIMIT ?
        """

        try:
            cursor = self.connection.cursor()
            cursor.execute(select_sql, (limit,))
            rows = cursor.fetchall()

            records = []
            for row in rows:
                record = TranscriptionRecord(
                    id=row['id'],
                    timestamp=row['timestamp'],
                    filepath=row['filepath'],
                    file_size=row['file_size'],
                    duration=row['duration'],
                    transcription_time=row['transcription_time'],
                    model_name=row['model_name'],
                    language=row['language'],
                    alignment_enabled=bool(row['alignment_enabled']),
                    diarization_enabled=bool(row['diarization_enabled']),
                    result_raw=row['result_raw'],
                    result_timestamped=row['result_timestamped'],
                    result_speakers=row['result_speakers'],
                    result_full=row['result_full'],
                    error_message=row['error_message'],
                    status=row['status']
                )
                records.append(record)

            return records
        except Exception as e:
            print(f"Error retrieving transcription records: {e}")
            return []

    def get_transcription_by_id(self, record_id: int) -> Optional[TranscriptionRecord]:
        """
        Get a specific transcription record by ID.

        Args:
            record_id: ID of the record to retrieve

        Returns:
            TranscriptionRecord or None if not found
        """
        select_sql = """
        SELECT * FROM transcription_history
        WHERE id = ?
        """

        try:
            cursor = self.connection.cursor()
            cursor.execute(select_sql, (record_id,))
            row = cursor.fetchone()

            if row:
                return TranscriptionRecord(
                    id=row['id'],
                    timestamp=row['timestamp'],
                    filepath=row['filepath'],
                    file_size=row['file_size'],
                    duration=row['duration'],
                    transcription_time=row['transcription_time'],
                    model_name=row['model_name'],
                    language=row['language'],
                    alignment_enabled=bool(row['alignment_enabled']),
                    diarization_enabled=bool(row['diarization_enabled']),
                    result_raw=row['result_raw'],
                    result_timestamped=row['result_timestamped'],
                    result_speakers=row['result_speakers'],
                    result_full=row['result_full'],
                    error_message=row['error_message'],
                    status=row['status']
                )
            return None
        except Exception as e:
            print(f"Error retrieving transcription record {record_id}: {e}")
            return None

    def delete_transcription(self, record_id: int) -> bool:
        """
        Delete a transcription record.

        Args:
            record_id: ID of the record to delete

        Returns:
            bool: True if successful, False otherwise
        """
        delete_sql = "DELETE FROM transcription_history WHERE id = ?"

        try:
            cursor = self.connection.cursor()
            cursor.execute(delete_sql, (record_id,))
            self.connection.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error deleting transcription record {record_id}: {e}")
            self.connection.rollback()
            return False

    def search_transcriptions(self, query: str, limit: int = 50) -> List[TranscriptionRecord]:
        """
        Search transcriptions by filepath or transcription content.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of matching TranscriptionRecord objects
        """
        search_sql = """
        SELECT * FROM transcription_history
        WHERE filepath LIKE ? 
           OR result_raw LIKE ?
           OR result_full LIKE ?
        ORDER BY timestamp DESC
        LIMIT ?
        """

        search_pattern = f"%{query}%"

        try:
            cursor = self.connection.cursor()
            cursor.execute(search_sql, (search_pattern, search_pattern, search_pattern, limit))
            rows = cursor.fetchall()

            records = []
            for row in rows:
                record = TranscriptionRecord(
                    id=row['id'],
                    timestamp=row['timestamp'],
                    filepath=row['filepath'],
                    file_size=row['file_size'],
                    duration=row['duration'],
                    transcription_time=row['transcription_time'],
                    model_name=row['model_name'],
                    language=row['language'],
                    alignment_enabled=bool(row['alignment_enabled']),
                    diarization_enabled=bool(row['diarization_enabled']),
                    result_raw=row['result_raw'],
                    result_timestamped=row['result_timestamped'],
                    result_speakers=row['result_speakers'],
                    result_full=row['result_full'],
                    error_message=row['error_message'],
                    status=row['status']
                )
                records.append(record)

            return records
        except Exception as e:
            print(f"Error searching transcription records: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about transcription history.

        Returns:
            Dict with statistics (total count, total time, etc.)
        """
        stats_sql = """
        SELECT 
            COUNT(*) as total_count,
            SUM(transcription_time) as total_time,
            AVG(transcription_time) as avg_time,
            SUM(file_size) as total_file_size,
            COUNT(CASE WHEN status = 'success' THEN 1 END) as success_count,
            COUNT(CASE WHEN status = 'error' THEN 1 END) as error_count
        FROM transcription_history
        """

        try:
            cursor = self.connection.cursor()
            cursor.execute(stats_sql)
            row = cursor.fetchone()

            return {
                'total_count': row['total_count'] or 0,
                'total_time': row['total_time'] or 0.0,
                'avg_time': row['avg_time'] or 0.0,
                'total_file_size': row['total_file_size'] or 0,
                'success_count': row['success_count'] or 0,
                'error_count': row['error_count'] or 0
            }
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None