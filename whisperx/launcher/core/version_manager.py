"""
Version management and update checking for WhisperX Launcher.
Manages application versions, checks for updates, and handles downloads.
"""

import json
import logging
import platform
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from packaging import version

# Import version from single source of truth
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from whisperx.__version__ import __version__ as CURRENT_VERSION

logger = logging.getLogger(__name__)


class VersionManager:
    """Manages application versions and update checking."""

    # GitHub API configuration
    REPO_OWNER = "xlazarik"
    REPO_NAME = "whisperX"
    RELEASES_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases"

    # Cache configuration
    CACHE_DURATION_HOURS = 6  # Check for updates max once per 6 hours

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize VersionManager.

        Args:
            config_dir: Directory for storing version cache and config
        """
        self.config_dir = config_dir or (Path.home() / '.whisperx_app')
        self.config_dir.mkdir(exist_ok=True)

        self.cache_file = self.config_dir / 'version_cache.json'
        self.current_version = CURRENT_VERSION

    def get_current_version(self) -> str:
        """Get the current installed version."""
        return self.current_version

    def check_for_updates(self, force: bool = False) -> Dict[str, Any]:
        """
        Check for available updates from GitHub releases.

        Args:
            force: Force check even if cache is fresh

        Returns:
            dict: {
                'update_available': bool,
                'latest_version': str,
                'current_version': str,
                'download_url': str,
                'release_notes': str,
                'release_date': str,
                'asset_size': int,
                'requires_reinstall': bool,
                'checked_at': str
            }
        """
        # Check cache first
        if not force:
            cached = self._get_cached_update_info()
            if cached:
                logger.info("Using cached update information")
                return cached

        logger.info(f"Checking for updates (current version: {self.current_version})")

        try:
            # Fetch latest release from GitHub
            response = requests.get(
                f"{self.RELEASES_API_URL}/latest",
                timeout=10,
                headers={'Accept': 'application/vnd.github.v3+json'}
            )
            response.raise_for_status()

            release_data = response.json()
            latest_version = release_data['tag_name'].lstrip('v')

            # Check if update available
            update_available = version.parse(latest_version) > version.parse(self.current_version)

            # Get platform-specific download URL
            download_info = self._get_platform_download_info(release_data)

            # Determine if reinstall required (major/minor version change)
            requires_reinstall = self._requires_reinstall(self.current_version, latest_version)

            update_info = {
                'update_available': update_available,
                'latest_version': latest_version,
                'current_version': self.current_version,
                'download_url': download_info['url'],
                'asset_name': download_info['name'],
                'asset_size': download_info['size'],
                'release_notes': release_data.get('body', 'No release notes available'),
                'release_date': release_data.get('published_at', ''),
                'requires_reinstall': requires_reinstall,
                'checked_at': datetime.now().isoformat()
            }

            # Cache the result
            self._cache_update_info(update_info)

            logger.info(f"Update check complete: {'Update available' if update_available else 'Up to date'}")
            return update_info

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to check for updates: {e}")
            return {
                'update_available': False,
                'error': str(e),
                'current_version': self.current_version,
                'checked_at': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Unexpected error checking for updates: {e}")
            return {
                'update_available': False,
                'error': f"Unexpected error: {str(e)}",
                'current_version': self.current_version,
                'checked_at': datetime.now().isoformat()
            }

    def get_all_releases(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get list of all available releases.

        Args:
            limit: Maximum number of releases to return

        Returns:
            List of release info dictionaries
        """
        try:
            response = requests.get(
                f"{self.RELEASES_API_URL}?per_page={limit}",
                timeout=10,
                headers={'Accept': 'application/vnd.github.v3+json'}
            )
            response.raise_for_status()

            releases = []
            for release_data in response.json():
                download_info = self._get_platform_download_info(release_data)

                releases.append({
                    'version': release_data['tag_name'].lstrip('v'),
                    'name': release_data['name'],
                    'release_notes': release_data.get('body', ''),
                    'release_date': release_data.get('published_at', ''),
                    'download_url': download_info['url'],
                    'asset_name': download_info['name'],
                    'asset_size': download_info['size'],
                    'prerelease': release_data.get('prerelease', False)
                })

            return releases

        except Exception as e:
            logger.error(f"Failed to fetch releases: {e}")
            return []

    def _get_platform_download_info(self, release_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract the correct installer download URL for current platform.

        Args:
            release_data: GitHub release API response

        Returns:
            dict: {'url': str, 'name': str, 'size': int}
        """
        system = platform.system()
        assets = release_data.get('assets', [])

        # Platform-specific file patterns
        patterns = {
            'Windows': ['.exe', '-win-', '-windows-'],
            'Linux': ['.AppImage', '.deb', '-linux-'],
            'Darwin': ['.dmg', '.pkg', '-macos-', '-darwin-']
        }

        # Find matching asset
        for asset in assets:
            name = asset['name'].lower()
            for pattern in patterns.get(system, []):
                if pattern.lower() in name:
                    return {
                        'url': asset['browser_download_url'],
                        'name': asset['name'],
                        'size': asset['size']
                    }

        # No specific installer found, return first asset or empty
        if assets:
            logger.warning(f"No platform-specific installer found, using first asset")
            return {
                'url': assets[0]['browser_download_url'],
                'name': assets[0]['name'],
                'size': assets[0]['size']
            }

        return {
            'url': None,
            'name': None,
            'size': 0
        }

    def _requires_reinstall(self, current_ver: str, new_ver: str) -> bool:
        """
        Determine if update requires full reinstall.

        Args:
            current_ver: Current version string
            new_ver: New version string

        Returns:
            bool: True if reinstall required
        """
        try:
            current = version.parse(current_ver)
            new = version.parse(new_ver)

            # Major version change always requires reinstall
            if new.major > current.major:
                return True

            # Minor version change may require reinstall (dependencies might change)
            if new.minor > current.minor:
                return True

            # Patch version changes are usually safe
            return False

        except Exception:
            # If we can't parse versions, assume reinstall required for safety
            return True

    def _get_cached_update_info(self) -> Optional[Dict[str, Any]]:
        """
        Get cached update information if still fresh.

        Returns:
            Cached update info dict or None if cache is stale/missing
        """
        if not self.cache_file.exists():
            return None

        try:
            with open(self.cache_file, 'r') as f:
                cached = json.load(f)

            # Check if cache is still fresh
            checked_at = datetime.fromisoformat(cached['checked_at'])
            cache_age = datetime.now() - checked_at

            if cache_age < timedelta(hours=self.CACHE_DURATION_HOURS):
                return cached

            logger.debug("Update cache is stale")
            return None

        except Exception as e:
            logger.debug(f"Failed to read update cache: {e}")
            return None

    def _cache_update_info(self, update_info: Dict[str, Any]) -> None:
        """
        Cache update information to disk.

        Args:
            update_info: Update information to cache
        """
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(update_info, f, indent=2)
            logger.debug("Update info cached successfully")
        except Exception as e:
            logger.warning(f"Failed to cache update info: {e}")

    def clear_cache(self) -> None:
        """Clear the update cache."""
        if self.cache_file.exists():
            self.cache_file.unlink()
            logger.info("Update cache cleared")

    def format_release_notes(self, release_notes: str, max_length: int = 500) -> str:
        """
        Format and truncate release notes for display.

        Args:
            release_notes: Raw release notes (markdown)
            max_length: Maximum length of formatted notes

        Returns:
            Formatted release notes string
        """
        if not release_notes:
            return "No release notes available"

        # Truncate if too long
        if len(release_notes) > max_length:
            release_notes = release_notes[:max_length] + "..."

        return release_notes

    def format_file_size(self, size_bytes: int) -> str:
        """
        Format file size in human-readable format.

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted size string (e.g., "125.5 MB")
        """
        if size_bytes == 0:
            return "0 B"

        units = ['B', 'KB', 'MB', 'GB']
        unit_index = 0
        size = float(size_bytes)

        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1

        return f"{size:.1f} {units[unit_index]}"


if __name__ == '__main__':
    # Test version manager
    logging.basicConfig(level=logging.INFO)

    vm = VersionManager()
    print(f"\n=== Version Manager Test ===")
    print(f"Current Version: {vm.get_current_version()}")

    print("\nChecking for updates...")
    update_info = vm.check_for_updates(force=True)

    if 'error' in update_info:
        print(f"Error: {update_info['error']}")
    else:
        print(f"Update Available: {update_info['update_available']}")
        print(f"Latest Version: {update_info.get('latest_version', 'N/A')}")
        if update_info['update_available']:
            print(f"Download URL: {update_info['download_url']}")
            print(f"File Size: {vm.format_file_size(update_info['asset_size'])}")
            print(f"Requires Reinstall: {update_info['requires_reinstall']}")
            print(f"\nRelease Notes:\n{vm.format_release_notes(update_info['release_notes'])}")
