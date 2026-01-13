"""
Diagnostic script for WhisperX Launcher
Run this to check if the fixes are applied and Python path is correct
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_executable():
    """Check if the Python executable can be found."""
    print("\n" + "=" * 60)
    print("PYTHON EXECUTABLE CHECK")
    print("=" * 60)

    # Check if running frozen (bundled)
    is_frozen = getattr(sys, 'frozen', False)
    print(f"Running as bundled executable: {is_frozen}")
    print(f"Current Python: {sys.executable}")
    print(f"Platform: {sys.platform}")

    # Try to find bundled Python
    if is_frozen:
        base_path = Path(sys.executable).parent
        print(f"\nBase path (from sys.executable): {base_path}")

        possible_paths = [
            base_path / 'python' / 'python.exe',
            base_path / 'python' / 'bin' / 'python3',
            base_path / 'SmartVoice' / 'python' / 'python.exe',
            Path(sys._MEIPASS) / 'python' / 'python.exe' if hasattr(sys, '_MEIPASS') else None,
        ]

        print("\nChecking possible Python locations:")
        for path in possible_paths:
            if path:
                exists = path.exists()
                print(f"  {'✓' if exists else '✗'} {path}")
                if exists:
                    print(f"    → This is a valid Python executable!")
    else:
        print("\nRunning from source, will use system Python")
        print(f"System Python exists: {Path(sys.executable).exists()}")


def check_dependency_manager_code():
    """Check if the fix has been applied to dependency_manager.py"""
    print("\n" + "=" * 60)
    print("DEPENDENCY MANAGER CODE CHECK")
    print("=" * 60)

    try:
        # Try to import and check the code
        import whisperx.launcher.core.dependency_manager as dm_module

        # Get the source file path
        source_file = Path(dm_module.__file__)
        print(f"dependency_manager.py location: {source_file}")
        print(f"File exists: {source_file.exists()}")

        if source_file.exists():
            # Read the file and check for the fix
            with open(source_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for the old broken pattern
            has_old_bug = "cmd_parts[0] = str(self.python_exe)" in content and \
                          "cmd_parts[1] = '-m'" in content

            # Check for the new fixed pattern
            has_fix = "packages_and_args = cmd_parts[2:]" in content and \
                      "'-m'," in content and \
                      "'pip'," in content and \
                      "'install'" in content

            print("\nCode Analysis:")
            print(f"  Contains old bug pattern: {'YES ❌' if has_old_bug else 'NO ✓'}")
            print(f"  Contains fix pattern: {'YES ✓' if has_fix else 'NO ❌'}")

            if has_old_bug:
                print("\n  ⚠️  WARNING: The old buggy code is still present!")
                print("  ⚠️  The fix has NOT been applied correctly!")
            elif has_fix:
                print("\n  ✓ The fix appears to be applied correctly!")
            else:
                print("\n  ⚠️  Code doesn't match either pattern - manual review needed")

            # Check the _install_pytorch method specifically
            if '_install_pytorch' in content:
                # Find the method
                start = content.find('def _install_pytorch(')
                if start > 0:
                    # Get next 2000 characters to show the method
                    method_snippet = content[start:start + 2000]
                    print("\n_install_pytorch method snippet:")
                    print("-" * 60)
                    # Show first 20 lines
                    lines = method_snippet.split('\n')[:25]
                    for i, line in enumerate(lines, 1):
                        print(f"{i:3d}: {line}")

    except ImportError as e:
        print(f"❌ Could not import dependency_manager: {e}")
    except Exception as e:
        print(f"❌ Error analyzing code: {e}")


def test_subprocess_command():
    """Test if we can construct and validate the command."""
    print("\n" + "=" * 60)
    print("COMMAND CONSTRUCTION TEST")
    print("=" * 60)

    try:
        from whisperx.launcher.core import DependencyManager, InstallType
        from whisperx.launcher.core.hardware_detection import HardwareDetector

        # Get install directory
        if getattr(sys, 'frozen', False):
            install_dir = Path(sys.executable).parent
        else:
            install_dir = Path(__file__).parent

        # Create dependency manager
        dm = DependencyManager(install_dir)

        print(f"\nDependencyManager initialized:")
        print(f"  Install directory: {dm.install_dir}")
        print(f"  Python executable: {dm.python_exe}")
        print(f"  Python exe exists: {dm.python_exe.exists()}")

        # Test command construction
        install_cmd = HardwareDetector.get_pytorch_install_command('cpu')
        print(f"\nBase command from HardwareDetector:")
        print(f"  {install_cmd}")

        # Try to construct the command the way the code should
        cmd_parts = install_cmd.split()
        print(f"\nSplit into parts ({len(cmd_parts)} parts):")
        for i, part in enumerate(cmd_parts):
            print(f"  [{i}] {part}")

        # Show what the CORRECT construction should be
        if len(cmd_parts) >= 3 and cmd_parts[0] == 'pip' and cmd_parts[1] == 'install':
            packages_and_args = cmd_parts[2:]
            correct_cmd = [
                              str(dm.python_exe),
                              '-m',
                              'pip',
                              'install'
                          ] + packages_and_args

            print(f"\nCORRECT command construction:")
            print(f"  Full command list:")
            for i, part in enumerate(correct_cmd):
                print(f"    [{i}] {part}")
            print(f"\n  As string: {' '.join(correct_cmd)}")

    except Exception as e:
        print(f"❌ Error testing command construction: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all diagnostics."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "WhisperX Launcher Diagnostics" + " " * 14 + "║")
    print("╚" + "=" * 58 + "╝")

    check_python_executable()
    check_dependency_manager_code()
    test_subprocess_command()

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. If the fix is not applied, replace dependency_manager.py")
    print("2. If Python executable doesn't exist, check your installation")
    print("3. Review the command construction to ensure it's correct")
    print("\nPress Enter to exit...")
    input()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Diagnostic failed with error: {e}")
        import traceback

        traceback.print_exc()
        print("\nPress Enter to exit...")
        input()