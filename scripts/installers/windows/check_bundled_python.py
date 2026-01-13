"""
Diagnostic script to check Python and pip installation in bundled app
Run this to see what's wrong with pip
"""

import subprocess
import sys
from pathlib import Path

def check_bundled_python():
    """Check if bundled Python works and if pip is installed."""
    
    print("="*70)
    print("BUNDLED PYTHON DIAGNOSTIC")
    print("="*70)
    
    # Path to bundled Python
    python_path = Path(r"C:\Program Files\WhisperX\SmartVoice\python\python.exe")
    
    print(f"\n1. Python Executable Check")
    print(f"   Path: {python_path}")
    print(f"   Exists: {python_path.exists()}")
    
    if not python_path.exists():
        print("   ❌ ERROR: Python executable not found!")
        return
    
    # Test Python
    print(f"\n2. Testing Python")
    try:
        result = subprocess.run(
            [str(python_path), '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        print(f"   Return code: {result.returncode}")
        print(f"   Output: {result.stdout.strip()}")
        if result.stderr:
            print(f"   Errors: {result.stderr.strip()}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return
    
    # Test pip
    print(f"\n3. Testing pip")
    try:
        result = subprocess.run(
            [str(python_path), '-m', 'pip', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        print(f"   Return code: {result.returncode}")
        
        if result.returncode == 0:
            print(f"   ✅ pip is installed!")
            print(f"   Version: {result.stdout.strip()}")
        else:
            print(f"   ❌ pip is NOT installed or not working")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
    except Exception as e:
        print(f"   ❌ ERROR testing pip: {e}")
    
    # Check for get-pip.py
    print(f"\n4. Checking for get-pip.py")
    get_pip = python_path.parent / 'get-pip.py'
    print(f"   Path: {get_pip}")
    print(f"   Exists: {get_pip.exists()}")
    
    if not get_pip.exists():
        print(f"   ⚠️  WARNING: get-pip.py not found!")
        print(f"   Download from: https://bootstrap.pypa.io/get-pip.py")
        print(f"   Save to: {get_pip}")
    
    # Check python._pth configuration
    print(f"\n5. Checking python._pth configuration")
    pth_files = list(python_path.parent.glob("python*._pth"))
    if pth_files:
        pth_file = pth_files[0]
        print(f"   Found: {pth_file.name}")
        try:
            with open(pth_file, 'r') as f:
                content = f.read()
            print(f"   Content:")
            for line in content.split('\n'):
                print(f"      {line}")
            
            # Check for required lines
            if 'import site' in content and not content.strip().startswith('#import site'):
                print(f"   ✅ 'import site' is enabled")
            else:
                print(f"   ❌ 'import site' is NOT enabled or commented out")
            
            if 'Lib\\site-packages' in content or 'Lib/site-packages' in content:
                print(f"   ✅ 'Lib\\site-packages' is present")
            else:
                print(f"   ❌ 'Lib\\site-packages' is NOT present")
        except Exception as e:
            print(f"   ❌ ERROR reading file: {e}")
    else:
        print(f"   ❌ No python._pth file found!")
    
    # Check Lib and Scripts directories
    print(f"\n6. Checking directory structure")
    lib_dir = python_path.parent / 'Lib' / 'site-packages'
    scripts_dir = python_path.parent / 'Scripts'
    
    print(f"   Lib/site-packages: {lib_dir.exists()}")
    print(f"   Scripts: {scripts_dir.exists()}")
    
    # Try to install pip manually if needed
    print(f"\n7. Attempt to install pip")
    if get_pip.exists():
        print(f"   Running: {python_path} {get_pip}")
        try:
            result = subprocess.run(
                [str(python_path), str(get_pip)],
                capture_output=True,
                text=True,
                timeout=120
            )
            print(f"   Return code: {result.returncode}")
            
            if result.returncode == 0:
                print(f"   ✅ pip installation successful!")
                print(f"   Output (last 500 chars):")
                print(f"   {result.stdout[-500:]}")
            else:
                print(f"   ❌ pip installation failed!")
                print(f"   stdout: {result.stdout}")
                print(f"   stderr: {result.stderr}")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    else:
        print(f"   ⚠️  Skipped (get-pip.py not found)")
    
    # Final pip check
    print(f"\n8. Final pip verification")
    try:
        result = subprocess.run(
            [str(python_path), '-m', 'pip', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"   ✅ pip is now working!")
            print(f"   {result.stdout.strip()}")
        else:
            print(f"   ❌ pip still not working")
            print(f"   stderr: {result.stderr}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)

if __name__ == '__main__':
    try:
        check_bundled_python()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nPress Enter to exit...")
    input()
