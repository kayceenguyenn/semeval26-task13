#!/usr/bin/env python3
"""
Environment verification script for SemEval 2026 Task 13
Checks that all dependencies are correctly installed with correct versions
"""

import sys
import importlib.metadata

# Required packages with exact versions from requirements.txt
REQUIRED_PACKAGES = {
    'pandas': '2.1.4',
    'numpy': '1.26.2',
    'scikit-learn': '1.3.2',
    'typer': '0.9.0',
    'rich': '13.7.0',
    'loguru': '0.7.2',
    'pyarrow': '14.0.1',
    'jupyter': '1.0.0',
    'matplotlib': '3.8.2',
    'seaborn': '0.13.0',
    'pytest': '7.4.3',
}

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (3.9+ required)")
        return False

def check_package(package_name, required_version):
    """Check if package is installed with correct version"""
    try:
        installed_version = importlib.metadata.version(package_name)
        if installed_version == required_version:
            print(f"   ✅ {package_name}=={installed_version}")
            return True
        else:
            print(f"   ⚠️  {package_name}=={installed_version} (expected {required_version})")
            return False
    except importlib.metadata.PackageNotFoundError:
        print(f"   ❌ {package_name} not installed")
        return False

def main():
    print("=" * 60)
    print("🔍 SemEval 2026 Task 13 - Environment Verification")
    print("=" * 60)
    print()
    
    all_good = True
    
    # Check Python version
    if not check_python_version():
        all_good = False
    print()
    
    # Check packages
    print("📦 Checking package versions...")
    for package, version in REQUIRED_PACKAGES.items():
        if not check_package(package, version):
            all_good = False
    print()
    
    # Final verdict
    print("=" * 60)
    if all_good:
        print("✅ Environment is correctly configured!")
        print("   All dependencies match requirements.txt")
        print()
        print("🎯 You're ready to start!")
        print("   Run: python3 src/generate_data.py --task A")
        return 0
    else:
        print("❌ Environment has issues!")
        print()
        print("🔧 To fix:")
        print("   1. Activate virtual environment: source venv/bin/activate")
        print("   2. Reinstall dependencies: pip install -r requirements.txt")
        print("   3. Run this script again: python3 verify_environment.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())
