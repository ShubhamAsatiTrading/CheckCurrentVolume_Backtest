# debug_kiteconnect.py - Diagnostic Script to Check kiteconnect Installation
# Run this to diagnose kiteconnect installation issues

import sys
import os

print("[SEARCH] KITECONNECT DIAGNOSTIC SCRIPT")
print("=" * 50)

# Check Python environment
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

print("\n[PACKAGE] PACKAGE IMPORT TESTS:")

# Test imports
packages_to_test = [
    'pandas',
    'streamlit', 
    'dotenv',
    'requests',
    'kiteconnect'
]

for package in packages_to_test:
    try:
        if package == 'dotenv':
            import dotenv
            print(f"[SUCCESS] {package}: {dotenv.__version__}")
        elif package == 'kiteconnect':
            import kiteconnect
            print(f"[SUCCESS] {package}: {kiteconnect.__version__}")
        else:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"[SUCCESS] {package}: {version}")
    except ImportError as e:
        print(f"[ERROR] {package}: FAILED - {e}")

print("\n[SEARCH] KITECONNECT SPECIFIC TESTS:")

try:
    from kiteconnect import KiteConnect
    print("[SUCCESS] KiteConnect class import successful")
    
    # Test basic KiteConnect instantiation
    try:
        kite = KiteConnect(api_key="test_key")
        print("[SUCCESS] KiteConnect instantiation successful")
    except Exception as e:
        print(f"[WARNING] KiteConnect instantiation issue: {e}")
        
except ImportError as e:
    print(f"[ERROR] KiteConnect import failed: {e}")
    
    # Provide installation suggestions
    print("\n[CONFIG] SUGGESTED FIXES:")
    print("1. Install kiteconnect:")
    print("   pip install kiteconnect")
    print("\n2. If using virtual environment, activate it first:")
    print("   # Windows:")
    print("   trade_venv\\Scripts\\activate")
    print("   # Linux/Mac:")
    print("   source trade_venv/bin/activate")
    print("\n3. Then install:")
    print("   pip install kiteconnect")
    
print("\n🌍 ENVIRONMENT VARIABLES:")
env_vars = ['KITE_API_KEY', 'KITE_API_SECRET', 'TRADING_ACCESS_TOKEN']
for var in env_vars:
    value = os.getenv(var, 'Not Set')
    if value and value != 'Not Set':
        print(f"[SUCCESS] {var}: {'*' * (len(value) - 4) + value[-4:]}")  # Mask most of the value
    else:
        print(f"[ERROR] {var}: {value}")

print("\n FILE STRUCTURE CHECK:")
required_files = ['main.py', 'code_1.py', 'run.py', '.env', 'common.txt']
for file in required_files:
    if os.path.exists(file):
        print(f"[SUCCESS] {file}: Found")
    else:
        print(f"[ERROR] {file}: Missing")

print("\n" + "=" * 50)
print("[TARGET] DIAGNOSTIC COMPLETE")

# Additional suggestion
print("\n[TIP] IF KITECONNECT STILL FAILS:")
print("Run this in your terminal:")
print(f"{sys.executable} -m pip install kiteconnect --upgrade")
print("\nThis ensures installation in the EXACT Python environment being used.")

