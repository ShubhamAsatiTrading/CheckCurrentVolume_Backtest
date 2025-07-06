# run.py - Smart Startup Script for Trading System
# Handles port conflicts, file validation, and automatic setup

import subprocess
import sys
import os
import socket
import time
import platform

def find_free_port(start_port=8501, max_attempts=10):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None

def kill_process_on_port(port):
    """Kill process running on specified port"""
    try:
        if platform.system() == "Windows":
            # Windows command
            result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            for line in lines:
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True)
                        print(f"[SUCCESS] Killed process {pid} on port {port}")
                        return True
        else:
            # Linux/Mac command
            result = subprocess.run(['lsof', '-ti', f':{port}'], capture_output=True, text=True)
            if result.stdout.strip():
                subprocess.run(['kill', '-9'] + result.stdout.strip().split(), capture_output=True)
                print(f"[SUCCESS] Killed process on port {port}")
                return True
    except Exception as e:
        print(f"[WARNING] Could not kill process on port {port}: {e}")
        pass
    return False

def check_python_packages():
    """Check if required Python packages are installed"""
    # Map pip package names to import names
    package_mapping = {
        'streamlit': 'streamlit',
        'pandas': 'pandas', 
        'python-dotenv': 'dotenv',
        'kiteconnect': 'kiteconnect',
        'requests': 'requests'
    }
    
    missing_packages = []
    
    for pip_name, import_name in package_mapping.items():
        try:
            module = __import__(import_name)
            if import_name == 'kiteconnect':
                print(f"[SUCCESS] {pip_name}: v{module.__version__}")
            else:
                print(f"[SUCCESS] {pip_name}: installed")
        except ImportError:
            missing_packages.append(pip_name)
            print(f"[ERROR] {pip_name}: missing")
    
    if missing_packages:
        print(f"\n[PACKAGE] Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        print(f"\n[TIP] Or install all at once:")
        print(f"pip install streamlit pandas python-dotenv kiteconnect requests")
        return False
    
    print("[SUCCESS] All required packages are installed")
    return True

def check_requirements():
    """Check if all required files exist and create defaults if needed"""
    print("[SEARCH] Checking system requirements...")
    
    # Check Python packages first
    if not check_python_packages():
        return False
    
    required_files = [
        'main.py',
        'code_1.py'
    ]
    
    # Check for essential files
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"[ERROR] Missing essential files: {missing}")
        print("[INFO] Please ensure you have saved the main.py and code_1.py files")
        return False
    
    # Create missing config files with defaults
    files_created = 0
    
    if not os.path.exists('.env'):
        print("[WARNING] Creating default .env file...")
        with open('.env', 'w') as f:
            f.write("# Kite Connect API Credentials\n")
            f.write("KITE_API_KEY=your_kite_api_key_here\n")
            f.write("KITE_API_SECRET=your_kite_api_secret_here\n")
            f.write("\n# Telegram Notifications\n")
            f.write("TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here\n")
            f.write("TELEGRAM_CHAT_ID=your_telegram_chat_id_here\n")
            f.write("\n# Trading System Settings\n")
            f.write("TRADING_MODE=development\n")
            f.write("LOG_LEVEL=INFO\n")
        files_created += 1
        print("[LOG] Please edit .env file with your actual credentials")
    
    if not os.path.exists('common.txt'):
        print("[WARNING] Creating default common.txt...")
        with open('common.txt', 'w') as f:
            f.write("stop_loss=4.0\n")
            f.write("target=10.0\n")
            f.write("ohlc_value=open\n")
            f.write("trade_today_flag=no\n")
        files_created += 1
    
    if not os.path.exists('symbols.txt'):
        print("[WARNING] Creating default symbols.txt...")
        with open('symbols.txt', 'w') as f:
            f.write("RELIANCE\n")
            f.write("CUMMINSIND\n")
        files_created += 1
    
    if files_created > 0:
        print(f"[SUCCESS] Created {files_created} default configuration files")
    
    return True

def create_directories():
    """Create required directories"""
    dirs = [
        'stocks_historical_data',
        'aggregated_data', 
        'Volume_boost_consolidated',
        'backtest_results',
        'input_stocks'
    ]
    
    created = 0
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            created += 1
    
    if created > 0:
        print(f"[SUCCESS] Created {created} directories")
    else:
        print("[SUCCESS] All directories already exist")

def validate_environment():
    """Validate environment variables"""
    print("[CONFIG] Validating environment...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        kite_api_key = os.getenv('KITE_API_KEY', '')
        kite_api_secret = os.getenv('KITE_API_SECRET', '')
        
        if not kite_api_key or kite_api_key == 'your_kite_api_key_here':
            print("[WARNING] KITE_API_KEY not configured in .env file")
            
        if not kite_api_secret or kite_api_secret == 'your_kite_api_secret_here':
            print("[WARNING] KITE_API_SECRET not configured in .env file")
            
        if not kite_api_key or not kite_api_secret or 'your_' in kite_api_key or 'your_' in kite_api_secret:
            print("🔑 Please update your .env file with actual Kite Connect credentials")
            return False
            
        print("[SUCCESS] Environment validation passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Environment validation failed: {e}")
        return False

def main():
    print("[START] TRADING SYSTEM STARTUP")
    print("=" * 50)
    
    # Show Python environment info
    print(f"[PYTHON] Python: {sys.executable}")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("[SUCCESS] Running in virtual environment")
    else:
        print("[WARNING] Not running in virtual environment")
    
    # Check requirements
    if not check_requirements():
        print("\n[ERROR] Setup incomplete. Please fix the issues above and try again.")
        print("\n[CONFIG] If kiteconnect is installed but still showing errors:")
        print("   Run: python debug_kiteconnect.py")
        input("Press Enter to exit...")
        return
    
    # Create directories
    create_directories()
    
    # Validate environment
    env_valid = validate_environment()
    if not env_valid:
        print("[WARNING] Environment issues detected. You can still run the system,")
        print("   but you'll need to configure credentials in the web interface.")
        print()
    
    # Handle port conflicts
    preferred_port = 8501
    print(f"[SEARCH] Checking port availability...")
    
    free_port = find_free_port(preferred_port)
    
    if free_port == preferred_port:
        print(f"[SUCCESS] Port {preferred_port} is available")
    else:
        print(f"[WARNING] Port {preferred_port} is busy!")
        
        # Ask user what to do
        print("\nOptions:")
        print("1. Kill process on port 8501 and use it")
        print("2. Use alternative port")
        print("3. Exit")
        
        while True:
            choice = input("\nChoose option (1/2/3): ").strip()
            
            if choice == "1":
                print(f"[LOADING] Attempting to free port {preferred_port}...")
                if kill_process_on_port(preferred_port):
                    time.sleep(2)  # Wait for port to be freed
                    free_port = find_free_port(preferred_port)
                    if free_port == preferred_port:
                        print(f"[SUCCESS] Port {preferred_port} is now available")
                        break
                    else:
                        print("[ERROR] Could not free port. Using alternative port.")
                        free_port = find_free_port(8502)
                        break
                else:
                    print("[ERROR] Could not free port. Using alternative port.")
                    free_port = find_free_port(8502)
                    break
            
            elif choice == "2":
                free_port = find_free_port(8502)
                break
            
            elif choice == "3":
                print("[GOODBYE] Exiting...")
                return
            
            else:
                print("[ERROR] Invalid choice. Please enter 1, 2, or 3.")
    
    if not free_port:
        print("[ERROR] No free ports available in range 8501-8510")
        input("Press Enter to exit...")
        return
    
    # Start Streamlit
    print(f"\n Starting Streamlit Trading System...")
    print(f"[DATA] Access URL: http://localhost:{free_port}")
    print(f"[WEB] Network URL: http://{get_local_ip()}:{free_port}")
    print("\n[TIP] Tips:")
    print("   • Use the web interface to authenticate with Kite")
    print("   • Configure your trading parameters in the sidebar")
    print("   • Monitor live signals and system status")
    print(f"\n[FAST] Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main.py", 
            "--server.port", str(free_port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light"
        ], check=True)
    except KeyboardInterrupt:
        print("\n\n[GOODBYE] Trading system stopped. Goodbye!")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Error starting Streamlit: {e}")
        print("[TIP] Troubleshooting:")
        print("   • Try: pip install streamlit")
        print("   • Check if Python is in PATH")
        print("   • Restart terminal/command prompt")
        input("Press Enter to exit...")
    except Exception as e:
        print(f"\n[CRITICAL] Unexpected error: {e}")
        input("Press Enter to exit...")

def get_local_ip():
    """Get local IP address for network access"""
    try:
        # Connect to a dummy address to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

if __name__ == "__main__":
    main()

