# debug_system.py - Volume Analysis System Diagnostic Tool

import os
import sys
from datetime import datetime

def check_authentication():
    """Check Kite authentication setup"""
    print("🔐 AUTHENTICATION CHECK")
    print("-" * 30)
    
    # Check .env file
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        return False
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('KITE_API_KEY', '')
    api_secret = os.getenv('KITE_API_SECRET', '')
    
    if not api_key or api_key == 'your_api_key_here':
        print("❌ KITE_API_KEY not set in .env")
        return False
    
    if not api_secret or api_secret == 'your_api_secret_here':
        print("❌ KITE_API_SECRET not set in .env")
        return False
    
    print(f"✅ API Key: {api_key[:10]}...")
    print(f"✅ API Secret: {api_secret[:10]}...")
    
    # Check token file
    if os.path.exists('kite_token.txt'):
        with open('kite_token.txt', 'r') as f:
            token = f.read().strip()
        if token:
            print(f"✅ Access Token: {token[:15]}...")
        else:
            print("❌ Empty token file")
            return False
    else:
        print("❌ No access token file found")
        return False
    
    # Test connection
    try:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(token)
        profile = kite.profile()
        print(f"✅ Connected as: {profile.get('user_name', 'User')}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def check_files_and_folders():
    """Check required files and folders"""
    print("\n📁 FILES & FOLDERS CHECK")
    print("-" * 30)
    
    required_files = [
        'main.py',
        'historical_data_download.py',
        'symbols.txt',
        'common.txt'
    ]
    
    optional_files = [
        'volume_average.py',
        'live_data_downloader_parallel.py',
        'volume_logger.py'
    ]
    
    required_folders = [
        'stocks_historical_data',
        'Volume_Avg_Data',
        'StockliveData'
    ]
    
    all_good = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - REQUIRED")
            all_good = False
    
    for file in optional_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"⚠️  {file} - Optional but recommended")
    
    for folder in required_folders:
        if os.path.exists(folder):
            file_count = len([f for f in os.listdir(folder) if f.endswith('.csv')])
            print(f"✅ {folder}/ ({file_count} CSV files)")
        else:
            print(f"❌ {folder}/ - Will be created")
            os.makedirs(folder, exist_ok=True)
    
    return all_good

def test_historical_download():
    """Test historical data download"""
    print("\n📈 HISTORICAL DATA TEST")
    print("-" * 30)
    
    try:
        # Check if we can import the module
        import historical_data_download
        print("✅ historical_data_download module imported")
        
        # Check if symbols.txt exists and has content
        if not os.path.exists('symbols.txt'):
            print("❌ symbols.txt not found")
            return False
        
        with open('symbols.txt', 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        
        if not symbols:
            print("❌ symbols.txt is empty")
            return False
        
        print(f"✅ Found {len(symbols)} symbols: {symbols[:5]}...")
        
        # Check authentication
        if not check_authentication():
            print("❌ Authentication failed - cannot test download")
            return False
        
        print("✅ Ready for historical data download")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_volume_analysis():
    """Test volume analysis"""
    print("\n📊 VOLUME ANALYSIS TEST")
    print("-" * 30)
    
    try:
        # Check if volume_average module exists
        if not os.path.exists('volume_average.py'):
            print("❌ volume_average.py not found")
            return False
        
        import volume_average
        print("✅ volume_average module imported")
        
        # Check if VolumeAverage class exists
        if hasattr(volume_average, 'VolumeAverage'):
            print("✅ VolumeAverage class found")
            
            # Check if correct method exists
            if hasattr(volume_average.VolumeAverage, 'calculate_average_volume_data'):
                print("✅ calculate_average_volume_data method found")
            else:
                print("❌ calculate_average_volume_data method not found")
                return False
        else:
            print("❌ VolumeAverage class not found")
            return False
        
        # Check for historical data
        hist_folder = 'stocks_historical_data'
        if os.path.exists(hist_folder):
            hist_files = [f for f in os.listdir(hist_folder) if f.endswith('_historical.csv')]
            if hist_files:
                print(f"✅ Found {len(hist_files)} historical data files")
            else:
                print("⚠️  No historical data files found - run historical download first")
        else:
            print("❌ stocks_historical_data folder not found")
            return False
        
        print("✅ Ready for volume analysis")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_live_data():
    """Test live data downloader"""
    print("\n📡 LIVE DATA TEST")
    print("-" * 30)
    
    try:
        if not os.path.exists('live_data_downloader_parallel.py'):
            print("❌ live_data_downloader_parallel.py not found")
            return False
        
        print("✅ live_data_downloader_parallel.py found")
        
        # Check common.txt configuration
        config = {}
        if os.path.exists('common.txt'):
            with open('common.txt', 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        config[key] = value
        
        print(f"✅ Rerun interval: {config.get('rerun_minute', '1')} minute(s)")
        print(f"✅ Live download: {config.get('live_data_download', 'no')}")
        
        if not check_authentication():
            print("❌ Authentication failed - cannot run live data")
            return False
        
        print("✅ Ready for live data download")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_full_diagnostic():
    """Run complete system diagnostic"""
    print("🔧 VOLUME ANALYSIS SYSTEM DIAGNOSTIC")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print("=" * 50)
    
    tests = [
        check_files_and_folders,
        check_authentication,
        test_historical_download,
        test_volume_analysis,
        test_live_data
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    if all(results):
        print("🎉 ALL TESTS PASSED! System is ready to use.")
        print("\n📋 Next steps:")
        print("1. Run: python run.py")
        print("2. Open the dashboard in your browser") 
        print("3. Follow the 4-step process in the dashboard")
    else:
        print("⚠️  SOME TESTS FAILED. Please fix the issues above.")
        print("\n🔧 Common fixes:")
        print("- Update .env with your Kite API credentials")
        print("- Get access token from Kite and save it")
        print("- Ensure all required files are present")
    
    print("=" * 50)

if __name__ == "__main__":
    run_full_diagnostic()