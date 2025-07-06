# code_1.py - FIXED VERSION with encoding issue resolved
# Replace the logging setup and emoji usage in your code_1.py

import os
import sys
import logging
import json
from datetime import datetime
from dotenv import load_dotenv
from scrapper import scrape_and_save_all
import requests

# Load environment variables
load_dotenv()

# Setup logging with UTF-8 encoding fix
def setup_logging():
    """Setup logging with proper UTF-8 encoding"""
    # Force UTF-8 encoding for stdout/stderr on Windows
    if sys.platform.startswith('win'):
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    
    # Setup logging with UTF-8 encoding
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('trading_system.log', mode='a', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

# Token management
TOKEN_FILE = "kite_token.txt"

def get_credentials():
    """Get API credentials from environment or token file"""
    # Try to get from web interface token (priority)
    access_token = os.getenv('TRADING_ACCESS_TOKEN')
    
    # If not available, try to get from saved token file
    if not access_token:
        access_token = get_saved_token()
    
    api_key = os.getenv('KITE_API_KEY')
    
    if not access_token:
        logger.error("No access token available (neither from environment nor saved file)")
        logger.info("Please use the web interface to authenticate or set TRADING_ACCESS_TOKEN environment variable")
        return None, None
    
    if not api_key:
        logger.error("KITE_API_KEY not found in environment")
        return None, None
    
    return access_token, api_key

def get_saved_token():
    """Get saved token if valid for today"""
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r", encoding='utf-8') as f:
                token_data = json.loads(f.read().strip())
                today = datetime.now().strftime("%Y-%m-%d")
                
                if token_data.get("date") == today and token_data.get("access_token"):
                    logger.info("Using saved token from file")
                    return token_data["access_token"]
                else:
                    logger.info("Saved token expired, removing old token file")
                    os.remove(TOKEN_FILE)  # Delete old token
        except Exception as e:
            logger.warning(f"Error reading token file: {e}")
            if os.path.exists(TOKEN_FILE):
                os.remove(TOKEN_FILE)
    return None

def read_config():
    """Read configuration from common.txt"""
    config = {
        'stop_loss': 4.0,
        'target': 10.0,
        'ohlc_value': 'open',
        'trade_today_flag': 'no'
    }
    try:
        with open('common.txt', 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    if key in ['stop_loss', 'target']:
                        config[key] = float(value)
                    else:
                        config[key] = value
        logger.info(f"Configuration loaded: {config}")
        return config
    except Exception as e:
        logger.error(f"Error reading config: {e}")
        return config

def is_market_hours():
    """Check if within market hours (9:15 AM - 3:30 PM)"""
    now = datetime.now().time()
    market_start = datetime.strptime("09:15", "%H:%M").time()
    market_end = datetime.strptime("15:30", "%H:%M").time()
    return market_start <= now <= market_end

def execute_module(module_name, function_name, *args, **kwargs):
    """Safely execute module function with error handling - NO EMOJIS"""
    try:
        module = __import__(module_name)
        function = getattr(module, function_name)
        logger.info(f"Executing {module_name}.{function_name}")
        result = function(*args, **kwargs)
        logger.info(f"SUCCESS: {module_name}.{function_name} completed")
        return result, True
    except ImportError:
        logger.warning(f"WARNING: Module {module_name} not found - skipping")
        return None, False
    except AttributeError:
        logger.warning(f"WARNING: Function {function_name} not found in {module_name} - skipping")
        return None, False
    except Exception as e:
        logger.error(f"ERROR in {module_name}.{function_name}: {e}")
        return None, False

def scrape_stocks():
    """Scrape stock data from screener.in"""
    try:
        logger.info("Scraping latest stock data from screener.in...")
        scrape_and_save_all(verbose=False, delay=1.0)
        logger.info("SUCCESS: Stock data scraping completed")
    except Exception as e:
        logger.error(f"ERROR: Stock scraping failed: {e}")

def main():
    """Main trading system execution"""
    logger.info("TRADING SYSTEM STARTING")
    logger.info("=" * 60)
    
    try:
        # Step 0: Scrape latest stock data
        logger.info("Step 0: Scraping stock fundamentals...")
        scrape_stocks()
        
        # Step 1: Test kiteconnect import
        logger.info("Step 1: Testing kiteconnect import...")
        try:
            import kiteconnect
            logger.info(f"SUCCESS: kiteconnect version: {kiteconnect.__version__}")
        except ImportError as e:
            logger.error(f"ERROR: kiteconnect import failed: {e}")
            logger.error(f"Python path: {sys.executable}")
            logger.error(f"Python version: {sys.version}")
            return False
        
        # Step 2: Get credentials
        logger.info("Step 2: Getting credentials...")
        access_token, api_key = get_credentials()
        if not access_token or not api_key:
            logger.error("Failed to get required credentials")
            return False
        
        # Step 3: Initialize Kite
        logger.info("Step 3: Initializing Kite connection...")
        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=api_key, timeout=60)
            session = requests.Session()
            session.adapters.DEFAULT_RETRIES = 3  # Retry failed requests
            kite._session = session
            kite.set_access_token(access_token)
            profile = kite.profile()
            logger.info(f"SUCCESS: Logged in as: {profile['user_name']} (ID: {profile['user_id']})")
        except Exception as e:
            logger.error(f"ERROR: Kite initialization failed: {e}")
            return False
        
        # Step 4: Read configuration
        logger.info("Step 4: Reading configuration...")
        config = read_config()
        is_trading_time = is_market_hours()
        trade_enabled = config.get('trade_today_flag') == 'yes'
        
        logger.info(f"Market hours active: {is_trading_time}")
        logger.info(f"Trading enabled: {trade_enabled}")
        logger.info(f"Stop Loss: {config['stop_loss']}%, Target: {config['target']}%")
        logger.info(f"OHLC Value: {config['ohlc_value']}")
        
        # Step 5: Execute workflow based on conditions
        logger.info("Step 5: Executing trading workflow...")
        
        if is_trading_time and trade_enabled:
            logger.info("CONDITIONS MET: Executing live trading workflow")
            # Execute live trading
            execute_module('live_data_downloader', 'download_live_data', kite)
        else:
            logger.info("MARKET CLOSED OR TRADING DISABLED: Executing historical data workflow")
        
        # Step 6: Core workflow (always run)
        logger.info("Step 6: Downloading historical data...")
        execute_module('historical_data_download', 'download_historical_data', kite)
        
        logger.info("Step 7: Aggregating data...")
        execute_module('data_aggregate', 'interactive_aggregation',
                      interval_minutes=0, interval_days=1,
                      input_folder="stocks_historical_data", save_files=True)
        
        logger.info("Step 8: Running backtest...")
        execute_module('long_term_backtest', 'run_backtest')
        
        logger.info("Step 9: Running live scanner and telegram notifications...")
        execute_module('consolidated_volume', 'complete_trading_workflow',
                      interval_minutes=0, interval_days=1, 
                      kite_instance=kite,
                      telegram_bot_token=os.getenv('TELEGRAM_BOT_TOKEN'),
                      telegram_chat_id=os.getenv('TELEGRAM_CHAT_ID'))
        
        logger.info("=" * 60)
        logger.info("SUCCESS: TRADING SYSTEM COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"CRITICAL ERROR: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

