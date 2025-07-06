# live_data_downloader_parallel.py - High-Performance Async Live Data Downloader
# Downloads live stock data with parallel processing and auto-performance adjustment

import os
import pandas as pd
import time
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import math

# Configuration
SYMBOLS_FILE = "symbols.txt"
CONFIG_FILE = "common.txt"
TOKEN_FILE = "kite_token.txt"
LOG_FILE = f"trading_system_logs_{datetime.now().strftime('%Y%m%d')}.log"

# Market timing
MARKET_START_TIME = "09:15"
MARKET_END_TIME = "15:30"

# API Rate Limits (Conservative estimates for Kite Connect)
MAX_CONCURRENT_REQUESTS = 10  # Maximum simultaneous requests
REQUESTS_PER_SECOND = 5      # Rate limit per second
BATCH_DELAY = 0.2             # Delay between batches in seconds

class Logger:
    """Async-compatible logger class"""
    
    @staticmethod
    def log_to_file(message, level="INFO"):
        """Log message to file with timestamp"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"[{timestamp}] {level}: {message}\n"
            
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(log_entry)
                
            # Also print to console for real-time monitoring
            print(f"[{timestamp}] {level}: {message}")
        except Exception as e:
            print(f"Failed to write to log file: {e}")

class ConfigManager:
    """Enhanced configuration manager with parallel processing settings"""
    
    @staticmethod
    def load():
        """Load configuration from common.txt"""
        defaults = {
            "live_data_download": "no",
            "rerun_minute": 1,
            "symbols_per_worker": 5,
            "max_concurrent_requests": 20,
            "stop_loss": 4.0,
            "target": 10.0,
            "ohlc_value": "open",
            "trade_today_flag": "no"
        }
        
        try:
            config = {}
            with open(CONFIG_FILE, 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        # Convert numeric values
                        if key in ['rerun_minute', 'symbols_per_worker', 'max_concurrent_requests']:
                            config[key] = int(value)
                        elif key in ['stop_loss', 'target']:
                            config[key] = float(value)
                        else:
                            config[key] = value
            
            result_config = {**defaults, **config}
            Logger.log_to_file(f"Configuration loaded: {result_config}")
            return result_config
        except Exception as e:
            Logger.log_to_file(f"Error loading config, using defaults: {e}", "WARNING")
            return defaults
    
    @staticmethod
    def update_rerun_minute(new_value):
        """Update rerun_minute in common.txt"""
        try:
            # Read current config
            lines = []
            updated = False
            
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    lines = f.readlines()
            
            # Update or add rerun_minute
            for i, line in enumerate(lines):
                if line.startswith('rerun_minute='):
                    lines[i] = f"rerun_minute={new_value}\n"
                    updated = True
                    break
            
            if not updated:
                lines.append(f"rerun_minute={new_value}\n")
            
            # Write back to file
            with open(CONFIG_FILE, 'w') as f:
                f.writelines(lines)
            
            Logger.log_to_file(f"Updated rerun_minute to {new_value} minutes")
            return True
            
        except Exception as e:
            Logger.log_to_file(f"Error updating rerun_minute: {e}", "ERROR")
            return False

class TokenManager:
    """Token manager for Kite authentication"""
    
    @staticmethod
    def get_valid_token():
        """Get valid token from kite_token.txt"""
        if os.path.exists(TOKEN_FILE):
            try:
                with open(TOKEN_FILE, "r") as f:
                    token_data = json.loads(f.read().strip())
                    today = datetime.now().strftime("%Y-%m-%d")
                    
                    if token_data.get("date") == today and token_data.get("access_token"):
                        Logger.log_to_file(f"Valid token found for {today}")
                        return token_data["access_token"]
                    else:
                        Logger.log_to_file("Token expired or invalid", "ERROR")
                        return None
            except Exception as e:
                Logger.log_to_file(f"Error reading token file: {e}", "ERROR")
                return None
        else:
            Logger.log_to_file("Token file not found", "ERROR")
            return None

class ParallelLiveDataDownloader:
    """High-performance parallel live data downloader"""
    
    def __init__(self):
        self.access_token = None
        self.api_key = None
        self.symbols = []
        self.data_cache = {}
        self.running = False
        self.session = None
        self.semaphore = None
        
        # Performance tracking
        self.cycle_start_time = None
        self.slow_cycles = 0
        self.current_rerun_minute = 1
        self.max_rerun_minute = 10
        
        # Configuration
        self.config = {}
        
    async def initialize(self):
        """Initialize the parallel downloader"""
        try:
            # Load configuration
            self.config = ConfigManager.load()
            
            if self.config.get("live_data_download", "no").lower() != "yes":
                Logger.log_to_file("Live data download is disabled in configuration")
                return False
            
            # Get current rerun_minute from config
            self.current_rerun_minute = self.config.get("rerun_minute", 1)
            
            # Get Kite token
            self.access_token = TokenManager.get_valid_token()
            if not self.access_token:
                Logger.log_to_file("No valid access token available", "ERROR")
                return False
            
            # Get API key
            from dotenv import load_dotenv
            load_dotenv()
            
            self.api_key = os.getenv('KITE_API_KEY')
            if not self.api_key:
                Logger.log_to_file("KITE_API_KEY not found in environment", "ERROR")
                return False
            
            # Load symbols
            if not await self.load_symbols():
                return False
            
            # Initialize semaphore for rate limiting
            max_concurrent = self.config.get("max_concurrent_requests", MAX_CONCURRENT_REQUESTS)
            self.semaphore = asyncio.Semaphore(max_concurrent)
            
            # Create aiohttp session
            connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
            
            Logger.log_to_file("Parallel live data downloader initialized successfully")
            Logger.log_to_file(f"Configuration: {len(self.symbols)} symbols, {max_concurrent} max concurrent requests")
            Logger.log_to_file(f"Workers: {self.calculate_workers()} batches, {self.config.get('symbols_per_worker', 5)} symbols per worker")
            Logger.log_to_file(f"Update interval: {self.current_rerun_minute} minute(s)")
            
            return True
            
        except Exception as e:
            Logger.log_to_file(f"Initialization failed: {e}", "ERROR")
            return False
    
    async def load_symbols(self):
        """Load symbols from symbols.txt"""
        try:
            if not os.path.exists(SYMBOLS_FILE):
                Logger.log_to_file(f"Symbols file {SYMBOLS_FILE} not found", "ERROR")
                return False
            
            with open(SYMBOLS_FILE, 'r') as f:
                self.symbols = [line.strip() for line in f if line.strip()]
            
            if not self.symbols:
                Logger.log_to_file("No symbols found in symbols file", "ERROR")
                return False
            
            Logger.log_to_file(f"Loaded {len(self.symbols)} symbols")
            return True
            
        except Exception as e:
            Logger.log_to_file(f"Error loading symbols: {e}", "ERROR")
            return False
    
    def calculate_workers(self):
        """Calculate number of worker batches based on symbols"""
        symbols_per_worker = self.config.get("symbols_per_worker", 5)
        return math.ceil(len(self.symbols) / symbols_per_worker)
    
    def create_symbol_batches(self):
        """Create batches of symbols for parallel processing"""
        symbols_per_worker = self.config.get("symbols_per_worker", 5)
        batches = []
        
        for i in range(0, len(self.symbols), symbols_per_worker):
            batch = self.symbols[i:i + symbols_per_worker]
            batches.append(batch)
        
        return batches
    
    def is_market_open(self):
        """Check if market is currently open"""
        now = datetime.now()
        current_time = now.time()
        current_weekday = now.weekday()
        
        # Check if it's a weekday (Monday=0, Sunday=6)
        if current_weekday >= 5:  # Saturday or Sunday
            return False
        
        # Check market hours
        market_start = datetime.strptime(MARKET_START_TIME, "%H:%M").time()
        market_end = datetime.strptime(MARKET_END_TIME, "%H:%M").time()
        
        return market_start <= current_time <= market_end
    
    def get_filename(self):
        """Generate daily filename"""
        today = datetime.now().strftime("%Y-%m-%d")
        directory = "StockliveData"
        if not os.path.exists(directory):
            os.makedirs(directory)
        return os.path.join(directory, f"StockLiveData_{today}.csv")
    
    def initialize_daily_cache(self):
        """Initialize or load existing data cache for the day"""
        filename = self.get_filename()
        
        # Initialize cache for all symbols
        for symbol in self.symbols:
            self.data_cache[symbol] = {
                'symbol': symbol,
                'last_price': 0.0,
                'volume_sum': 0.0,
                'volume_count': 0,
                'volume_avg': 0.0,
                'vwap_sum': 0.0,
                'vwap_volume_sum': 0.0,
                'vwap': 0.0,
                'last_updated': None,
                'error_count': 0
            }
        
        # Load existing data if file exists
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                for _, row in df.iterrows():
                    symbol = row['Symbol']
                    if symbol in self.data_cache:
                        self.data_cache[symbol].update({
                            'last_price': float(row.get('Last_Price', 0)),
                            'volume_avg': float(row.get('Volume_Avg', 0)),
                            'vwap': float(row.get('VWAP', 0))
                        })
                Logger.log_to_file(f"Loaded existing data from {filename}")
            except Exception as e:
                Logger.log_to_file(f"Error loading existing file: {e}", "WARNING")
    
    async def fetch_symbol_data(self, symbol):
        """Fetch data for a single symbol with rate limiting"""
        async with self.semaphore:  # Rate limiting
            try:
                # Kite Connect API endpoints
                ltp_url = "https://api.kite.trade/quote/ltp"
                quote_url = "https://api.kite.trade/quote"
                
                headers = {
                    "X-Kite-Version": "3",
                    "Authorization": f"token {self.api_key}:{self.access_token}"
                }
                
                formatted_symbol = f"NSE:{symbol}"
                
                # Fetch LTP (Last Traded Price) - faster endpoint
                async with self.session.get(
                    ltp_url,
                    params={"i": formatted_symbol},
                    headers=headers
                ) as response:
                    
                    if response.status != 200:
                        Logger.log_to_file(f"LTP API error for {symbol}: Status {response.status}", "WARNING")
                        return None
                    
                    ltp_data = await response.json()
                    
                    if "data" not in ltp_data or formatted_symbol not in ltp_data["data"]:
                        Logger.log_to_file(f"No LTP data for {symbol}", "WARNING")
                        return None
                    
                    last_price = float(ltp_data["data"][formatted_symbol]["last_price"])
                
                # Fetch quote data for volume - if needed
                volume = 0
                try:
                    async with self.session.get(
                        quote_url,
                        params={"i": formatted_symbol},
                        headers=headers
                    ) as vol_response:
                        
                        if vol_response.status == 200:
                            quote_data = await vol_response.json()
                            if "data" in quote_data and formatted_symbol in quote_data["data"]:
                                volume = float(quote_data["data"][formatted_symbol].get("volume", 0))
                except Exception as e:
                    Logger.log_to_file(f"Volume fetch error for {symbol}: {e}", "WARNING")
                
                return {
                    "symbol": symbol,
                    "last_price": last_price,
                    "volume": volume,
                    "timestamp": datetime.now()
                }
                
            except asyncio.TimeoutError:
                Logger.log_to_file(f"Timeout fetching data for {symbol}", "WARNING")
                return None
            except Exception as e:
                Logger.log_to_file(f"Error fetching data for {symbol}: {e}", "WARNING")
                return None
    
    async def fetch_batch_data(self, symbol_batch, batch_id):
        """Fetch data for a batch of symbols"""
        Logger.log_to_file(f"Processing batch {batch_id + 1}: {symbol_batch}")
        
        # Create tasks for all symbols in this batch
        tasks = [self.fetch_symbol_data(symbol) for symbol in symbol_batch]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_updates = 0
        for result in results:
            if isinstance(result, dict) and result is not None:
                symbol = result["symbol"]
                self.update_cache(symbol, result)
                successful_updates += 1
            elif isinstance(result, Exception):
                Logger.log_to_file(f"Batch {batch_id + 1} exception: {result}", "WARNING")
        
        Logger.log_to_file(f"Batch {batch_id + 1} completed: {successful_updates}/{len(symbol_batch)} successful")
        return successful_updates
    
    def update_cache(self, symbol, data):
        """Update cache with new data"""
        try:
            cache = self.data_cache[symbol]
            current_time = data["timestamp"]
            current_price = data["last_price"]
            current_volume = data["volume"]
            
            # Update price
            cache['last_price'] = current_price
            
            # Update running volume average
            if current_volume > 0:
                cache['volume_sum'] += current_volume
                cache['volume_count'] += 1
                cache['volume_avg'] = cache['volume_sum'] / cache['volume_count']
            
            # Update VWAP
            if current_volume > 0:
                cache['vwap_sum'] += (current_price * current_volume)
                cache['vwap_volume_sum'] += current_volume
                
                if cache['vwap_volume_sum'] > 0:
                    cache['vwap'] = cache['vwap_sum'] / cache['vwap_volume_sum']
            else:
                # If no volume, use price for VWAP calculation
                if cache['volume_count'] > 0:
                    cache['vwap'] = (cache['vwap'] * cache['volume_count'] + current_price) / (cache['volume_count'] + 1)
                else:
                    cache['vwap'] = current_price
            
            cache['last_updated'] = current_time
            cache['error_count'] = 0  # Reset error count on success
            
        except Exception as e:
            Logger.log_to_file(f"Error updating cache for {symbol}: {e}", "ERROR")
    
    async def fetch_all_data_parallel(self):
        """Fetch data for all symbols using parallel batches"""
        try:
            self.cycle_start_time = time.time()
            
            # Create symbol batches
            batches = self.create_symbol_batches()
            
            Logger.log_to_file(f"Starting parallel fetch: {len(batches)} batches, {len(self.symbols)} total symbols")
            
            # Process all batches concurrently
            batch_tasks = [
                self.fetch_batch_data(batch, i) 
                for i, batch in enumerate(batches)
            ]
            
            # Wait for all batches to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Calculate total successful updates
            total_successful = sum(
                result for result in batch_results 
                if isinstance(result, int)
            )
            
            cycle_time = time.time() - self.cycle_start_time
            
            Logger.log_to_file(f"Parallel fetch completed: {total_successful}/{len(self.symbols)} symbols updated in {cycle_time:.2f}s")
            
            # Check performance and adjust timing if needed
            await self.check_performance(cycle_time)
            
            return total_successful > 0
            
        except Exception as e:
            Logger.log_to_file(f"Error in parallel fetch: {e}", "ERROR")
            return False
    
    async def check_performance(self, cycle_time):
        """Check performance and adjust rerun_minute if needed"""
        try:
            # Define "slow" as taking more than 80% of the rerun_minute interval
            slow_threshold = (self.current_rerun_minute * 60) * 0.8
            
            if cycle_time > slow_threshold:
                self.slow_cycles += 1
                Logger.log_to_file(f"Slow cycle detected: {cycle_time:.2f}s (threshold: {slow_threshold:.2f}s)")
                
                # If 2 consecutive slow cycles, increase rerun_minute
                if self.slow_cycles >= 2 and self.current_rerun_minute < self.max_rerun_minute:
                    new_rerun_minute = min(self.current_rerun_minute + 1, self.max_rerun_minute)
                    
                    Logger.log_to_file(f"Performance adjustment: Increasing interval from {self.current_rerun_minute} to {new_rerun_minute} minutes")
                    
                    # Update configuration file
                    if ConfigManager.update_rerun_minute(new_rerun_minute):
                        self.current_rerun_minute = new_rerun_minute
                        self.slow_cycles = 0
            else:
                # Reset slow cycle counter on good performance
                self.slow_cycles = 0
                
        except Exception as e:
            Logger.log_to_file(f"Error in performance check: {e}", "WARNING")
    
    async def save_to_csv(self):
        """Save current cache data to CSV"""
        try:
            filename = self.get_filename()
            
            # Prepare data for CSV
            csv_data = []
            for symbol, cache in self.data_cache.items():
                csv_data.append({
                    'Symbol': symbol,
                    'Last_Price': round(cache['last_price'], 2),
                    'Volume_Avg': round(cache['volume_avg'], 2),
                    'VWAP': round(cache['vwap'], 2),
                    'Last_Updated': cache['last_updated'].strftime('%H:%M:%S') if cache['last_updated'] else 'Never'
                })
            
            # Create DataFrame and save
            df = pd.DataFrame(csv_data)
            df.to_csv(filename, index=False)
            
            Logger.log_to_file(f"Data saved to {filename}")
            return True
            
        except Exception as e:
            Logger.log_to_file(f"Error saving to CSV: {e}", "ERROR")
            return False
    
    async def run(self):
        """Main async run loop"""
        Logger.log_to_file("=== PARALLEL LIVE DATA DOWNLOADER STARTED ===")
        
        if not await self.initialize():
            Logger.log_to_file("Initialization failed, exiting", "ERROR")
            return
        
        self.running = True
        self.initialize_daily_cache()
        
        Logger.log_to_file(f"Starting parallel live data collection")
        Logger.log_to_file(f"Market hours: {MARKET_START_TIME} - {MARKET_END_TIME}")
        Logger.log_to_file(f"Update interval: {self.current_rerun_minute} minute(s)")
        Logger.log_to_file(f"Parallel configuration: {self.calculate_workers()} batches, max {self.config.get('max_concurrent_requests', MAX_CONCURRENT_REQUESTS)} concurrent requests")
        
        try:
            while self.running:
                # Reload configuration to check for changes
                self.config = ConfigManager.load()
                
                if self.config.get("live_data_download", "no").lower() != "yes":
                    Logger.log_to_file("Live data download disabled in configuration, stopping")
                    break
                
                # Update current rerun_minute from config
                config_rerun_minute = self.config.get("rerun_minute", 1)
                if config_rerun_minute != self.current_rerun_minute:
                    Logger.log_to_file(f"Rerun interval updated: {self.current_rerun_minute} -> {config_rerun_minute} minutes")
                    self.current_rerun_minute = config_rerun_minute
                
                # Check market hours
                if not self.is_market_open():
                    if datetime.now().time() < datetime.strptime(MARKET_START_TIME, "%H:%M").time():
                        Logger.log_to_file("Market not yet open, waiting...")
                    else:
                        Logger.log_to_file("Market closed for the day")
                        break
                    
                    await asyncio.sleep(60)  # Wait 1 minute before checking again
                    continue
                
                # Fetch and save data in parallel
                if await self.fetch_all_data_parallel():
                    await self.save_to_csv()
                else:
                    Logger.log_to_file("No data updated in this cycle", "WARNING")
                
                # Wait for next update cycle
                wait_time = self.current_rerun_minute * 60
                if self.running:
                    Logger.log_to_file(f"Waiting {self.current_rerun_minute} minute(s) for next update...")
                    await asyncio.sleep(wait_time)
        
        except KeyboardInterrupt:
            Logger.log_to_file("Received interrupt signal, stopping gracefully")
        except Exception as e:
            Logger.log_to_file(f"Unexpected error in main loop: {e}", "ERROR")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if self.session:
            await self.session.close()
        Logger.log_to_file("=== PARALLEL LIVE DATA DOWNLOADER STOPPED ===")
    
    def stop(self):
        """Stop the downloader"""
        self.running = False
        Logger.log_to_file("Stop signal received")

# Global downloader instance for signal handling
downloader = None

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    global downloader
    print("\nReceived shutdown signal...")
    if downloader:
        downloader.stop()
    
    # Give time for cleanup
    import time
    time.sleep(2)
    sys.exit(0)

async def main():
    """Main async function"""
    global downloader
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 Parallel Live Stock Data Downloader")
    print("=" * 60)
    print(f"⏰ Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Market hours: {MARKET_START_TIME} - {MARKET_END_TIME}")
    print(f"📄 Symbols file: {SYMBOLS_FILE}")
    print(f"⚙️ Config file: {CONFIG_FILE}")
    print(f"🔑 Token file: {TOKEN_FILE}")
    print(f"📝 Log file: {LOG_FILE}")
    print(f"⚡ Mode: HIGH-PERFORMANCE PARALLEL PROCESSING")
    print("=" * 60)
    print("💡 Press Ctrl+C to stop gracefully")
    print("=" * 60)
    
    # Create and run downloader
    downloader = ParallelLiveDataDownloader()
    await downloader.run()

if __name__ == "__main__":
    # Run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")
    except Exception as e:
        print(f"Fatal error: {e}")