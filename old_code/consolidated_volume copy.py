import pandas as pd
import os
import glob
import requests
from datetime import datetime, timedelta

def consolidate_volume_boost_data(interval_minutes=0,
                                interval_days=1, 
                                input_folder="aggregated_data", 
                                output_folder="Volume_boost_consolidated",
                                save_files=True):
    """
    Consolidates last rows from all volume boost aggregated files into a single file.
    Now filters from check_from_date (in common.txt) to current date.
    """
    if not save_files:
        return
    
    # Read config to get check_from_date
    config = read_config()
    check_from_date_str = config.get('check_from_date')
    
    if not check_from_date_str:
        print("[ERROR] check_from_date not found in common.txt")
        return
    
    # Parse the date - use it as start date for filtering
    try:
        from_date = pd.to_datetime(check_from_date_str)
        current_date = datetime.now()
        print(f"[DATA] Filtering data from: {from_date.strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"[ERROR] Error parsing check_from_date: {e}")
        return
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Determine file pattern based on interval_minutes
    if interval_minutes == 0:
        pattern = os.path.join(input_folder, f"*_{interval_days}day_aggregated.csv")
    else:
        pattern = os.path.join(input_folder, f"*_{interval_minutes}min_aggregated.csv")
    
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    # Collect last rows from all files (after 3-month filtering)
    last_rows = []
    for file in files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                # Convert date column to datetime for filtering
                df['date'] = pd.to_datetime(df['date'])
                
                # Normalize dates to date-only for comparison (removes time and timezone)
                df['date_only'] = df['date'].dt.date
                from_date_only = from_date.date()
                current_date_only = current_date.date()
                
                # Filter from check_from_date to current date - optimized: sort by date desc and filter from recent dates
                df_sorted = df.sort_values('date', ascending=False)
                df_filtered = df_sorted[
                    (df_sorted['date_only'] >= from_date_only) & 
                    (df_sorted['date_only'] <= current_date_only)
                ]
                
                # Drop the temporary date_only column
                df_filtered = df_filtered.drop('date_only', axis=1)
                
                if not df_filtered.empty:
                    # Take the most recent row from filtered data (first row since sorted desc)
                    last_row = df_filtered.iloc[0:1].copy()
                    # Extract symbol from filename
                    symbol = os.path.basename(file).split('_')[0]
                    last_row['symbol'] = symbol
                    last_rows.append(last_row)
                else:
                    print(f"[WARNING] No data in specified date range for {os.path.basename(file)}")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if last_rows:
        # Concatenate all last rows
        consolidated_df = pd.concat(last_rows, ignore_index=True)
        
        # Save consolidated data
        output_path = os.path.join(output_folder, "consolidated_data.csv")
        consolidated_df.to_csv(output_path, index=False)
        print(f"[SUCCESS] Consolidated {len(last_rows)} symbols to: {output_path}")
    else:
        print("[ERROR] No data to consolidate from specified date range")


def read_config(config_file="common.txt"):
    """Read configuration from common.txt file"""
    config = {}
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        return config
    except Exception as e:
        print(f"Error reading {config_file}: {e}")
        return {}


def send_telegram_signal(bot_token, chat_id, message):
    """Send signal to Telegram"""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, data=data)
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


def live_market_scanner(kite_instance, telegram_bot_token, telegram_chat_id, 
                       consolidated_file="Volume_boost_consolidated/consolidated_data.csv",
                       config_file="common.txt"):
    """
    Main function to scan live market and send signals
    
    Args:
        kite_instance: Initialized KiteConnect instance
        telegram_bot_token: Your Telegram bot token
        telegram_chat_id: Your Telegram chat ID
        consolidated_file: Path to consolidated CSV file
        config_file: Path to common.txt config file
    """
    
    print("[START] Starting Live Market Scanner...")
    
    # Step 1: Read configuration
    config = read_config(config_file)
    ohlc_value = config.get('ohlc_value', 'close')
    
    if not ohlc_value:
        print("[ERROR] ohlc_value not found in config file")
        return
    
    print(f"[DATA] Monitoring: {ohlc_value}")
    
    # Step 2: Read consolidated data
    try:
        df = pd.read_csv(consolidated_file)
        if df.empty:
            print("[ERROR] No data in consolidated file")
            return
    except Exception as e:
        print(f"[ERROR] Error reading consolidated file: {e}")
        return
    
    # Step 3: Prepare symbols list for Kite API
    symbols = df['symbol'].unique().tolist()
    
    # Format symbols for NSE (adjust exchange as needed)
    formatted_symbols = [f"NSE:{symbol}" for symbol in symbols]
    
    print(f"[TRADING] Scanning {len(symbols)} symbols...")
    
    # Step 4: Get live market data in bulk
    try:
        live_data = kite_instance.ltp(formatted_symbols)
    except Exception as e:
        print(f"[ERROR] Error fetching live data: {e}")
        return
    
    # Step 5: Compare and generate signals
    signals_sent = 0
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for _, row in df.iterrows():
        symbol = row['symbol']
        csv_value = row[ohlc_value]
        formatted_symbol = f"NSE:{symbol}"
        
        if formatted_symbol in live_data:
            live_price = live_data[formatted_symbol]['last_price']
            
            # Check if live price > CSV price
            if live_price > csv_value:
                # Prepare signal message
                percentage_increase = ((live_price - csv_value) / csv_value) * 100
                csv_date = row['date']
                
                message = f"""
🚨 <b>TRADING SIGNAL</b> 🚨

[DATA] <b>Symbol:</b> {symbol}
📅 <b>CSV Date:</b> {csv_date}
[TRADING] <b>CSV {ohlc_value.upper()}:</b> ₹{csv_value:,.2f}
💰 <b>Live Price:</b> ₹{live_price:,.2f}
[DATA] <b>Increase:</b> +{percentage_increase:.2f}%

🕒 <b>Current Time:</b> {current_time}
                """
                
                # Send Telegram signal
                if send_telegram_signal(telegram_bot_token, telegram_chat_id, message):
                    signals_sent += 1
                    print(f"[SUCCESS] Signal sent for {symbol}: {live_price} > {csv_value}")
                else:
                    print(f"[ERROR] Failed to send signal for {symbol}")
    
    print(f"[TARGET] Scan completed! {signals_sent} signals sent")


def complete_trading_workflow(interval_minutes=0, interval_days=1, 
                            input_folder="aggregated_data",
                            kite_instance=None, telegram_bot_token=None, 
                            telegram_chat_id=None):
    """
    Complete workflow: Consolidate data + Scan live market
    
    Usage:
        complete_trading_workflow(
            interval_minutes=0,
            interval_days=1, 
            kite_instance=your_kite_instance,
            telegram_bot_token="your_bot_token",
            telegram_chat_id="your_chat_id"
        )
    """
    
    # Step 1: Consolidate volume boost data
    print("[LOADING] Consolidating data...")
    consolidate_volume_boost_data(
        interval_minutes=interval_minutes,
        interval_days=interval_days,
        input_folder=input_folder,
        save_files=True
    )
    
    # Step 2: Scan live market if API credentials provided
    if kite_instance and telegram_bot_token and telegram_chat_id:
        print("[SEARCH] Starting live market scan...")
        live_market_scanner(
            kite_instance=kite_instance,
            telegram_bot_token=telegram_bot_token,
            telegram_chat_id=telegram_chat_id
        )
    else:
        print("[WARNING] Kite/Telegram credentials not provided. Skipping live scan.")