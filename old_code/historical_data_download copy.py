import os
import datetime
import pandas as pd
from typing import Optional, Tuple, List, Dict
import time
import pytz

def get_last_business_day() -> datetime.date:
    """Get the last business day (Monday-Friday)"""
    today = datetime.date.today()
    if today.weekday() >= 5:  # Saturday=5, Sunday=6
        days = today.weekday() - 4
        today = today - datetime.timedelta(days=days)
    return today

def is_market_open_today() -> bool:
    """Check if script should run today based on config - UNCHANGED for backward compatibility"""
    try:
        with open('common.txt', 'r') as file:
            content = file.read()
            
        # Look for RUN_TODAY_OR_HOLIDAY setting
        for line in content.splitlines():
            line = line.strip()
            if line.startswith('RUN_TODAY_OR_HOLIDAY'):
                # Extract the value after = sign
                if '=' in line:
                    value = line.split('=', 1)[1].strip()
                    # Return False only if explicitly set to "No"
                    if value.upper() == 'NO':
                        return False
                    elif value.upper() == 'YES':
                        return True
        
        # Default return True if setting not found
        return True
        
    except FileNotFoundError:
        # File doesn't exist, return default True
        return True
    except Exception:
        # Any other error, return default True
        return True

def is_actual_market_day() -> bool:
    """Check if today is an actual market trading day (Monday-Friday only)"""
    today = datetime.date.today()
    return today.weekday() < 5  # Monday=0 to Friday=4

def get_last_market_time(interval: str = '5minute') -> datetime.datetime:
    """Get the last market update time based on market status and interval"""
    ist = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.datetime.now(ist)
    
    # FIXED: Use actual market day check instead of config-based check
    if is_actual_market_day():
        # Market is open today - get current time or last 5-minute mark
        if interval == '5minute':
            # Check if current time is after 3:30 PM
            if now_ist.hour > 15 or (now_ist.hour == 15 and now_ist.minute >= 30):
                # After market closes - fix to 3:25 PM
                target_time = now_ist.replace(hour=15, minute=25, second=0, microsecond=0)
            else:
                # Within market hours (9:15 AM to 3:30 PM) - round down to last 5-minute mark
                minutes = (now_ist.minute // 5) * 5
                target_time = now_ist.replace(minute=minutes, second=0, microsecond=0)
        else:
            # For daily data, current time is fine
            target_time = now_ist
    else:
        # Market is closed (weekend/holiday) - get last business day's closing time (3:30 PM IST)
        last_bday = get_last_business_day()
        
        # If last business day is today but market is closed, use today's close time
        if last_bday == now_ist.date() and now_ist.hour >= 15:
            target_time = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
        else:
            # Use last business day's closing time (3:30 PM IST)
            target_time = ist.localize(datetime.datetime.combine(last_bday, datetime.time(15, 30)))
    
    # Convert to naive datetime (remove timezone) for consistency
    return target_time.replace(tzinfo=None)

def get_last_5_minutes() -> datetime.datetime:
    """Get the last 5-minute mark for intraday updates - DEPRECATED, use get_last_market_time"""
    return get_last_market_time('5minute')

def normalize_dataframe_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize date column in DataFrame to ensure consistent datetime format"""
    if df.empty or 'date' not in df.columns:
        return df
    
    try:
        # Convert date column to datetime, handling mixed types
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Remove any rows with invalid dates
        df = df.dropna(subset=['date'])
        
        # Reset index after dropping rows
        df = df.reset_index(drop=True)
        
        return df
    except Exception as e:
        print(f"   [WARNING]  Error normalizing dates in DataFrame: {e}")
        return df

def normalize_datetime(dt) -> datetime.datetime:
    """Normalize datetime to remove timezone info for comparison"""
    # Handle pandas Timestamp or string input
    if isinstance(dt, str):
        dt = pd.to_datetime(dt)
    
    # Convert pandas Timestamp to datetime if needed
    if hasattr(dt, 'to_pydatetime'):
        dt = dt.to_pydatetime()
    
    # Remove timezone info for comparison - convert to IST first if timezone aware
    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        # Convert to IST timezone first, then remove timezone info
        ist = pytz.timezone('Asia/Kolkata')
        if dt.tzinfo != ist:
            dt = dt.astimezone(ist)
        dt = dt.replace(tzinfo=None)
    
    return dt

def load_symbols(symbols_file: str = 'symbols.txt') -> List[str]:
    """Load symbols from file with validation"""
    if not os.path.exists(symbols_file):
        raise FileNotFoundError(f"Symbols file {symbols_file} not found!")
    
    with open(symbols_file, 'r') as f:
        symbols = [line.strip() for line in f if line.strip()]
    
    if not symbols:
        raise ValueError("No symbols found in symbols.txt file!")
    
    return symbols

def get_missing_symbols(symbols: List[str], output_folder: str) -> List[str]:
    """Get list of symbols that don't have CSV files"""
    missing_symbols = []
    for symbol in symbols:
        csv_path = os.path.join(output_folder, f"{symbol}_historical.csv")
        if not os.path.exists(csv_path):
            missing_symbols.append(symbol)
    return missing_symbols

def check_reliance_update_status(output_folder: str, interval: str = '5minute') -> bool:
    """Check if RELIANCE data is up to date to determine if updates are needed"""
    reliance_path = os.path.join(output_folder, "RELIANCE_historical.csv")
    
    if not os.path.exists(reliance_path):
        print("[DATA] RELIANCE file not found - updates needed")
        return True
    
    try:
        df = pd.read_csv(reliance_path)
        if df.empty or 'date' not in df.columns:
            print("[DATA] RELIANCE file is empty - updates needed")
            return True
        
        # Normalize dates in DataFrame
        df = normalize_dataframe_dates(df)
        if df.empty:
            print("[DATA] RELIANCE file has invalid dates - updates needed")
            return True
        
        # Get last date from file
        last_datetime = df['date'].iloc[-1]
        last_datetime = normalize_datetime(last_datetime)
        
        # Get target market time (this is the latest time we should have data for)
        target_time = get_last_market_time(interval)
        print("target_time:", target_time)
        print("last_datetime:", last_datetime)
        
        # Debug: Print the actual times for troubleshooting
        print(f"   [SEARCH] Debug: Last data time: {last_datetime}")
        print(f"   [SEARCH] Debug: Target market time: {target_time}")
        
        # Check if data is up to date based on market time only
        if interval == '5minute':
            # Compare against target market time, not current time
            time_diff_minutes = (target_time - last_datetime).total_seconds() / 60
            
            # For market close scenarios, be more lenient
            # If we have data within 10 minutes of market close (3:30 PM), consider it complete
            if target_time.hour == 15 and target_time.minute == 30:  # Market close time
                # Check if last data is close to market close
                if last_datetime.hour == 15 and last_datetime.minute >= 20:  # After 3:20 PM
                    is_updated = True
                    status_msg = f"up to date (close to market close at {last_datetime.strftime('%H:%M')})"
                else:
                    is_updated = False
                    status_msg = f"missing {time_diff_minutes:.0f} minutes of market data"
            else:
                # During market hours, use stricter 5-minute rule
                if time_diff_minutes <= 5:
                    is_updated = True
                    status_msg = "up to date"
                else:
                    is_updated = False
                    if time_diff_minutes > 0:
                        status_msg = f"missing {time_diff_minutes:.0f} minutes of market data"
                    else:
                        status_msg = f"data is {abs(time_diff_minutes):.0f} minutes ahead of target"
            
            # UPDATED: Use actual market day check for status display
            market_status = "OPEN" if is_actual_market_day() else "CLOSED"
            print(f"[DATA] RELIANCE status: {status_msg} | Market: {market_status} | Target: {target_time.strftime('%Y-%m-%d %H:%M')}")
            
        else:
            # For daily data, check against target date
            target_date = target_time.date()
            last_date = last_datetime.date()
            is_updated = last_date >= target_date
            
            if is_updated:
                status_msg = "up to date"
            else:
                days_behind = (target_date - last_date).days
                status_msg = f"missing {days_behind} day(s) of data (last: {last_date}, target: {target_date})"
            
            print(f"[DATA] RELIANCE status: {status_msg}")
        
        return not is_updated
        
    except Exception as e:
        print(f"[WARNING]  Error checking RELIANCE status: {e}")
        return True

def check_file_needs_update(file_path: str, interval: str = 'day') -> Tuple[bool, pd.DataFrame]:
    """Check if a file needs updating and return existing data if valid"""
    if not os.path.exists(file_path):
        return True, pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path)
        if df.empty or 'date' not in df.columns:
            return True, pd.DataFrame()
        
        # Normalize dates in DataFrame
        df = normalize_dataframe_dates(df)
        if df.empty:
            return True, pd.DataFrame()
        
        # Check if data is up to date based on interval
        last_datetime = df['date'].iloc[-1]
        last_datetime = normalize_datetime(last_datetime)
        
        # Get target market time (not current time)
        target_time = get_last_market_time(interval)
        
        if interval == '5minute':
            # Compare against target market time only
            time_diff_minutes = (target_time - last_datetime).total_seconds() / 60
            
            # Special handling for market close scenarios
            if target_time.hour == 15 and target_time.minute == 30:  # Market close time
                # If we have data after 3:20 PM, consider it complete
                if last_datetime.hour == 15 and last_datetime.minute >= 20:
                    needs_update = False
                else:
                    needs_update = time_diff_minutes > 5
            else:
                # Regular check - need update if missing more than 5 minutes
                needs_update = time_diff_minutes > 5
        else:
            last_date = last_datetime.date()
            target_date = target_time.date()
            needs_update = last_date < target_date
        
        return needs_update, df
        
    except Exception as e:
        print(f"   [WARNING]  Error reading existing file: {e}")
        return True, pd.DataFrame()

def get_instrument_token(instruments: pd.DataFrame, symbol: str) -> Optional[int]:
    """Get instrument token for symbol"""
    row = instruments[instruments['tradingsymbol'] == symbol]
    return int(row['instrument_token'].values[0]) if not row.empty else None

def fetch_data_in_chunks(kite, symbol: str, token: int, start_date: datetime.datetime, 
                        end_date: datetime.datetime, interval: str) -> pd.DataFrame:
    """Fetch historical data in chunks - 60 days for 5min, 100 days for daily"""
    all_data = []
    current_start = start_date
    
    # Ensure start_date and end_date are datetime objects
    if isinstance(start_date, str):
        current_start = pd.to_datetime(start_date).to_pydatetime()
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).to_pydatetime()
    
    # Set chunk size based on interval
    chunk_days = 60 if interval == '5minute' else 100
    
    chunk_count = 0
    total_chunks = ((end_date - current_start).days // chunk_days) + 1
    
    while current_start < end_date:
        chunk_end = min(current_start + datetime.timedelta(days=chunk_days), end_date)
        chunk_count += 1
        
        try:
            print(f"   📥 Fetching {symbol}: {current_start.date()} to {chunk_end.date()} (Chunk {chunk_count}/{total_chunks})")
            data = kite.historical_data(token, current_start, chunk_end, interval)
            
            if data:
                all_data.extend(data)
                print(f"      [SUCCESS] Retrieved {len(data)} records")
                
                time.sleep(0.5)  # Rate limiting
            else:
                print(f"      [WARNING]  No data for this chunk")
            
        except Exception as e:
            print(f"   [ERROR] Error fetching chunk for {symbol} ({current_start.date()}): {e}")
            time.sleep(2)  # Wait longer on error
        
        current_start = chunk_end + datetime.timedelta(days=1)
    
    if all_data:
        print(f"   [DATA] Total records retrieved for {symbol}: {len(all_data)}")
    
    return pd.DataFrame(all_data) if all_data else pd.DataFrame()

def save_not_found_symbols(symbols_not_found: List[str], output_folder: str):
    """Save symbols that were not found to Excel file"""
    if not symbols_not_found:
        return
    
    df_not_found = pd.DataFrame({
        'Symbol': symbols_not_found,
        'Reason': 'Token not found in NSE instruments',
        'Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    excel_path = os.path.join(output_folder, "symbols_not_found.xlsx")
    
    try:
        df_not_found.to_excel(excel_path, index=False, sheet_name='Not_Found_Symbols')
        print(f"[DATA] Saved {len(symbols_not_found)} not found symbols to: {excel_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save Excel file: {e}")
        # Fallback to CSV
        csv_path = os.path.join(output_folder, "symbols_not_found.csv")
        df_not_found.to_csv(csv_path, index=False)
        print(f"[DATA] Saved as CSV instead: {csv_path}")

def process_full_download(kite, instruments: pd.DataFrame, symbol: str, start_date: datetime.datetime, 
                         interval: str, output_folder: str) -> bool:
    """Process full historical data download for a symbol"""
    token = get_instrument_token(instruments, symbol)
    if not token:
        return False
    
    file_path = os.path.join(output_folder, f"{symbol}_historical.csv")
    fetch_end = datetime.datetime.now()
    
    print(f"   📥 Downloading full data from {start_date.date()} to {fetch_end.date()}")
    if interval == '5minute':
        print(f"   ⏳ This may take a while for 5-minute data ({(fetch_end - start_date).days} days)...")
    
    new_df = fetch_data_in_chunks(kite, symbol, token, start_date, fetch_end, interval)
    
    if not new_df.empty:
        try:
            # Normalize dates before sorting
            new_df = normalize_dataframe_dates(new_df)
            if not new_df.empty:
                # Sort by date before saving
                new_df.sort_values('date', inplace=True)
                new_df.to_csv(file_path, index=False)
                print(f"   [SUCCESS] Saved {len(new_df)} rows")
                return True
            else:
                print(f"   [WARNING]  No valid data after date normalization")
                return False
        except Exception as e:
            print(f"   [ERROR] Error saving: {e}")
            return False
    else:
        print(f"   [WARNING]  No data retrieved")
        return False

def process_update_download(kite, instruments: pd.DataFrame, symbol: str, start_date: datetime.datetime, 
                           interval: str, output_folder: str) -> bool:
    """Process incremental update for existing symbol data"""
    token = get_instrument_token(instruments, symbol)
    if not token:
        return False
    
    file_path = os.path.join(output_folder, f"{symbol}_historical.csv")
    needs_update, existing_df = check_file_needs_update(file_path, interval)
    
    if not needs_update:
        print(f"   [SUCCESS] Already up to date")
        return True
    
    # Determine fetch range for update
    if not existing_df.empty:
        # existing_df already has normalized dates from check_file_needs_update
        last_datetime = existing_df['date'].iloc[-1]
        last_datetime = normalize_datetime(last_datetime)
        
        # Get target market time
        target_time = get_last_market_time(interval)
        
        # Check if we actually need new data
        if interval == '5minute':
            time_diff_minutes = (target_time - last_datetime).total_seconds() / 60
            if time_diff_minutes <= 5:
                print(f"   [SUCCESS] Already up to date (within 5 minutes of target)")
                return True
            
            # Start fetching from next 5-minute interval after last data
            fetch_start = last_datetime + datetime.timedelta(minutes=5)
        else:
            fetch_start = last_datetime + datetime.timedelta(days=1)
        
        fetch_end = target_time
    else:
        fetch_start = start_date
        fetch_end = get_last_market_time(interval)
    
    # Ensure both dates are datetime objects
    if isinstance(fetch_start, str):
        fetch_start = pd.to_datetime(fetch_start).to_pydatetime()
    if isinstance(fetch_end, str):
        fetch_end = pd.to_datetime(fetch_end).to_pydatetime()
    
    # Don't fetch if start time is >= end time
    if fetch_start >= fetch_end:
        print(f"   [SUCCESS] Already up to date (no new data to fetch)")
        return True
    
    print(f"   📥 Updating from {fetch_start} to {fetch_end}")
    
    new_df = fetch_data_in_chunks(kite, symbol, token, fetch_start, fetch_end, interval)
    
    
    if not new_df.empty:
        try:
            # Normalize dates in new data
            new_df = normalize_dataframe_dates(new_df)
            
            if not new_df.empty:
                # Combine with existing data
                if not existing_df.empty:
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                  
                    
                    combined_df.drop_duplicates(subset=['date'], inplace=True)
                    
                    # Ensure dates are normalized before sorting
                    # combined_df = normalize_dataframe_dates(combined_df)
                    # combined_df.sort_values('date', inplace=True)
                   
                    
                    final_df = combined_df
                else:
                    final_df = new_df
                
                final_df.to_csv(file_path, index=False)
                print(f"[SUCCESS] Updated: {len(final_df)} total rows (+{len(new_df)} new)")
                return True
            else:
                print(f"   [WARNING]  No valid new data after normalization")
                return True
                
        except Exception as e:
            print(f"   [ERROR] Error saving: {e}")
            return False
    else:
        print(f"   [WARNING]  No new data available")
        return True

def categorize_symbols_for_processing(symbols: List[str], output_folder: str, interval: str) -> Dict[str, List[str]]:
    """Categorize symbols based on their current status"""
    missing_symbols = get_missing_symbols(symbols, output_folder)
    existing_symbols = [s for s in symbols if s not in missing_symbols]
    
    # Check if updates are needed based on RELIANCE status
    update_needed = check_reliance_update_status(output_folder, interval)
    
    categorization = {
        'missing': missing_symbols,
        'existing_update_needed': existing_symbols if update_needed else [],
        'existing_up_to_date': [] if update_needed else existing_symbols
    }
    
    return categorization

def download_historical_data(
    kite,
    symbol_names_file: str = 'symbols.txt',
    output_folder: str = "stocks_historical_data",
    start: datetime.datetime = datetime.datetime(2020, 1, 1),
    interval: str = '5minute'
):
    """Download/update historical stock data with intelligent processing"""
    
    # Load instruments and symbols
    print("[DATA] Loading instruments and symbols...")
    instruments = pd.DataFrame(kite.instruments("NSE"))
    symbols = load_symbols(symbol_names_file)
    
    # Setup output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Categorize symbols for processing
    print("[SEARCH] Analyzing existing data...")
    symbol_categories = categorize_symbols_for_processing(symbols, output_folder, interval)
    
    # Report status
    print(f"[DATA] Data Status Summary:")
    print(f"   • Missing files: {len(symbol_categories['missing'])}")
    print(f"   • Need updates: {len(symbol_categories['existing_update_needed'])}")
    print(f"   • Up to date: {len(symbol_categories['existing_up_to_date'])}")
    
    # Check if any processing is needed
    total_to_process = len(symbol_categories['missing']) + len(symbol_categories['existing_update_needed'])
    if total_to_process == 0:
        print("[SUCCESS] All data is up to date! No processing needed.")
        return
    
    print(f"\n[LOADING] Processing {total_to_process} symbols...")
    
    # Track symbols not found
    symbols_not_found = []
    processed_count = 0
    
    # Process missing symbols (full download)
    for symbol in symbol_categories['missing']:
        processed_count += 1
        print(f"\n[DATA] Processing {symbol} ({processed_count}/{total_to_process}) - Full Download...")
        
        success = process_full_download(kite, instruments, symbol, start, interval, output_folder)
        if not success and get_instrument_token(instruments, symbol) is None:
            symbols_not_found.append(symbol)
    
    # Process existing symbols that need updates
    for symbol in symbol_categories['existing_update_needed']:
        processed_count += 1
        print(f"\n[DATA] Processing {symbol} ({processed_count}/{total_to_process}) - Update...")
        
        success = process_update_download(kite, instruments, symbol, start, interval, output_folder)
        if not success and get_instrument_token(instruments, symbol) is None:
            symbols_not_found.append(symbol)
    
    print("\n[COMPLETE] All processing complete!")
    
    # Save symbols not found
    if symbols_not_found:
        save_not_found_symbols(symbols_not_found, output_folder)
    
    # Cleanup invalid files after downloading
    print("\n" + "="*50)
    cleanup_invalid_files(output_folder, symbol_names_file)

def cleanup_invalid_files(
    output_folder: str = "stocks_historical_data",
    symbols_file: str = "symbols.txt",
    min_size_kb: int = 3
):
    """Delete files smaller than specified size or blank files and log them to Excel"""
    
    try:
        symbols = load_symbols(symbols_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return
    
    print(f"🧹 Cleaning up invalid files (< {min_size_kb}KB)...")
    
    deleted_files = []
    min_size_bytes = min_size_kb * 1024
    
    for symbol in symbols:
        csv_path = os.path.join(output_folder, f"{symbol}_historical.csv")
        
        if not os.path.exists(csv_path):
            deleted_files.append({
                'Symbol': symbol,
                'Filename': f"{symbol}_historical.csv",
                'Reason': 'File does not exist',
                'Size_KB': 0
            })
            continue
        
        file_size = os.path.getsize(csv_path)
        
        if file_size < min_size_bytes:
            # Check if file is truly empty/invalid
            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    reason = 'Empty file (no data)'
                else:
                    reason = f'File too small ({file_size} bytes)'
            except:
                reason = f'Corrupted file ({file_size} bytes)'
            
            deleted_files.append({
                'Symbol': symbol,
                'Filename': f"{symbol}_historical.csv",
                'Reason': reason,
                'Size_KB': round(file_size / 1024, 2)
            })
            
            # Delete the file
            try:
                os.remove(csv_path)
                print(f"🗑️  Deleted: {symbol} ({file_size} bytes)")
            except Exception as e:
                print(f"[ERROR] Failed to delete {symbol}: {e}")
    
    # Save deleted files info to Excel
    if deleted_files:
        df_deleted = pd.DataFrame(deleted_files)
        excel_path = os.path.join(output_folder, "symbol_not_present.xlsx")
        
        try:
            df_deleted.to_excel(excel_path, index=False, sheet_name='Deleted_Files')
            print(f"[DATA] Saved {len(deleted_files)} deleted file records to: {excel_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save Excel file: {e}")
            # Fallback to CSV
            csv_path = os.path.join(output_folder, "symbol_not_present.csv")
            df_deleted.to_csv(csv_path, index=False)
            print(f"[DATA] Saved as CSV instead: {csv_path}")
    else:
        print("[SUCCESS] No files needed cleanup!")

# Example usage:
if __name__ == "__main__":
    # Usage after authentication:
    # download_historical_data(kite, interval='5minute')  # For 5-minute data from Jan 1, 2020
    # download_historical_data(kite, interval='day')      # For daily data from Jan 1, 2020
    pass

