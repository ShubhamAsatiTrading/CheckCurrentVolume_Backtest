import pandas as pd
import os
import traceback


def extract_symbol_from_filename(filename):
    """
    FIXED: Use consistent symbol extraction (same as aggregation script)
    Handles: RELIANCE_60min_aggregated.csv -> RELIANCE
    """
    # Remove .csv extension
    name_without_ext = filename.replace('.csv', '')
    
    # Split by underscore and take first part as symbol
    parts = name_without_ext.split('_')
    
    if len(parts) >= 2:
        symbol = parts[0]
        return symbol
    else:
        # If no underscore found, use the whole filename (without .csv)
        return name_without_ext


def parse_filename(filename):
    """
    Parse filename to extract symbol (everything before first underscore)
    Examples:
    - CUMMINSIND_historical_60min_aggregated.csv → CUMMINSIND
    - RELIANCE_60min_aggregated.csv → RELIANCE
    - AAPL_5day_aggregated.csv → AAPL
    
    Args:
        filename (str): Name of the CSV file
        
    Returns:
        str: symbol name
    """
    # FIXED: Now uses consistent extraction method
    return extract_symbol_from_filename(filename)


def run_backtest():
    """
    Main function to run backtesting for all files in aggregated_data folder
    Automatically detects and processes all CSV files
    
    Returns:
        dict: Results for all processed files
    """
    
    # Step 1: Get all CSV files from aggregated_data folder
    aggregated_folder = "aggregated_data"
    if not os.path.exists(aggregated_folder):
        raise FileNotFoundError(f"Folder not found: {aggregated_folder}")
    
    csv_files = [f for f in os.listdir(aggregated_folder) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in aggregated_data folder")
        return {
            'config_values': {},
            'total_files': 0,
            'results': {}
        }
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f}")
    
    # Step 2: Read configuration values from common.txt
    config_values = read_ohlc_from_common()
    print(f"\nConfiguration: {config_values}")
    
    all_results = {}
    processed_count = 0
    
    # Step 3: Process each file
    for filename in csv_files:
        try:
            print(f"\n--- Processing {filename} ---")
            
            # Extract symbol from filename (everything before first underscore)
            symbol = parse_filename(filename)
            print(f"Parsed: Symbol={symbol}")
            
            # Read the aggregated data file
            data_file_path = os.path.join(aggregated_folder, filename)
            df = pd.read_csv(data_file_path)
            print(f"Loaded aggregated data: {len(df)} rows")
            
            # Read the historical data file
            historical_file_path = f"stocks_historical_data/{symbol}_historical.csv"
            if os.path.exists(historical_file_path):
                historical_df = pd.read_csv(historical_file_path)
                print(f"Loaded historical data: {len(historical_df)} rows")
            else:
                historical_df = None
                print(f"Warning: Historical data file not found: {historical_file_path}")
            
            # Get stock categorization info
            stock_info = get_stock_categorization(symbol)
            print(f"Stock info: {stock_info}")
            
            # Perform backtesting with both aggregated and historical data
            backtest_results = perform_backtest(df, historical_df, config_values, stock_info)
            
            # Store results using symbol as key
            all_results[symbol] = {
                'symbol': symbol,
                'filename': filename,
                'data_rows': len(df),
                'historical_rows': len(historical_df) if historical_df is not None else 0,
                'stock_info': stock_info,
                'backtest_results': backtest_results
            }
            
            processed_count += 1
            print(f" Completed: {symbol}")
            
        except Exception as e:
            print(f"✗ Error processing {filename}: {str(e)}")
            continue
    
    print(f"\n=== Backtesting Summary ===")
    print(f"Files found: {len(csv_files)}")
    print(f"Files processed: {processed_count}")
    print(f"Files failed: {len(csv_files) - processed_count}")
    
    return {
        'config_values': config_values,
        'total_files': processed_count,
        'results': all_results
    }


def read_ohlc_from_common():
    """
    Read configuration values from common.txt file
    Expected format:
    stop_loss=2.0
    target=3.0
    ohlc_value=close
    
    Returns:
        dict: Configuration values including stop_loss, target, ohlc_value
    """
    try:
        config_values = {}
        
        with open('common.txt', 'r') as file:
            lines = file.readlines()
            
        for line in lines:
            line = line.strip()
            if line and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Convert numeric values
                if key in ['stop_loss', 'target']:
                    try:
                        config_values[key] = float(value)
                    except ValueError:
                        config_values[key] = value
                else:
                    config_values[key] = value
        
        # Ensure required keys exist with defaults
        required_keys = {
            'stop_loss': 2.0,
            'target': 3.0,
            'ohlc_value': 'close'
        }
        
        for key, default_value in required_keys.items():
            if key not in config_values:
                print(f"Warning: {key} not found in common.txt, using default: {default_value}")
                config_values[key] = default_value
        
        return config_values
        
    except FileNotFoundError:
        print("Warning: common.txt file not found, using default values")
        return {
            'stop_loss': 2.0,
            'target': 3.0,
            'ohlc_value': 'close'
        }


def get_stock_categorization(symbol_name):
    """
    Get market cap type, category, and days_check for the given symbol
    
    Args:
        symbol_name (str): Stock symbol to lookup
        
    Returns:
        dict: Stock categorization info including days_check
    """
    try:
        categorization_df = pd.read_csv('input_stocks/stocks_categorization.csv')
        
        # Find matching symbol (case insensitive)
        match = categorization_df[categorization_df['symbol'].str.upper() == symbol_name.upper()]
        
        if not match.empty:
            # Convert numpy types to native Python types
            days_check_value = match.iloc[0]['days_check']
            if pd.notna(days_check_value):  # Check if not NaN
                days_check_value = int(days_check_value)  # Convert np.int64 to int
            else:
                days_check_value = None
                
            return {
                'market_cap_type': str(match.iloc[0]['market_cap_type']),
                'category': str(match.iloc[0]['category']),
                'days_check': days_check_value,
                'symbol': str(match.iloc[0]['symbol'])
            }
        else:
            print(f"Warning: Symbol {symbol_name} not found in categorization file")
            return {
                'market_cap_type': 'Unknown',
                'category': 'Unknown',
                'days_check': None,
                'symbol': symbol_name
            }
            
    except FileNotFoundError:
        print("Warning: stocks_categorization.csv file not found")
        return {'market_cap_type': 'Unknown', 'category': 'Unknown', 'days_check': None, 'symbol': symbol_name}
    except KeyError as e:
        print(f"Warning: Column {e} not found in categorization file")
        return {'market_cap_type': 'Unknown', 'category': 'Unknown', 'days_check': None, 'symbol': symbol_name}
    except Exception as e:
        print(f"Warning: Error reading categorization file: {e}")
        return {'market_cap_type': 'Unknown', 'category': 'Unknown', 'days_check': None, 'symbol': symbol_name}


def perform_backtest(df, historical_df, config_values, stock_info):
    """
    Perform backtesting logic with trade tracking and Excel export
    """
    print("*"*60)
    trades_data = []
    
    # Check if we should process this stock
    market_cap_type = stock_info.get('market_cap_type', '')
    category = stock_info.get('category', '')
    days_check = stock_info.get('days_check')
    symbol = stock_info.get('symbol', 'Unknown')
    
    # if market_cap_type != "Large_Cap" or category != "A":
    #     return {
    #         'total_trades': 0,
    #         'winning_trades': 0,
    #         'losing_trades': 0,
    #         'total_return': 0.0,
    #         'win_rate': 0.0,
    #         'message': f'Skipped: market_cap_type={market_cap_type}, category={category} (not Large_Cap + A)',
    #         'trades_exported': 0
    #     }
    
    if historical_df is None or days_check is None:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_return': 0.0,
            'win_rate': 0.0,
            'message': 'Skipped: historical data or days_check not available',
            'trades_exported': 0
        }
    
    # Get configuration values
    stop_loss_pct = config_values['stop_loss'] / 100
    target_pct = config_values['target'] / 100
    ohlc_column = config_values['ohlc_value']
    
    # Check what date/time column exists and convert to datetime with timezone
    if 'timestamp' in df.columns:
        time_col = 'timestamp'
        df[time_col] = pd.to_datetime(df[time_col], utc=False)
        historical_df[time_col] = pd.to_datetime(historical_df[time_col], utc=False)
    elif 'date' in df.columns:
        time_col = 'date'
        df[time_col] = pd.to_datetime(df[time_col], utc=False)
        historical_df[time_col] = pd.to_datetime(historical_df[time_col], utc=False)
    else:
        print(f"Error: No timestamp or date column found. Available columns: {df.columns.tolist()}")
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_return': 0.0,
            'win_rate': 0.0,
            'message': 'Error: No timestamp or date column found',
            'trades_exported': 0
        }
    
    # Calculate technical indicators on historical data
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Add technical indicators to historical data
    historical_df['ema_50'] = historical_df['close'].ewm(span=50).mean()
    historical_df['ema_200'] = historical_df['close'].ewm(span=200).mean()
    historical_df['rsi_14'] = calculate_rsi(historical_df['close'])
    
    print(f"Processing {symbol}: stop_loss={stop_loss_pct*100}%, target={target_pct*100}%, ohlc_column={ohlc_column}")
    print(f"Using time column: {time_col} | Features: Trailing Stop Loss + Technical Indicators")
    print("#"*60)
    
    # Loop through each row in aggregated data
    for idx, row in df.iterrows():
        try:
            boost_time = row[time_col]
            check_start_time = boost_time + pd.Timedelta(days=days_check)
            
            # Find historical data after check_start_time
            future_historical = historical_df[historical_df[time_col] > check_start_time].copy()
            if future_historical.empty:
                continue
                
            # Get the aggregated row's OHLC value for comparison
            agg_ohlc_value = row[ohlc_column]
            agg_ohlc_value += 1
            
            # Look for entry condition: historical ohlc > aggregated ohlc
            entry_opportunities = future_historical[future_historical[ohlc_column] > agg_ohlc_value]
            if entry_opportunities.empty:
                continue
                
            # Take the first entry opportunity
            entry_row = entry_opportunities.iloc[0]
            entry_time = entry_row[time_col]
            entry_price = entry_row[ohlc_column]
            
            # Get technical indicator values at entry
            ema_50_val = entry_row['ema_50']
            ema_200_val = entry_row['ema_200']
            rsi_val = entry_row['rsi_14']
            
            # Check technical indicator conditions
            if pd.isna(ema_50_val) or pd.isna(ema_200_val) or pd.isna(rsi_val):
                continue  # Skip if indicators not available
                
            # EMA condition: current_price > EMA50 AND current_price > EMA200
            if not (entry_price > ema_50_val and entry_price > ema_200_val):
                continue
                
            # RSI condition: RSI < 60
            if not (rsi_val < 90):
                continue
            
            # Calculate target price and initialize trailing stop
            target_price = entry_price * (1 + target_pct)
            initial_trailing_stop = entry_price * (1 - stop_loss_pct)
            trailing_stop = initial_trailing_stop
            
            # Track the trade from entry time onwards
            trade_tracking = historical_df[historical_df[time_col] > entry_time].copy()
            if trade_tracking.empty:
                continue
                
            exit_time = None
            exit_price = None
            hit_type = None
            highest_price = entry_price
            
            # Check each time after entry for trailing stop or target hit
            for _, tracking_row in trade_tracking.iterrows():
                low_price = tracking_row['low']
                high_price = tracking_row['high']
                
                # Update highest price seen and trailing stop
                if high_price > highest_price:
                    highest_price = high_price
                    new_trailing_stop = highest_price * (1 - stop_loss_pct)
                    if new_trailing_stop > trailing_stop:
                        trailing_stop = new_trailing_stop
                
                # Check if trailing stop hit first
                if low_price <= trailing_stop:
                    exit_time = tracking_row[time_col]
                    exit_price = trailing_stop
                    hit_type = 'Trailing Stop'
                    break
                    
                # Check if target hit
                if high_price >= target_price:
                    exit_time = tracking_row[time_col]
                    exit_price = target_price
                    hit_type = 'Target'
                    break
            
            # If no exit found, skip this trade
            if exit_time is None:
                continue
                
            # Calculate profit/loss
            profit_loss = exit_price - entry_price
            profit_loss_pct = (profit_loss / entry_price) * 100
            
            # Convert timezone-aware datetimes to strings for Excel compatibility
            def format_time(dt):
                return dt.strftime('%Y-%m-%d %H:%M:%S%z') if pd.notnull(dt) else None
            
            # Record trade data with all new features
            trade_record = {
                'boost_time': format_time(boost_time),
                'check_start_time': format_time(check_start_time),
                'symbol': symbol,
                'entry_time': format_time(entry_time),
                'entry_price': round(entry_price, 2),
                'ema_50': round(ema_50_val, 2),
                'ema_200': round(ema_200_val, 2),
                'rsi_14': round(rsi_val, 2),
                'highest_price': round(highest_price, 2),
                'final_trailing_stop': round(trailing_stop, 2),
                'exit_time': format_time(exit_time),
                'exit_price': round(exit_price, 2),
                'profit_loss': round(profit_loss, 2),
                'profit_loss_pct': round(profit_loss_pct, 2),
                'hit_type': hit_type
            }
            
            trades_data.append(trade_record)
            
        except Exception as e:
            print(f"Error processing row {idx} for {symbol}: {e}")
            continue
    
    # Export trades to Excel
    trades_exported = 0
    if trades_data:
        trades_df = pd.DataFrame(trades_data)
        
        # Create results folder if it doesn't exist
        results_folder = "backtest_results"
        os.makedirs(results_folder, exist_ok=True)
        
        # Export to Excel
        excel_filename = f"{results_folder}/{symbol}_trades.xlsx"
        trades_df.to_excel(excel_filename, index=False)
        trades_exported = len(trades_data)
        print(f" Exported {trades_exported} trades to {excel_filename}")
    
    # Calculate summary statistics
    total_trades = len(trades_data)
    winning_trades = len([t for t in trades_data if t['profit_loss'] > 0])
    losing_trades = total_trades - winning_trades
    total_return = sum([t['profit_loss'] for t in trades_data])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'total_return': round(total_return, 2),
        'win_rate': round(win_rate, 2),
        'trades_exported': trades_exported,
        'message': f'Completed: {total_trades} trades, {winning_trades} wins, {losing_trades} losses, Win rate: {win_rate:.1f}%'
    }


def test_backtesting_setup():
    """
    Test function to verify backtesting setup and file structure
    """
    print("Testing backtesting setup...")
    
    # Test 1: Check folder structure
    required_folders = ["aggregated_data", "stocks_historical_data", "input_stocks"]
    for folder in required_folders:
        if os.path.exists(folder):
            print(f" {folder} folder exists")
        else:
            print(f"✗ {folder} folder missing")
    
    # Test 2: Check required files
    required_files = ["common.txt", "input_stocks/stocks_categorization.csv"]
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f" {file_path} file exists")
        else:
            print(f"✗ {file_path} file missing")
    
    # Test 3: Check CSV files in aggregated_data
    aggregated_folder = "aggregated_data"
    if os.path.exists(aggregated_folder):
        csv_files = [f for f in os.listdir(aggregated_folder) if f.endswith('.csv')]
        print(f" Found {len(csv_files)} CSV files in {aggregated_folder}")
        if csv_files:
            print("Sample files:")
            for i, f in enumerate(csv_files[:3]):  # Show first 3 files
                symbol = extract_symbol_from_filename(f)
                print(f"  - {f} -> Symbol: {symbol}")
    else:
        print(f"✗ {aggregated_folder} folder not found")
    
    # Test 4: Test configuration loading
    try:
        config = read_ohlc_from_common()
        print(f" Configuration loaded: {config}")
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
    
    # Test 5: Test stock categorization loading
    try:
        test_symbol = "RELIANCE"  # Use a known symbol
        stock_info = get_stock_categorization(test_symbol)
        print(f" Stock categorization test ({test_symbol}): {stock_info}")
    except Exception as e:
        print(f"✗ Error loading stock categorization: {e}")


def test_symbol_extraction():
    """Test symbol extraction with various filename formats"""
    test_cases = [
        "RELIANCE_60min_aggregated.csv",
        "HDFC_120min_aggregated.csv", 
        "TCS_5day_aggregated.csv",
        "INFY_1day_aggregated.csv",
        "ICICIBANK_historical_60min_aggregated.csv"
    ]
    
    print("Testing symbol extraction:")
    for filename in test_cases:
        symbol = extract_symbol_from_filename(filename)
        print(f"  {filename} -> {symbol}")


if __name__ == "__main__":
    print("=== Testing Backtest Engine ===")
    
    # Run setup test first
    print("1. TESTING SETUP:")
    test_backtesting_setup()
    
    print("\n2. TESTING SYMBOL EXTRACTION:")
    test_symbol_extraction()
    
    print("\n" + "="*50)
    
    try:
        results = run_backtest()
        print("\n Batch processing completed!")
        print(f"Total files processed: {results['total_files']}")
        
        if results['results']:
            print("\nProcessed files:")
            for symbol, result in results['results'].items():
                print(f"  {symbol}: Agg={result['data_rows']} rows, Hist={result['historical_rows']} rows - {result['stock_info']['market_cap_type']}")
                print(f"    Trades: {result['backtest_results']['total_trades']}, Exported: {result['backtest_results']['trades_exported']}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()

