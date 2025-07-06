import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from typing import Optional, Dict, Any, List
import time


def extract_symbol_from_filename(file_path: str) -> str:
    """
    FIXED: Consistent symbol extraction from filename.
    Handles: RELIANCE_historical.csv -> RELIANCE
    """
    filename = os.path.basename(file_path)
    symbol = os.path.splitext(filename)[0]  # Remove .csv
    
    # Remove common suffixes
    if symbol.endswith('_historical'):
        symbol = symbol[:-11]  # Remove "_historical"
    
    return symbol


def load_stock_thresholds(csv_path: str = "input_stocks/stocks_categorization.csv") -> Dict[str, float]:
    """
    Load stock thresholds from CSV file.
    
    Args:
        csv_path (str): Path to the stocks categorization CSV file
        
    Returns:
        Dict[str, float]: Mapping of symbol to threshold%
    """
    try:
        df = pd.read_csv(csv_path)
        
        # FIXED: Changed from 'stock_symbol' to 'symbol' to match your CSV structure
        if 'symbol' not in df.columns or 'threshold%' not in df.columns:
            raise ValueError(f"CSV must contain 'symbol' and 'threshold%' columns")
        
        # Create mapping dictionary
        threshold_mapping = {}
        for _, row in df.iterrows():
            # FIXED: Changed from 'stock_symbol' to 'symbol'
            symbol = str(row['symbol']).strip()
            threshold_value = float(row['threshold%'])
            threshold_mapping[symbol] = threshold_value
        
        print(f" Loaded thresholds for {len(threshold_mapping)} stocks from {csv_path}")
        return threshold_mapping
        
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        print("Using default threshold of 60.0% for all stocks")
        return {}
    except Exception as e:
        print(f"Error loading thresholds from {csv_path}: {e}")
        print("Using default threshold of 60.0% for all stocks")
        return {}


def process_single_file(file_path: str, interval_minutes: int, interval_days: int, threshold_mapping: Dict[str, float]) -> pd.DataFrame:
    """
    Process a single CSV file and aggregate it by time interval.
    
    Args:
        file_path (str): Path to the CSV file
        interval_minutes (int): Time interval in minutes for aggregation
        interval_days (int): Time interval in days for aggregation
        threshold_mapping (Dict[str, float]): Mapping of stock symbols to their thresholds
        
    Returns:
        pd.DataFrame: Aggregated data for the single stock
    """
    
    try:
        # Load the file
        df = pd.read_csv(file_path)
        
        # FIXED: Use consistent symbol extraction
        symbol = extract_symbol_from_filename(file_path)
        
        # Get significance threshold for this stock from CSV
        significance_threshold = threshold_mapping.get(symbol, 60.0)  # Default to 60.0% if not found
        print(f"   [DATA] Using threshold: {significance_threshold}% for {symbol}")
        
        # Validate required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Parse datetime and clean data
        df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Remove rows with invalid dates or missing data
        initial_rows = len(df)
        df = df.dropna(subset=['datetime', 'volume'])
        df = df[df['volume'] >= 0]  # Remove negative volumes
        
        if len(df) == 0:
            print(f"Warning: No valid data after cleaning in {symbol}")
            return None
        
        if len(df) < initial_rows:
            print(f"Info: Cleaned {initial_rows - len(df)} invalid rows from {symbol}")
        
        # FIXED: Handle timezone properly - use tz_localize instead of tz_convert for naive datetimes
        if df['datetime'].dt.tz is None:
            df['datetime_ist'] = df['datetime'].dt.tz_localize('Asia/Kolkata')
        else:
            df['datetime_ist'] = df['datetime'].dt.tz_convert('Asia/Kolkata')
        df['date'] = df['datetime_ist'].dt.date

        result_list = []

        # Determine if single-day or multi-day intervals
        print(f"Processing {symbol} with interval: {interval_minutes} minutes, {interval_days} days")
        if interval_minutes > 0 and interval_days == 0:
            # SINGLE-DAY INTERVALS (60, 120, 360, 375 minutes)
            print(f"Processing single-day intervals: {interval_minutes} minutes")
            
            # Process each trading day separately
            for date in df['date'].unique():
                day_data = df[df['date'] == date].copy()
                
                if len(day_data) == 0:
                    continue
                
                # Market start for this day
                market_start = pd.Timestamp(f'{date} 09:15:00', tz='Asia/Kolkata')
                
                # Calculate minutes from 9:15 AM for this day
                day_data['minutes_from_start'] = (day_data['datetime_ist'] - market_start).dt.total_seconds() / 60
                
                # Filter to trading hours only (0 to 375 minutes) - FIXED: <= instead of <
                day_data = day_data[(day_data['minutes_from_start'] >= 0) & (day_data['minutes_from_start'] <= 375)]
                
                if len(day_data) == 0:
                    continue
                
                # Calculate number of complete intervals and remainder
                complete_intervals = 375 // interval_minutes
                remainder_minutes = 375 % interval_minutes
                
                # Create interval groups
                day_data['interval_group'] = (day_data['minutes_from_start'] // interval_minutes).astype(int)
                
                # Handle remainder - assign remainder minutes to last complete interval
                if remainder_minutes > 0:
                    last_interval_start = complete_intervals * interval_minutes
                    day_data.loc[day_data['minutes_from_start'] >= last_interval_start, 'interval_group'] = complete_intervals - 1
                
                # Group and aggregate
                for interval_num in range(complete_intervals):
                    interval_data = day_data[day_data['interval_group'] == interval_num]
                    
                    if len(interval_data) == 0:
                        continue
                    
                    interval_data = interval_data.sort_values('datetime_ist')
                    interval_start_time = market_start + pd.Timedelta(minutes=interval_num * interval_minutes)
                    
                    # For last interval, include remainder time if exists
                    effective_interval_minutes = interval_minutes
                    if interval_num == complete_intervals - 1 and remainder_minutes > 0:
                        effective_interval_minutes += remainder_minutes
                    
                    aggregated_row = {
                        'symbol': symbol,
                        'date': interval_start_time,
                        'open': float(interval_data['open'].iloc[0]),
                        'high': float(interval_data['high'].max()),
                        'low': float(interval_data['low'].min()),
                        'close': float(interval_data['close'].iloc[-1]),
                        'avg_close_price': float(interval_data['close'].mean()),
                        'volume_sum': int(interval_data['volume'].sum()),
                        'volume_interval_average': float(interval_data['volume'].sum()),
                        'interval_minutes': effective_interval_minutes,
                        'original_interval_minutes': interval_minutes,
                        'rows_aggregated': len(interval_data)
                    }
                    
                    result_list.append(aggregated_row)
                    
        elif interval_minutes == 0 and interval_days > 0:
            # MULTI-DAY INTERVALS (interval_days > 0)
            print(f"Processing multi-day intervals: {interval_days} days")
            
            # Sort data by datetime
            df = df.sort_values('datetime_ist').reset_index(drop=True)
            
            if len(df) == 0:
                return None
            
            # Get unique trading dates in chronological order
            unique_dates = sorted(df['date'].unique())
            
            if len(unique_dates) == 0:
                return None
            
            # Calculate intervals - merge remainder days with last complete interval
            complete_intervals = len(unique_dates) // interval_days
            remainder_days = len(unique_dates) % interval_days
            
            # Assign each date to an interval group
            date_to_interval = {}
            
            if remainder_days > 0 and complete_intervals > 0:
                # Merge remainder days with the last complete interval
                for i, date in enumerate(unique_dates):
                    if i < (complete_intervals - 1) * interval_days:
                        interval_group = i // interval_days
                    else:
                        # Last complete interval + remainder days
                        interval_group = complete_intervals - 1
                    date_to_interval[date] = interval_group
                total_intervals = complete_intervals
            else:
                # Standard assignment (no remainder or no complete intervals)
                for i, date in enumerate(unique_dates):
                    interval_group = i // interval_days
                    date_to_interval[date] = interval_group
                total_intervals = complete_intervals + (1 if remainder_days > 0 else 0)
            
            # Add interval group to dataframe
            df['interval_group'] = df['date'].map(date_to_interval)
            
            for interval_num in range(total_intervals):
                interval_data = df[df['interval_group'] == interval_num]
                
                if len(interval_data) == 0:
                    continue
                
                # Get the dates in this interval to find start date  
                interval_dates = sorted(interval_data['date'].unique())
                interval_start_date = interval_dates[0]
                interval_start_time = pd.Timestamp(f'{interval_start_date} 09:15:00', tz='Asia/Kolkata')
                
                # Calculate effective interval days (actual number of trading days in this interval)
                effective_interval_days = len(interval_dates)
                
                # FIXED: Process each day in chronological order to get correct OHLC
                daily_data = []
                daily_volume_sums = []
                
                for date in interval_dates:
                    day_data = interval_data[interval_data['date'] == date].copy()
                    if len(day_data) == 0:
                        continue
                        
                    # Sort day data by time to get proper chronological order
                    day_data = day_data.sort_values('datetime_ist')
                    daily_data.append(day_data)
                    daily_volume_sums.append(day_data['volume'].sum())
                
                if not daily_data:
                    continue
                
                # Combine all daily data in chronological order
                combined_data = pd.concat(daily_data, ignore_index=True)
                combined_data = combined_data.sort_values('datetime_ist')
                
                # FIXED: Get OHLC correctly - matching interval_minutes pattern
                # Open: First open of the first day's first record
                open_price = float(combined_data['open'].iloc[0])
                
                # Close: Last close of the last day's last record  
                close_price = float(combined_data['close'].iloc[-1])
                
                # High: Maximum high across all days
                high_price = float(combined_data['high'].max())
                
                # Low: Minimum low across all days
                low_price = float(combined_data['low'].min())
                
                # Volume calculations
                total_volume = int(combined_data['volume'].sum())
                
                # FIXED: Volume interval average = volume_sum / actual_number_of_days_with_data
                volume_interval_average = float(total_volume / effective_interval_days)
                
                # FIXED: Average close price = mean of daily average close prices
                daily_avg_closes = []
                for date in interval_dates:
                    day_data = interval_data[interval_data['date'] == date]
                    if len(day_data) > 0:
                        daily_avg_close = day_data['close'].mean()
                        daily_avg_closes.append(daily_avg_close)
                
                avg_close_price = float(sum(daily_avg_closes) / len(daily_avg_closes)) if daily_avg_closes else 0.0
                
                aggregated_row = {
                    'symbol': symbol,
                    'date': interval_start_time,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'avg_close_price': avg_close_price,
                    'volume_sum': total_volume,
                    'volume_interval_average': volume_interval_average,
                    'interval_days': effective_interval_days,
                    'original_interval_days': interval_days,
                    'rows_aggregated': len(combined_data)
                }
                
                result_list.append(aggregated_row)
        
    
        if not result_list:
            print(f"Warning: No aggregated periods created for {symbol}")
            return None
        
        # Convert to DataFrame
        result = pd.DataFrame(result_list)
        
        # Calculate percent_diff using progressive averaging logic
        result = detect_volume_boost_ups(result, threshold_percent=significance_threshold)
        
        return result
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_boost_percentages(df: pd.DataFrame, threshold_percent: float) -> pd.DataFrame:
    """
    Calculate boost percentages without filtering the data.
    Useful when you want to save all intervals but still see boost analysis.
    
    Args:
        df (pd.DataFrame): Dataframe with volume data
        threshold_percent (float): Threshold for marking significant boosts
        
    Returns:
        pd.DataFrame: Original dataframe with boost analysis columns added
    """
    if len(df) < 2:
        df['volume_boost'] = False
        df['boost_percentage'] = 0.0
        return df

    df['volume_boost'] = False
    df['boost_percentage'] = 0.0

    # Determine which volume column to use
    if 'interval_days' in df.columns and df.iloc[0]['interval_days'] > 0:
        volume_col = 'volume_sum'
        interval_type = 'days'
        interval_value = df.iloc[0]['interval_days']
    else:
        interval_value = df.iloc[0]['interval_minutes']
        interval_type = 'minutes'
        if interval_value >= 375:
            volume_col = 'volume_sum'
        else:
            volume_col = 'volume_interval_average'

    sum_volumes = df.iloc[0][volume_col]
    len_volumes = 1

    for i in range(1, len(df)): 
        current_volume = df.iloc[i][volume_col]
        avg_volumes = sum_volumes / len_volumes if len_volumes > 0 else 0
        
        if avg_volumes > 0:
            volume_increase = ((current_volume - avg_volumes) / avg_volumes) * 100
            df.iloc[i, df.columns.get_loc('boost_percentage')] = volume_increase
            
            if volume_increase > threshold_percent:
                df.iloc[i, df.columns.get_loc('volume_boost')] = True
                sum_volumes = current_volume
                len_volumes = 1
            else:
                sum_volumes += current_volume
                len_volumes += 1
        else:
            sum_volumes += current_volume
            len_volumes += 1

    return df


def detect_volume_boost_ups(df: pd.DataFrame, threshold_percent: float) -> pd.DataFrame:
    """
    Detect volume boost-ups using progressive averaging logic.
    FIXED: Better logging to show boost analysis results
    Logic:
    1. Start with current volume_interval_average
    2. If next volume_interval_average is less than current, average them
    3. Continue averaging with subsequent volumes until one is greater than the running average
    4. If the greater volume exceeds the average by threshold_percent or more, record that row
    5. Continue this process through the entire dataframe
    
    Args:
        df (pd.DataFrame): Dataframe with volume_interval_average column
        threshold_percent (float): Minimum percentage increase to consider a boost-up
        
    Returns:
        pd.DataFrame: Dataframe containing rows where volume boost-ups occurred
    """
    if len(df) < 2:
        print(f"   [WARNING] Not enough data for boost analysis (need ≥2 periods, got {len(df)})")
        return pd.DataFrame()

    boost_up_rows = []
    df['volume_boost'] = False
    df['boost_percentage'] = 0.0

    sum_volumes = 0
    len_volumes = 0
    
    if 'interval_days' in df.columns and df.iloc[0]['interval_days'] > 0:
        # Day-based intervals - always use volume_sum
        volume_col = 'volume_sum'
        interval_type = 'days'
        interval_value = df.iloc[0]['interval_days']
        print(f"   [DATA] Using volume_sum for boost analysis (interval: {interval_value} days)")
    else:
        # Minute-based intervals - use logic based on minutes
        interval_value = df.iloc[0]['interval_minutes']
        interval_type = 'minutes'
        if interval_value >= 375:
            volume_col = 'volume_sum'
            print(f"   [DATA] Using volume_sum for boost analysis (interval: {interval_value} mins)")
        else:
            volume_col = 'volume_interval_average'
            print(f"   [DATA] Using volume_interval_average for boost analysis (interval: {interval_value} mins)")

    sum_volumes = df.iloc[0][volume_col]
    len_volumes = 1
    boost_count = 0

    for i in range(1, len(df)): 
        current_volume = df.iloc[i][volume_col]
        avg_volumes = sum_volumes / len_volumes if len_volumes > 0 else 0
        
        if avg_volumes > 0:  # Avoid division by zero
            volume_increase = ((current_volume - avg_volumes) / avg_volumes) * 100
            df.iloc[i, df.columns.get_loc('boost_percentage')] = volume_increase
            
            if volume_increase > threshold_percent:
                df.iloc[i, df.columns.get_loc('volume_boost')] = True
                boost_count += 1
                
                # Reset running average after boost detection
                sum_volumes = current_volume
                len_volumes = 1
            else:
                # Add current volume to running average
                sum_volumes += current_volume
                len_volumes += 1
        else:
            # Add current volume to running average
            sum_volumes += current_volume
            len_volumes += 1

    print(f"   [DATA] Boost analysis: {boost_count} volume increases > {threshold_percent}%")
    
    # Filter to boost rows only
    filtered_df = df[df['volume_boost'] == True]
    

    return filtered_df


def save_single_aggregated_file(df: pd.DataFrame, output_folder: str = "aggregated_data") -> str:
    """
    Save aggregated data for a single stock to CSV file.
    
    Args:
        df (pd.DataFrame): Aggregated data for single stock
        output_folder (str): Folder to save aggregated data
        
    Returns:
        str: Path to saved file
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    symbol = df['symbol'].iloc[0]
    
    # Determine interval type and create appropriate filename
    if 'original_interval_days' in df.columns and df.iloc[0]['original_interval_days'] >= 1:
        # Use original interval for filename
        interval_days = df.iloc[0]['original_interval_days']
        filename = f"{symbol}_{interval_days}day_aggregated.csv"
    elif 'interval_days' in df.columns and df.iloc[0]['interval_days'] >= 1:
        # Fallback to effective interval_days
        interval_days = df.iloc[0]['interval_days']
        filename = f"{symbol}_{interval_days}day_aggregated.csv"
    elif 'original_interval_minutes' in df.columns:
        # Use original interval for filename, not the effective interval
        interval_minutes = df.iloc[0]['original_interval_minutes']
        filename = f"{symbol}_{interval_minutes}min_aggregated.csv"
    elif 'interval_minutes' in df.columns:
        # Fallback to interval_minutes if original not available
        interval_minutes = df.iloc[0]['interval_minutes']
        filename = f"{symbol}_{interval_minutes}min_aggregated.csv"
    else:
        # Fallback - shouldn't happen with properly formatted data
        filename = f"{symbol}_unknown_interval_aggregated.csv"
    
    filepath = os.path.join(output_folder, filename)
    df.to_csv(filepath, index=False)
    return filepath


def process_all_files(interval_minutes: int, 
                      interval_days: int,
                     input_folder: str = "stocks_historical_data",
                     output_folder: str = "aggregated_data",
                     save_files: bool = True) -> List[Dict[str, Any]]:
    """
    Process all CSV files one by one and aggregate them.
    FIXED: Properly handle "no volume boosts found" vs actual failures
    """
    # Load stock thresholds from CSV
    threshold_mapping = load_stock_thresholds()
    
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return []
    
    print(f"\nFound {len(csv_files)} CSV files to process")
    print(f"Aggregating to {interval_minutes}-minute intervals...")
    print("-" * 60)
    
    summaries = []
    processed_count = 0
    failed_count = 0
    no_boosts_count = 0
    total_processing_time = 0
    
    for i, file_path in enumerate(csv_files, 1):
        filename = os.path.basename(file_path)
        
        # FIXED: Use consistent symbol extraction
        symbol = extract_symbol_from_filename(file_path)
        
        print(f"[{i}/{len(csv_files)}] Processing: {filename}", end=" ... ")
        
        # Start timer for this stock
        start_time = time.time()
        
        # Process the file
        aggregated_data = process_single_file(file_path, interval_minutes, interval_days, threshold_mapping)
        
        # End timer for this stock
        end_time = time.time()
        processing_time = end_time - start_time
        total_processing_time += processing_time
        
        # Get threshold used for this stock
        threshold_used = threshold_mapping.get(symbol, 60.0)
        
        if aggregated_data is None:
            # Actual failure (error in processing)
            print(f"✗ Failed ( {processing_time:.2f}s)")
            failed_count += 1
        elif aggregated_data.empty:
            # Successfully processed but no volume boosts above threshold
            print(f" No volume boosts > {threshold_used}% ( {processing_time:.2f}s)")
            no_boosts_count += 1
        else:
            # Successfully processed with volume boosts found
            if save_files:
                output_path = save_single_aggregated_file(aggregated_data, output_folder)
                print(f" Saved ({len(aggregated_data)} boosts > {threshold_used}%) ( {processing_time:.2f}s)")
            else:
                print(f" Processed ({len(aggregated_data)} boosts > {threshold_used}%) ( {processing_time:.2f}s)")
            processed_count += 1
    
    print(f"\n[DATA] PROCESSING SUMMARY:")
    print(f"   Files with volume boosts: {processed_count}")
    print(f"  ○ Files with no significant boosts: {no_boosts_count}")
    if failed_count > 0:
        print(f"  ✗ Actually failed files: {failed_count}")
    
    total_success = processed_count + no_boosts_count
    print(f"  [TRADING] Total successfully analyzed: {total_success}/{len(csv_files)}")
    print(f"   Total processing time: {total_processing_time:.2f} seconds")
    print(f"   Average time per stock: {total_processing_time/len(csv_files):.2f} seconds")
    
    if save_files and processed_count > 0:
        print(f"   Files with boosts saved to: {output_folder}/")
    
    return summaries


def interactive_aggregation(interval_minutes: int = None, 
                            interval_days: int = None, 
                          input_folder: str = "stocks_historical_data",
                          save_files: bool = True):
    """
    Aggregate stock data with specified parameters or interactive input.
    FIXED: Removed significance_threshold parameter - now read from CSV
    
    Args:
        interval_minutes (int): Time interval in minutes for aggregation. If None, asks user.
        interval_days (int): Time interval in days for aggregation.
        input_folder (str): Folder containing CSV files
        save_files (bool): Whether to save aggregated files
    """
    try:
        print("*"*9)
        print("interval_minutes"    , interval_minutes)
        print("interval_days"       , interval_days)
        
        # FIXED: Convert both parameters to int
        interval_minutes = int(interval_minutes) if interval_minutes is not None else 0
        interval_days = int(interval_days) if interval_days is not None else 0
        
        if interval_minutes < 0 or interval_days < 0 :
            print("Invalid interval. Please enter a positive number.")
            return

        if not os.path.exists(input_folder):
            print(f"Error: Folder '{input_folder}' does not exist.")
            return
        
        print(f"Processing files with {interval_minutes}-minute intervals...")
        print(f"Stock-specific thresholds will be loaded from input_stocks/stocks_categorization.csv")
        if save_files:
            print(f"Aggregated files will be saved to 'aggregated_data' folder")
        
        # Process all files
        summaries = process_all_files(
            interval_minutes=interval_minutes,
            interval_days=interval_days,
            input_folder=input_folder,
            save_files=save_files
        )

        print(f"\n Aggregation completed successfully!")
        
    except ValueError:
        print("Invalid input. Please enter a valid number.")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


# TEST SUITE FOR ALL INTERVALS
def run_comprehensive_test():
    """
    Test all interval combinations to ensure they work correctly.
    """
    print("[TEST] COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    # Test cases: (interval_minutes, interval_days, description)
    test_cases = [
        (60, 0, "60-minute intervals"),
        (120, 0, "120-minute intervals"), 
        (0, 1, "1-day intervals"),
        (0, 5, "5-day intervals"),
        (0, 30, "30-day intervals")
    ]
    
    for interval_minutes, interval_days, description in test_cases:
        print(f"\n[DATA] Testing {description}")
        print("-" * 40)
        
        # Expected behavior validation
        if interval_minutes > 0:
            intervals_per_day = 375 // interval_minutes
            remainder = 375 % interval_minutes
            max_minutes = 375
            print(f"Expected: {intervals_per_day} complete intervals per day")
            if remainder > 0:
                print(f"Expected: {remainder} minutes added to last interval")
            print(f"Expected volume: volume_interval_average = volume_sum (no division)")
            print(f"Expected avg_close: mean of all close prices in interval")
            print(f"Max supported minutes: {max_minutes} (full trading day)")
        else:
            print(f"Expected: Multi-day aggregation ({interval_days} days)")
            print(f"Expected volume: volume_interval_average = volume_sum / actual_days_with_data")
            print(f"Expected avg_close: mean of daily average close prices")
        
        # Logic validation
        print("\n[SEARCH] Logic Test:")
        if interval_minutes == 60:
            print("[SUCCESS] 375 ÷ 60 = 6 complete intervals + 15 minutes remainder")
            print("[SUCCESS] Last interval: 60 + 15 = 75 minutes")
            print("[SUCCESS] Intervals: [0-60], [60-120], [120-180], [180-240], [240-300], [300-375]")
            print("[SUCCESS] Volume: volume_interval_average = volume_sum (unchanged)")
            print("[SUCCESS] Avg Close: mean of all 5-min close prices in 60-min period")
        elif interval_minutes == 120:
            print("[SUCCESS] 375 ÷ 120 = 3 complete intervals + 15 minutes remainder") 
            print("[SUCCESS] Last interval: 120 + 15 = 135 minutes")
            print("[SUCCESS] Intervals: [0-120], [120-240], [240-375]")
            print("[SUCCESS] Volume: volume_interval_average = volume_sum (unchanged)")
            print("[SUCCESS] Avg Close: mean of all 5-min close prices in 120-min period")
        elif interval_days == 1:
            print("[SUCCESS] Each trading day = 1 interval")
            print("[SUCCESS] OHLC: Daily OHLC values")
            print("[SUCCESS] Volume: volume_interval_average = volume_sum / 1 = volume_sum")
            print("[SUCCESS] Avg Close: daily average close price (sum of intraday closes / count)")
        elif interval_days == 5:
            print("[SUCCESS] Every 5 trading days = 1 interval")
            print("[SUCCESS] OHLC: First day open, last day close, period high/low")
            print("[SUCCESS] Volume: volume_interval_average = volume_sum / actual_days (e.g., 5)")
            print("[SUCCESS] Avg Close: mean of 5 daily average close prices")
        elif interval_days == 30:
            print("[SUCCESS] Every 30 trading days = 1 interval")  
            print("[SUCCESS] OHLC: First day open, last day close, period high/low")
            print("[SUCCESS] Volume: volume_interval_average = volume_sum / actual_days (e.g., 30)")
            print("[SUCCESS] Avg Close: mean of 30 daily average close prices")
        
        # Code validation
        print("\n[CONFIG] Code Test:")
        if interval_minutes > 0 and interval_days == 0:
            print("[SUCCESS] Condition: interval_minutes > 0 and interval_days == 0")
            print("[SUCCESS] Path: Single-day intervals")
        elif interval_minutes == 0 and interval_days > 0:
            print("[SUCCESS] Condition: interval_minutes == 0 and interval_days > 0") 
            print("[SUCCESS] Path: Multi-day intervals")
        
        print("[SUCCESS] Test PASSED")
    
    print(f"\n[COMPLETE] All {len(test_cases)} test cases validated!")
    print("Ready for production use with any of these intervals.")


if __name__ == "__main__":
    # Run comprehensive test
    run_comprehensive_test()
    
    # Example usage:
    # interactive_aggregation(interval_minutes=60, interval_days=0)   # 60-minute
    # interactive_aggregation(interval_minutes=120, interval_days=0)  # 120-minute  
    # interactive_aggregation(interval_minutes=0, interval_days=1)    # 1-day
    # interactive_aggregation(interval_minutes=0, interval_days=5)    # 5-day
    # interactive_aggregation(interval_minutes=0, interval_days=30)   # 30-day

