# volume_average.py - Volume Average Analysis Module (ENHANCED WITH HIGH/LOW)
# Calculates average volume, VWAP, and yesterday's high/low for stocks

import pandas as pd
import os
import glob
from datetime import datetime, timedelta

class Logger:
    """Simple logger for volume average operations"""
    
    @staticmethod
    def log_to_file_and_console(message, level="INFO"):
        """Log important messages only"""
        try:
            log_file = f"volume_average_logs_{datetime.now().strftime('%Y%m%d')}.log"
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"[{timestamp}] {level}: {message}\n"
            
            # Write to log file only for errors and critical info
            if level in ["ERROR", "WARNING"]:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(log_entry)
                    
        except Exception as e:
            print(f"Failed to write to log file: {e}")
    
    @staticmethod
    def log_progress(current, total, item_name="items"):
        """Log progress"""
        if current % 50 == 0 or current == total:  # Log every 50 items or at end
            percentage = (current / total) * 100 if total > 0 else 0
            print(f"Progress: {current}/{total} {item_name} ({percentage:.1f}%)")

class ConfigManager:
    """Simple config manager for volume average module"""
    
    @staticmethod
    def load():
        """Load configuration from common.txt"""
        defaults = {
            "stop_loss": 4.0, 
            "target": 10.0, 
            "ohlc_value": "open", 
            "trade_today_flag": "no", 
            "check_from_date": "2020-03-28",
            "avg_volume_days": 30
        }
        
        config_file = "common.txt"
        
        try:
            config = {}
            
            if not os.path.exists(config_file):
                return defaults
            
            with open(config_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        # Convert numeric values
                        if key in ['stop_loss', 'target', 'avg_volume_days']:
                            config[key] = float(value) if '.' in value else int(value)
                        else:
                            config[key] = value
            
            final_config = {**defaults, **config}
            return final_config
            
        except Exception as e:
            Logger.log_to_file_and_console(f"Error loading config, using defaults: {e}", "ERROR")
            return defaults

class VolumeAverage:
    """Volume Average and VWAP Calculation Class"""
    
    @staticmethod
    def calculate_average_volume_data():
        """Calculate average volume data for all stocks based on configured duration"""
        try:
            print("🚀 VOLUME AVERAGE CALCULATION STARTED")
            
            # Load configuration
            config = ConfigManager.load()
            duration_days = int(config.get('avg_volume_days', 30))
            print(f"Duration: {duration_days} weekdays")
            
            # Setup directories
            input_folder = "stocks_historical_data"
            output_folder = "Volume_Avg_Data"
            
            if not os.path.exists(input_folder):
                error_msg = f"❌ Input folder '{input_folder}' not found"
                Logger.log_to_file_and_console(error_msg, "ERROR")
                return False, error_msg
            
            # Create output folder if needed
            os.makedirs(output_folder, exist_ok=True)
            
            # Find all historical files
            pattern = os.path.join(input_folder, "*_historical.csv")
            historical_files = glob.glob(pattern)
            
            if not historical_files:
                error_msg = f"❌ No historical files found in {input_folder}"
                Logger.log_to_file_and_console(error_msg, "ERROR")
                return False, error_msg
            
            print(f"Found {len(historical_files)} historical files")
            
            # Process each stock
            results = []
            processed_count = 0
            error_count = 0
            
            for i, file_path in enumerate(historical_files):
                try:
                    # Extract stock name from filename
                    filename = os.path.basename(file_path)
                    stock_symbol = filename.replace("_historical.csv", "")
                    
                    # Show progress
                    Logger.log_progress(i + 1, len(historical_files), "stocks")
                    
                    # Process stock data
                    result = VolumeAverage._process_stock_data(file_path, stock_symbol, duration_days)
                    
                    if result:
                        results.append(result)
                        processed_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    error_count += 1
                    Logger.log_to_file_and_console(f"Error processing {file_path}: {e}", "ERROR")
            
            if not results:
                error_msg = "❌ No valid data processed from any stock"
                Logger.log_to_file_and_console(error_msg, "ERROR")
                return False, error_msg
            
            # Create output DataFrame
            df_results = pd.DataFrame(results)
            
            # Remove helper columns before saving
            columns_to_remove = ['start_date', 'end_date', 'days_processed']
            df_final = df_results.drop(columns=[col for col in columns_to_remove if col in df_results.columns])
            
            # Generate filename
            if results:
                end_date = results[0]['end_date']
                start_date = results[0]['start_date']
                filename = VolumeAverage._generate_filename(end_date)
                output_path = os.path.join(output_folder, filename)
                
                # Remove existing file if it exists
                if os.path.exists(output_path):
                    os.remove(output_path)
                
                # Save results
                df_final.to_csv(output_path, index=False)
                
                # Show summary
                print("✅ VOLUME AVERAGE CALCULATION COMPLETED!")
                print(f"📊 Successfully processed: {processed_count} stocks")
                print(f"❌ Failed to process: {error_count} stocks")
                print(f"📁 Output file: {output_path}")
                print(f"📅 Date range: {start_date} to {end_date} ({duration_days} weekdays)")
                
                return True, f"✅ Volume average data saved: {filename} ({processed_count} stocks processed, {error_count} errors)"
            
        except Exception as e:
            error_msg = f"💥 Critical error in volume average calculation: {e}"
            Logger.log_to_file_and_console(error_msg, "ERROR")
            return False, error_msg
    
    @staticmethod
    def _process_stock_data(file_path, stock_symbol, duration_days):
        """Process individual stock data and calculate averages"""
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Validate required columns - UPDATED to include high/low
            required_cols = ['date', 'close', 'volume', 'high', 'low']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                Logger.log_to_file_and_console(f"Missing columns in {stock_symbol}: {missing_cols}", "WARNING")
                return None
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            df['date_only'] = df['date'].dt.date
            
            # Sort by date to get latest data first
            df = df.sort_values('date', ascending=False)
            
            # Get latest date and values - UPDATED to include high/low
            latest_date = df['date_only'].iloc[0]
            latest_close = df['close'].iloc[0]
            
            # Get yesterday's high and low (from latest trading day)
            latest_day_data = df[df['date_only'] == latest_date]
            yest_high = latest_day_data['high'].max()  # Highest price of latest day
            yest_low = latest_day_data['low'].min()    # Lowest price of latest day
            
            # Calculate weekdays back
            start_date = VolumeAverage._get_weekdays_back(latest_date, duration_days)
            
            # Filter data for the duration
            df_filtered = df[df['date_only'] >= start_date].copy()
            
            if df_filtered.empty:
                Logger.log_to_file_and_console(f"No data available for {stock_symbol} in date range", "WARNING")
                return None
            
            # Group by date and calculate daily metrics
            daily_data = df_filtered.groupby('date_only').agg({
                'close': ['mean', 'last'],  # Average close and last close of day
                'volume': 'sum'  # Total volume for the day
            }).reset_index()
            
            # Flatten column names
            daily_data.columns = ['date', 'avg_close', 'last_close', 'total_volume']
            
            # Calculate Daily VWAP for each day
            daily_vwaps = []
            for date_val in daily_data['date']:
                day_data = df_filtered[df_filtered['date_only'] == date_val]
                if not day_data.empty and day_data['volume'].sum() > 0:
                    # VWAP = sum(close * volume) / sum(volume)
                    vwap = (day_data['close'] * day_data['volume']).sum() / day_data['volume'].sum()
                    daily_vwaps.append(vwap)
                else:
                    daily_vwaps.append(0)
            
            daily_data['daily_vwap'] = daily_vwaps
            
            # Calculate final averages
            avg_close_price = daily_data['last_close'].mean()  # Average of daily closing prices
            avg_volume = daily_data['total_volume'].mean()  # Average of daily volumes
            daily_vwap_average = daily_data['daily_vwap'].mean()  # Average of daily VWAPs
            
            # UPDATED result with high/low data
            result = {
                'Symbol': stock_symbol,
                'Yest_close': round(latest_close, 2),
                'Yest_Avg_Close_Price': round(avg_close_price, 2),
                'Yest_Avg_Vol': round(avg_volume, 2),
                'Yest_Avg_VWAP': round(daily_vwap_average, 2),
                'yest_high': round(float(yest_high), 2),  # NEW: Yesterday's high
                'yest_low': round(float(yest_low), 2),    # NEW: Yesterday's low
                'start_date': start_date,
                'end_date': latest_date,
                'days_processed': len(daily_data)
            }
            
            return result
            
        except Exception as e:
            Logger.log_to_file_and_console(f"Error processing {stock_symbol}: {e}", "ERROR")
            return None
    
    @staticmethod
    def _get_weekdays_back(end_date, weekdays_count):
        """Calculate start date by going back specified weekdays"""
        current_date = end_date
        weekdays_found = 0
        
        while weekdays_found < weekdays_count:
            current_date = current_date - timedelta(days=1)
            # Check if it's a weekday (Monday=0, Sunday=6)
            if current_date.weekday() < 5:  # Monday to Friday
                weekdays_found += 1
        
        return current_date
    
    @staticmethod
    def _generate_filename(end_date):
        """Generate filename in format: AvgData_till_02Jul2025.csv"""
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        end_formatted = f"{end_date.day:02d}{months[end_date.month-1]}{end_date.year}"
        
        return f"AvgData_till_{end_formatted}.csv"

# Test function for standalone execution
if __name__ == "__main__":
    print("🧪 Running Volume Average Calculation Test...")
    success, message = VolumeAverage.calculate_average_volume_data()
    
    if success:
        print(f"🎉 SUCCESS: {message}")
    else:
        print(f"💥 FAILED: {message}")
    
    print("✅ Test completed!")