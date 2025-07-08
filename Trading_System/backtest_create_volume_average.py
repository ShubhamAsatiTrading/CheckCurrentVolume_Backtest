# backtest_create_volume_average.py - Rolling Window Backtesting for Volume Average Analysis
# Creates AvgData files for each date with rolling window lookback logic

import pandas as pd
import os
import glob
from datetime import datetime, timedelta

class Logger:
    """Minimal logger for backtest operations"""
    
    @staticmethod
    def log(message):
        """Simple console logging"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

class ConfigManager:
    """Config manager for backtest module"""
    
    @staticmethod
    def load():
        """Load configuration from common.txt"""
        defaults = {
            "backtest_start_date": "2023-01-01",
            "avg_volume_days": 30
        }
        
        try:
            config = {}
            if os.path.exists("common.txt"):
                with open("common.txt", 'r') as f:
                    for line in f:
                        if '=' in line and not line.strip().startswith('#'):
                            key, value = line.strip().split('=', 1)
                            if key in ['avg_volume_days']:
                                config[key] = int(value)
                            else:
                                config[key] = value
            
            return {**defaults, **config}
        except Exception as e:
            Logger.log(f"Config error: {e}")
            return defaults

class BacktestCreateVolumeAverage:
    """Backtesting Volume Average Calculator with Rolling Window"""
    
    @staticmethod
    def run_backtest_create_volume_average():
        """Main function to run backtest create volume average calculation"""
        try:
            Logger.log("🚀 Starting Backtest Create Volume Average")
            
            # Load configuration
            config = ConfigManager.load()
            start_date_str = config.get('backtest_start_date', '2023-01-01')
            avg_volume_days = int(config.get('avg_volume_days', 30))
            
            # Parse start date
            try:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            except:
                return False, f"❌ Invalid backtest_start_date format: {start_date_str}"
            
            Logger.log(f"📅 Start: {start_date}, Lookback: {avg_volume_days} days")
            
            # Setup directories
            input_folder = "stocks_historical_data"
            output_folder = "vol_avg_data_backtest"
            
            if not os.path.exists(input_folder):
                return False, f"❌ Input folder not found: {input_folder}"
            
            os.makedirs(output_folder, exist_ok=True)
            
            # Load symbols
            symbols = BacktestCreateVolumeAverage._load_symbols()
            if not symbols:
                return False, "❌ No symbols found in symbols.txt"
            
            Logger.log(f"📊 Processing {len(symbols)} symbols")
            
            # Load all historical data
            all_data = BacktestCreateVolumeAverage._load_all_data(symbols, input_folder)
            if not all_data:
                return False, "❌ No historical data loaded"
            
            # Get all common dates
            common_dates = BacktestCreateVolumeAverage._get_common_dates(all_data, start_date)
            if not common_dates:
                return False, f"❌ No trading dates found from {start_date}"
            
            Logger.log(f"🗓️ Found {len(common_dates)} trading dates to process")
            
            # Process each date with rolling window
            processed_count = 0
            current_date = start_date
            
            for target_date in common_dates:
                if target_date < start_date:
                    continue
                
                # Calculate lookback start date (rolling window)
                lookback_start = BacktestCreateVolumeAverage._get_weekdays_back(target_date, avg_volume_days)
                
                # Process this date
                success = BacktestCreateVolumeAverage._process_date(
                    target_date, lookback_start, symbols, all_data, output_folder
                )
                
                if success:
                    processed_count += 1
                    if processed_count % 50 == 0:  # Progress every 50 files
                        Logger.log(f"📈 Processed {processed_count} dates")
            
            Logger.log(f"✅ Backtest completed: {processed_count} files created")
            return True, f"✅ Backtest completed: {processed_count} files in {output_folder}/"
            
        except Exception as e:
            error_msg = f"❌ Backtest error: {e}"
            Logger.log(error_msg)
            return False, error_msg
    
    @staticmethod
    def _load_symbols():
        """Load symbols from symbols.txt"""
        try:
            if not os.path.exists("symbols.txt"):
                return []
            
            with open("symbols.txt", 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
            return symbols
        except:
            return []
    
    @staticmethod
    def _load_all_data(symbols, input_folder):
        """Load all historical data into memory"""
        all_data = {}
        
        for symbol in symbols:
            file_path = os.path.join(input_folder, f"{symbol}_historical.csv")
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    required_cols = ['date', 'close', 'volume', 'high', 'low']
                    
                    if all(col in df.columns for col in required_cols):
                        df['date'] = pd.to_datetime(df['date'])
                        df['date_only'] = df['date'].dt.date
                        df = df.sort_values('date')
                        all_data[symbol] = df
                except:
                    continue
        
        return all_data
    
    @staticmethod
    def _get_common_dates(all_data, start_date):
        """Get all dates that exist across symbol files"""
        if not all_data:
            return []
        
        # Get all unique dates from all symbols
        all_dates = set()
        for symbol, df in all_data.items():
            all_dates.update(df['date_only'].tolist())
        
        # Filter dates >= start_date and sort
        valid_dates = [d for d in all_dates if d >= start_date]
        return sorted(valid_dates)
    
    @staticmethod
    def _process_date(target_date, lookback_start, symbols, all_data, output_folder):
        """Process single date with rolling window lookback"""
        try:
            results = []
            
            for symbol in symbols:
                if symbol not in all_data:
                    continue
                
                result = BacktestCreateVolumeAverage._calculate_symbol_averages(
                    symbol, target_date, lookback_start, all_data[symbol]
                )
                
                if result:
                    results.append(result)
            
            if not results:
                return False
            
            # Create output file
            df_results = pd.DataFrame(results)
            filename = BacktestCreateVolumeAverage._generate_filename(target_date)
            output_path = os.path.join(output_folder, filename)
            
            df_results.to_csv(output_path, index=False)
            return True
            
        except:
            return False
    
    @staticmethod
    def _calculate_symbol_averages(symbol, target_date, lookback_start, df):
        """Calculate averages for single symbol within date range"""
        try:
            # Filter data for the lookback period
            df_period = df[
                (df['date_only'] >= lookback_start) & 
                (df['date_only'] <= target_date)
            ].copy()
            
            if df_period.empty:
                return None
            
            # Get target date values
            target_data = df_period[df_period['date_only'] == target_date]
            if target_data.empty:
                return None
            
            latest_close = target_data['close'].iloc[-1]
            yest_high = target_data['high'].max()
            yest_low = target_data['low'].min()
            
            # Calculate daily aggregates
            daily_data = df_period.groupby('date_only').agg({
                'close': 'last',
                'volume': 'sum',
                'high': 'max',
                'low': 'min'
            }).reset_index()
            
            # Calculate daily VWAP
            daily_vwaps = []
            for date_val in daily_data['date_only']:
                day_data = df_period[df_period['date_only'] == date_val]
                if not day_data.empty and day_data['volume'].sum() > 0:
                    vwap = (day_data['close'] * day_data['volume']).sum() / day_data['volume'].sum()
                    daily_vwaps.append(vwap)
                else:
                    daily_vwaps.append(0)
            
            daily_data['daily_vwap'] = daily_vwaps
            
            # Calculate averages
            avg_volume = daily_data['volume'].mean()
            avg_vwap = daily_data['daily_vwap'].mean()
            
            return {
                'Symbol': symbol,
                'Yest_close': round(latest_close, 2),
                'Yest_Avg_Vol': round(avg_volume, 2),
                'Yest_Avg_VWAP': round(avg_vwap, 2),
                'yest_high': round(yest_high, 2),
                'yest_low': round(yest_low, 2)
            }
            
        except:
            return None
    
    @staticmethod
    def _get_weekdays_back(end_date, weekdays_count):
        """Calculate start date by going back specified weekdays"""
        current_date = end_date
        weekdays_found = 0
        
        while weekdays_found < weekdays_count:
            current_date = current_date - timedelta(days=1)
            if current_date.weekday() < 5:  # Monday to Friday
                weekdays_found += 1
        
        return current_date
    
    @staticmethod
    def _generate_filename(target_date):
        """Generate filename: AvgData_till_ddmmyyyy.csv"""
        formatted = f"{target_date.day:02d}{target_date.month:02d}{target_date.year}"
        return f"AvgData_till_{formatted}.csv"

# Test function
if __name__ == "__main__":
    print("🧪 Running Backtest Create Volume Average Test...")
    success, message = BacktestCreateVolumeAverage.run_backtest_create_volume_average()
    
    if success:
        print(f"🎉 SUCCESS: {message}")
    else:
        print(f"💥 FAILED: {message}")
    
    print("✅ Test completed!")