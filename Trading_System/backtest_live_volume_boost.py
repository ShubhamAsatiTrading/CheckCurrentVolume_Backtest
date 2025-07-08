# backtest_live_volume_boost.py - Live Volume Boost Backtesting Strategy
# Tests volume boost strategy with multiple time windows and trailing stops

import pandas as pd
import os
import glob
from datetime import datetime, timedelta, time
import math

class Logger:
    """Minimal logger"""
    
    @staticmethod
    def log(message):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

class ConfigManager:
    """Configuration manager for backtest parameters"""
    
    @staticmethod
    def load():
        """Load backtest configuration from common.txt"""
        defaults = {
            "backtest_max_price_rise": 2.0,
            "backtest_profit_target": 2.0,
            "backtest_stop_loss": 2.0,
            "backtest_volume_915": 50.0,
            "backtest_volume_930": 80.0,
            "backtest_volume_1000": 100.0,
            "backtest_volume_1030": 120.0,
            "backtest_volume_1100": 140.0,
            "backtest_volume_1130": 160.0,
            "backtest_volume_1200": 180.0
        }
        
        try:
            config = {}
            if os.path.exists("common.txt"):
                with open("common.txt", 'r') as f:
                    for line in f:
                        if '=' in line and not line.strip().startswith('#'):
                            key, value = line.strip().split('=', 1)
                            if key.startswith('backtest_'):
                                try:
                                    config[key] = float(value)
                                except:
                                    config[key] = value
            
            return {**defaults, **config}
        except Exception as e:
            Logger.log(f"Config error: {e}")
            return defaults

class BacktestLiveVolumeBoost:
    """Live Volume Boost Backtesting Engine"""
    
    @staticmethod
    def run_backtest_live_volume_boost():
        """Main function to run live volume boost backtesting"""
        try:
            Logger.log("🚀 Starting BUY+SELL Live Volume Boost Backtest")
            
            # Load configuration
            config = ConfigManager.load()
            
            # Setup directories
            avg_data_folder = "vol_avg_data_backtest"
            historical_folder = "stocks_historical_data"
            output_folder = "backtest_result_live_volume_boost"
            
            if not os.path.exists(avg_data_folder):
                return False, f"❌ Folder not found: {avg_data_folder}"
            
            os.makedirs(output_folder, exist_ok=True)
            
            # Get all AvgData files
            avg_files = glob.glob(os.path.join(avg_data_folder, "AvgData_till_*.csv"))
            if not avg_files:
                return False, "❌ No AvgData files found"
            
            avg_files.sort()
            Logger.log(f"📊 Processing {len(avg_files)} AvgData files")
            
            # Process each file
            total_trades = 0
            processed_days = 0
            
            for avg_file in avg_files:
                trades = BacktestLiveVolumeBoost._process_single_day(
                    avg_file, historical_folder, output_folder, config
                )
                
                if trades is not None:
                    total_trades += len(trades)
                    processed_days += 1
                    
                    if processed_days % 10 == 0:
                        Logger.log(f"📈 Processed {processed_days} days, {total_trades} total trades")
            
            Logger.log(f"✅ Backtest completed: {processed_days} days, {total_trades} trades")
            return True, f"✅ Backtest completed: {total_trades} trades in {output_folder}/"
            
        except Exception as e:
            error_msg = f"❌ Backtest error: {e}"
            Logger.log(error_msg)
            return False, error_msg
    
    @staticmethod
    def _process_single_day(avg_file, historical_folder, output_folder, config):
        """Process single day backtesting"""
        try:
            # Extract date from filename
            filename = os.path.basename(avg_file)
            date_part = filename.replace("AvgData_till_", "").replace(".csv", "")
            
            # Parse date (ddmmyyyy format)
            try:
                day = int(date_part[:2])
                month = int(date_part[2:4])
                year = int(date_part[4:8])
                target_date = datetime(year, month, day).date()
            except:
                return None
            
            # Get next trading day
            next_date = target_date + timedelta(days=1)
            
            # Load AvgData
            df_avg = pd.read_csv(avg_file)
            if df_avg.empty:
                return []
            
            # Process each stock
            all_trades = []
            
            for _, row in df_avg.iterrows():
                symbol = row['Symbol']
                trade_results = BacktestLiveVolumeBoost._test_stock_strategy(
                    symbol, row, next_date, historical_folder, config
                )
                
                if trade_results:
                    # trade_results is a list of trades (2 per strategy)
                    all_trades.extend(trade_results)
            
            # Save results
            if all_trades:
                results_df = pd.DataFrame(all_trades)
                output_file = os.path.join(output_folder, f"backtest_results_{date_part}.csv")
                results_df.to_csv(output_file, index=False)
            
            return all_trades
            
        except Exception as e:
            return None
    
    @staticmethod
    def _test_stock_strategy(symbol, avg_row, target_date, historical_folder, config):
        """Test strategy for single stock"""
        try:
            # Load historical data
            hist_file = os.path.join(historical_folder, f"{symbol}_historical.csv")
            if not os.path.exists(hist_file):
                return None
            
            df_hist = pd.read_csv(hist_file)
            df_hist['date'] = pd.to_datetime(df_hist['date'])
            df_hist['date_only'] = df_hist['date'].dt.date
            df_hist['time_only'] = df_hist['date'].dt.time
            
            # Filter for target date
            day_data = df_hist[df_hist['date_only'] == target_date].copy()
            if day_data.empty:
                return None
            
            day_data = day_data.sort_values('date')
            
            # Get reference values
            yest_high = float(avg_row['yest_high'])
            yest_close = float(avg_row['Yest_close'])
            yest_avg_vol = float(avg_row['Yest_Avg_Vol'])
            
            # Define time windows (no entry after 2PM)
            time_windows = [
                (time(9, 15), time(9, 30), config.get('backtest_volume_915', 50.0)),
                (time(9, 30), time(10, 0), config.get('backtest_volume_930', 80.0)),
                (time(10, 0), time(10, 30), config.get('backtest_volume_1000', 100.0)),
                (time(10, 30), time(11, 0), config.get('backtest_volume_1030', 120.0)),
                (time(11, 0), time(11, 30), config.get('backtest_volume_1100', 140.0)),
                (time(11, 30), time(12, 0), config.get('backtest_volume_1130', 160.0)),
                (time(12, 0), time(14, 0), config.get('backtest_volume_1200', 180.0))
            ]
            
            # Check each time window for entry
            for start_time, end_time, volume_threshold in time_windows:
                
                window_data = day_data[
                    (day_data['time_only'] >= start_time) & 
                    (day_data['time_only'] <= end_time)
                ].copy()
                
                if window_data.empty:
                    continue
                
                # Check minute by minute in this window
                for _, minute_row in window_data.iterrows():
                    current_time = minute_row['time_only']
                    current_price = float(minute_row['close'])
                    
                    # Calculate cumulative volume from market open
                    market_open_data = day_data[day_data['time_only'] <= current_time]
                    cumulative_volume = market_open_data['volume'].sum()
                    
                    # Check entry conditions - BUY or SELL (whichever triggers first)
                    volume_pct = (cumulative_volume / yest_avg_vol) * 100 if yest_avg_vol > 0 else 0
                    
                    # BUY condition: price > yest_high
                    if current_price > yest_high and volume_pct >= volume_threshold:
                        trade_results = BacktestLiveVolumeBoost._execute_dual_strategy_trade(
                            symbol, minute_row, day_data, yest_close, config, 
                            volume_pct, f"{start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}",
                            yest_avg_vol, yest_high, cumulative_volume, "BUY"
                        )
                        return trade_results
                    
                    # SELL condition: price < yest_low  
                    elif current_price < float(avg_row['yest_low']) and volume_pct >= volume_threshold:
                        trade_results = BacktestLiveVolumeBoost._execute_dual_strategy_trade(
                            symbol, minute_row, day_data, yest_close, config, 
                            volume_pct, f"{start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}",
                            yest_avg_vol, float(avg_row['yest_low']), cumulative_volume, "SELL"
                        )
                        return trade_results
            
            return None
            
        except Exception as e:
            return None
    
    @staticmethod
    def _execute_dual_strategy_trade(symbol, entry_row, day_data, yest_close, config, entry_volume_pct, entry_window, yest_volume, trigger_price, today_volume, trade_type):
        """Execute trade with dual trailing strategies (Lock_Profit and Maintain_Risk)"""
        try:
            # Run both trailing strategies
            trade_lock = BacktestLiveVolumeBoost._simulate_single_trade(
                symbol, entry_row, day_data, yest_close, config, entry_volume_pct, 
                entry_window, yest_volume, trigger_price, today_volume, trade_type, "Lock_Profit"
            )
            
            trade_maintain = BacktestLiveVolumeBoost._simulate_single_trade(
                symbol, entry_row, day_data, yest_close, config, entry_volume_pct, 
                entry_window, yest_volume, trigger_price, today_volume, trade_type, "Maintain_Risk"
            )
            
            return [trade_lock, trade_maintain]
            
        except Exception as e:
            return []
    
    @staticmethod
    def _simulate_single_trade(symbol, entry_row, day_data, yest_close, config, entry_volume_pct, entry_window, yest_volume, trigger_price, today_volume, trade_type, trailing_strategy):
        """Simulate single trade with specific trailing strategy"""
        try:
            entry_time = entry_row['date']
            entry_price = float(entry_row['close'])
            
            # Calculate targets based on trade type
            profit_target_pct = config.get('backtest_profit_target', 2.0) / 100
            stop_loss_pct = config.get('backtest_stop_loss', 2.0) / 100
            
            if trade_type == "BUY":
                profit_target = entry_price * (1 + profit_target_pct)
                initial_stop = max(entry_price * (1 - stop_loss_pct), yest_close)
            else:  # SELL
                profit_target = entry_price * (1 - profit_target_pct)
                initial_stop = entry_price * (1 + stop_loss_pct)
            
            # Get remaining day data
            remaining_data = day_data[
                (day_data['date'] > entry_time) & 
                (day_data['time_only'] <= time(15, 15))
            ].copy()
            
            if remaining_data.empty:
                return BacktestLiveVolumeBoost._create_trade_record(
                    symbol, entry_price, entry_price, entry_time, entry_time,
                    0.0, initial_stop, entry_price, 0.0, "No", 0, entry_window, 
                    entry_volume_pct, yest_volume, trigger_price, today_volume, trade_type, trailing_strategy
                )
            
            # Track trade progress
            current_stop = initial_stop
            if trade_type == "BUY":
                max_favorable_price = entry_price
            else:  # SELL
                max_favorable_price = entry_price
            
            max_profit_pct = 0.0
            
            for _, row in remaining_data.iterrows():
                current_price = float(row['close'])
                current_time = row['date']
                
                # Update max favorable price and trailing stop
                if trade_type == "BUY":
                    if current_price > max_favorable_price:
                        max_favorable_price = current_price
                        max_profit_pct = ((max_favorable_price - entry_price) / entry_price) * 100
                        
                        # Update trailing stop based on strategy
                        if trailing_strategy == "Lock_Profit":
                            new_stop = min(profit_target, max_favorable_price * (1 - stop_loss_pct))
                        else:  # Maintain_Risk
                            new_stop = max_favorable_price * (1 - stop_loss_pct)
                        
                        current_stop = max(current_stop, max(new_stop, yest_close))
                else:  # SELL
                    if current_price < max_favorable_price:
                        max_favorable_price = current_price
                        max_profit_pct = ((entry_price - max_favorable_price) / entry_price) * 100
                        
                        # Update trailing stop based on strategy
                        if trailing_strategy == "Lock_Profit":
                            new_stop = max(profit_target, max_favorable_price * (1 + stop_loss_pct))
                        else:  # Maintain_Risk
                            new_stop = max_favorable_price * (1 + stop_loss_pct)
                        
                        current_stop = min(current_stop, new_stop)
                
                # Check exit conditions
                exit_reason = None
                
                if trade_type == "BUY":
                    if current_price <= current_stop:
                        exit_reason = "Stop Loss"
                    elif current_price >= profit_target:
                        exit_reason = "Profit Target"
                else:  # SELL
                    if current_price >= current_stop:
                        exit_reason = "Stop Loss"
                    elif current_price <= profit_target:
                        exit_reason = "Profit Target"
                
                if current_time.time() >= time(15, 15):
                    exit_reason = "Market Close"
                
                if exit_reason:
                    exit_price = current_price
                    exit_time = current_time
                    
                    if trade_type == "BUY":
                        profit_loss_pct = ((exit_price - entry_price) / entry_price) * 100
                    else:  # SELL
                        profit_loss_pct = ((entry_price - exit_price) / entry_price) * 100
                    
                    trade_duration = int((exit_time - entry_time).total_seconds() / 60)
                    stop_loss_hit = "Yes" if exit_reason == "Stop Loss" else "No"
                    
                    return BacktestLiveVolumeBoost._create_trade_record(
                        symbol, entry_price, exit_price, entry_time, exit_time,
                        profit_loss_pct, current_stop, max_favorable_price, max_profit_pct,
                        stop_loss_hit, trade_duration, entry_window, 
                        entry_volume_pct, yest_volume, trigger_price, today_volume, trade_type, trailing_strategy
                    )
            
            # Exit at market close if no other exit triggered
            last_row = remaining_data.iloc[-1]
            exit_price = float(last_row['close'])
            exit_time = last_row['date']
            
            if trade_type == "BUY":
                profit_loss_pct = ((exit_price - entry_price) / entry_price) * 100
            else:  # SELL
                profit_loss_pct = ((entry_price - exit_price) / entry_price) * 100
            
            trade_duration = int((exit_time - entry_time).total_seconds() / 60)
            
            return BacktestLiveVolumeBoost._create_trade_record(
                symbol, entry_price, exit_price, entry_time, exit_time,
                profit_loss_pct, current_stop, max_favorable_price, max_profit_pct,
                "No", trade_duration, entry_window, 
                entry_volume_pct, yest_volume, trigger_price, today_volume, trade_type, trailing_strategy
            )
            
        except Exception as e:
            return None
    
    @staticmethod
    def _create_trade_record(symbol, entry_price, exit_price, entry_time, exit_time,
                           profit_loss_pct, trailing_stop, max_price, max_profit_pct,
                           stop_loss_hit, trade_duration, entry_window, entry_volume_pct,
                           yest_volume, trigger_price, today_volume, trade_type, trailing_strategy):
        """Create standardized trade record"""
        return {
            'Symbol': symbol,
            'Trade_Type': trade_type,
            'Trailing_Strategy': trailing_strategy,
            'Entry_Price': round(entry_price, 2),
            'Exit_Price': round(exit_price, 2),
            'Entry_Time': entry_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Exit_Time': exit_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Profit_Loss_Pct': round(profit_loss_pct, 2),
            'Trailing_Stop': round(trailing_stop, 2),
            'Entry_Volume_Pct': round(entry_volume_pct, 2),
            'Yest_Volume': round(yest_volume, 2),
            'Trigger_Price': round(trigger_price, 2),
            'Today_Volume': round(today_volume, 2),
            'Max_Price': round(max_price, 2),
            'Max_Profit_Pct': round(max_profit_pct, 2),
            'Stop_Loss_Hit': stop_loss_hit,
            'Trade_Duration_Min': trade_duration,
            'Entry_Window': entry_window
        }

# Test function
if __name__ == "__main__":
    print("🧪 Running BUY+SELL Live Volume Boost Backtest Test...")
    success, message = BacktestLiveVolumeBoost.run_backtest_live_volume_boost()
    
    if success:
        print(f"🎉 SUCCESS: {message}")
    else:
        print(f"💥 FAILED: {message}")
    
    print("✅ Test completed!")