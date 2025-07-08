# historical_data_download.py - Fixed Version
# Separated core logic from Streamlit UI

import os
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import time

def download_historical_data(kite, symbol_names_file="symbols.txt", output_folder="stocks_historical_data", 
                           start=None, interval="1minute", max_workers=3):
    """
    Core function to download historical data - NO STREAMLIT CODE
    
    Args:
        kite: KiteConnect instance
        symbol_names_file: File containing stock symbols
        output_folder: Directory to save data
        start: Start date for historical data
        interval: Data interval (e.g., "1minute", "1hour")
        max_workers: Number of parallel workers
    
    Returns:
        bool: Success status
        str: Status message
    """
    try:
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Read symbols
        if not os.path.exists(symbol_names_file):
            return False, f"Symbol file {symbol_names_file} not found"
        
        with open(symbol_names_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        
        if not symbols:
            return False, "No symbols found in file"
        
        # Set default start date if not provided
        if start is None:
            start = datetime.now() - timedelta(days=365)
        
        # Download data for each symbol
        success_count = 0
        error_count = 0
        
        print(f"Starting download for {len(symbols)} symbols...")
        
        def download_symbol(symbol):
            """Download data for a single symbol"""
            try:
                # Get historical data from Kite
                data = kite.historical_data(
                    instrument_token=kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["instrument_token"],
                    from_date=start,
                    to_date=datetime.now(),
                    interval=interval
                )
                
                if data:
                    # Convert to DataFrame
                    df = pd.DataFrame(data)
                    
                    # Save to CSV
                    filename = os.path.join(output_folder, f"{symbol}_historical.csv")
                    df.to_csv(filename, index=False)
                    
                    print(f"✅ {symbol}: {len(df)} records saved")
                    return True
                else:
                    print(f"❌ {symbol}: No data received")
                    return False
                    
            except Exception as e:
                print(f"❌ {symbol}: Error - {e}")
                return False
        
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(download_symbol, symbols))
        
        # Count results
        success_count = sum(results)
        error_count = len(symbols) - success_count
        
        message = f"Download completed: {success_count} successful, {error_count} failed"
        print(message)
        
        return success_count > 0, message
        
    except Exception as e:
        error_msg = f"Critical error in historical data download: {e}"
        print(error_msg)
        return False, error_msg

def download_historical_data_simple(kite, symbols_list=None, output_folder="stocks_historical_data", 
                                  days_back=365, interval="minute"):
    """
    Simplified version for direct calls from main.py
    
    Args:
        kite: KiteConnect instance
        symbols_list: List of symbols (if None, reads from symbols.txt)
        output_folder: Output directory
        days_back: Number of days to go back
        interval: Data interval
    
    Returns:
        bool: Success status
        str: Status message
    """
    try:
        # Get symbols
        if symbols_list is None:
            if os.path.exists("symbols.txt"):
                with open("symbols.txt", 'r') as f:
                    symbols_list = [line.strip() for line in f if line.strip()]
            else:
                return False, "No symbols provided and symbols.txt not found"
        
        # Calculate start date
        start_date = datetime.now() - timedelta(days=days_back)
        
        # Call main function
        return download_historical_data(
            kite=kite,
            symbol_names_file="symbols.txt" if symbols_list is None else None,
            output_folder=output_folder,
            start=start_date,
            interval=interval,
            max_workers=3
        )
        
    except Exception as e:
        return False, f"Error in simplified download: {e}"

# ==========================================
# STREAMLIT UI CODE - Only runs when script is executed directly
# ==========================================

if __name__ == "__main__":
    import streamlit as st
    from dotenv import load_dotenv
    
    # Only set page config when running as standalone script
    st.set_page_config(page_title="Historical Data Downloader", page_icon="📈", layout="wide")
    
    load_dotenv()
    
    st.title("📈 Historical Data Downloader")
    st.markdown("Download historical stock data using Kite Connect API")
    
    # Your existing Streamlit UI code goes here...
    # (All the UI elements, forms, buttons, etc.)
    
    # Example UI:
    with st.form("download_form"):
        symbol_file = st.text_input("Symbol File", value="symbols.txt")
        output_folder = st.text_input("Output Folder", value="stocks_historical_data")
        days_back = st.number_input("Days Back", min_value=1, max_value=3650, value=365)
        interval = st.selectbox("Interval", ["minute", "5minute", "15minute", "hour", "day"])
        max_workers = st.number_input("Max Workers", min_value=1, max_value=10, value=3)
        
        submitted = st.form_submit_button("Start Download")
        
        if submitted:
            # Get Kite instance (you'll need to implement this)
            # kite = get_kite_instance()
            
            st.info("This UI is for standalone use. Use main.py for integrated functionality.")