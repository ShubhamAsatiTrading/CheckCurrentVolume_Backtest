import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from typing import Dict, Optional, List
import time
import os

class ScreenerScraper:
    def __init__(self):
        self.base_url = "https://www.screener.in/company/{}/consolidated/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def get_stock_data(self, symbol: str) -> Dict:
        """
        Scrape stock data for a given symbol from screener.in
        
        Args:
            symbol (str): Stock symbol (e.g., 'RELIANCE', 'TCS', etc.)
            
        Returns:
            Dict: Dictionary containing all the scraped financial data
        """
        url = self.base_url.format(symbol.upper())
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract basic financial metrics
            financial_data = self._extract_financial_metrics(soup)
            
            # Extract shareholding pattern
            shareholding_data = self._extract_shareholding_pattern(soup)
            
            return {
                'symbol': symbol.upper(),
                'url': url,
                'financial_metrics': financial_data,
                'shareholding_pattern': shareholding_data,
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except requests.RequestException as e:
            return {'symbol': symbol.upper(), 'error': f'Request failed: {str(e)}'}
        except Exception as e:
            return {'symbol': symbol.upper(), 'error': f'Scraping failed: {str(e)}'}
    
    def _extract_financial_metrics(self, soup: BeautifulSoup) -> Dict:
        """Extract financial metrics from the webpage"""
        metrics = {}
        
        try:
            # Market Cap
            market_cap = self._find_metric_value(soup, 'Market Cap')
            metrics['market_cap'] = self._clean_currency_value(market_cap)
            
            # Current Price
            current_price = self._find_metric_value(soup, 'Current Price')
            metrics['current_price'] = self._clean_currency_value(current_price)
            
            # High / Low
            high_low = self._find_metric_value(soup, 'High / Low')
            if high_low:
                high_low_parts = high_low.split('/')
                if len(high_low_parts) == 2:
                    metrics['high'] = self._clean_currency_value(high_low_parts[0].strip())
                    metrics['low'] = self._clean_currency_value(high_low_parts[1].strip())
            
            # Stock P/E
            pe_ratio = self._find_metric_value(soup, 'Stock P/E')
            metrics['pe_ratio'] = self._clean_numeric_value(pe_ratio)
            
            # Book Value
            book_value = self._find_metric_value(soup, 'Book Value')
            metrics['book_value'] = self._clean_currency_value(book_value)
            
            # Dividend Yield
            dividend_yield = self._find_metric_value(soup, 'Dividend Yield')
            metrics['dividend_yield'] = self._clean_percentage_value(dividend_yield)
            
            # ROCE
            roce = self._find_metric_value(soup, 'ROCE')
            metrics['roce'] = self._clean_percentage_value(roce)
            
            # ROE
            roe = self._find_metric_value(soup, 'ROE')
            metrics['roe'] = self._clean_percentage_value(roe)
            
            # Face Value
            face_value = self._find_metric_value(soup, 'Face Value')
            metrics['face_value'] = self._clean_currency_value(face_value)
            
        except Exception as e:
            metrics['extraction_error'] = str(e)
        
        return metrics
    
    def _extract_shareholding_pattern(self, soup: BeautifulSoup) -> Dict:
        """Extract shareholding pattern data"""
        shareholding = {}
        
        try:
            # Look for shareholding pattern section
            shareholding_section = soup.find('section', {'id': 'shareholding'})
            if not shareholding_section:
                # Alternative search methods
                shareholding_section = soup.find('h2', string=re.compile(r'Shareholding Pattern', re.I))
                if shareholding_section:
                    shareholding_section = shareholding_section.find_parent()
            
            if shareholding_section:
                # Find the table or div containing shareholding data
                table = shareholding_section.find('table') or shareholding_section.find_next('table')
                
                if table:
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all(['td', 'th'])
                        if len(cells) >= 2:
                            category = cells[0].get_text(strip=True)
                            percentage = cells[-1].get_text(strip=True)
                            
                            # Clean and categorize the data
                            if 'promoter' in category.lower():
                                shareholding['promoters'] = self._clean_percentage_value(percentage)
                            elif 'fii' in category.lower() or 'foreign' in category.lower():
                                shareholding['fiis'] = self._clean_percentage_value(percentage)
                            elif 'dii' in category.lower() or 'domestic' in category.lower():
                                shareholding['diis'] = self._clean_percentage_value(percentage)
                            elif 'government' in category.lower():
                                shareholding['government'] = self._clean_percentage_value(percentage)
                            elif 'public' in category.lower():
                                shareholding['public'] = self._clean_percentage_value(percentage)
            
            # Alternative method: Look for specific patterns in the HTML
            if not shareholding:
                shareholding = self._extract_shareholding_alternative(soup)
                
        except Exception as e:
            shareholding['extraction_error'] = str(e)
        
        return shareholding
    
    def _extract_shareholding_alternative(self, soup: BeautifulSoup) -> Dict:
        """Alternative method to extract shareholding pattern"""
        shareholding = {}
        
        # Look for specific text patterns
        text_content = soup.get_text()
        
        # Use regex to find shareholding percentages
        patterns = {
            'promoters': r'Promoters[:\s]+(\d+\.?\d*)\s*%',
            'fiis': r'FIIs?[:\s]+(\d+\.?\d*)\s*%',
            'diis': r'DIIs?[:\s]+(\d+\.?\d*)\s*%',
            'government': r'Government[:\s]+(\d+\.?\d*)\s*%',
            'public': r'Public[:\s]+(\d+\.?\d*)\s*%'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                shareholding[key] = float(match.group(1))
        
        return shareholding
    
    def _find_metric_value(self, soup: BeautifulSoup, metric_name: str) -> Optional[str]:
        """Find the value for a specific metric"""
        # Method 1: Look for the metric name in spans or labels
        metric_element = soup.find(string=re.compile(metric_name, re.I))
        if metric_element:
            parent = metric_element.find_parent()
            if parent:
                # Look for the value in the next sibling or nearby elements
                next_element = parent.find_next_sibling()
                if next_element:
                    return next_element.get_text(strip=True)
                
                # Look for value in the same parent
                value_element = parent.find('span', class_=re.compile(r'value|number'))
                if value_element:
                    return value_element.get_text(strip=True)
                
                
        
        # Method 2: Look in ratio tables
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    if metric_name.lower() in cells[0].get_text(strip=True).lower():
                        return cells[1].get_text(strip=True)
        
        return None
    
    def _clean_currency_value(self, value: str) -> Optional[float]:
        """Clean currency values (remove ₹, commas, etc.)"""
        if not value:
            return None
        
        # Remove currency symbols and commas
        cleaned = re.sub(r'[₹,\s]', '', value)
        
        # Handle Cr. (Crores)
        if 'Cr' in cleaned:
            cleaned = re.sub(r'Cr.*', '', cleaned)
            try:
                return float(cleaned) * 10000000  # Convert crores to actual number
            except:
                return None
        
        try:
            return float(cleaned)
        except:
            return None
    
    def _clean_percentage_value(self, value: str) -> Optional[float]:
        """Clean percentage values"""
        if not value:
            return None
        
        cleaned = re.sub(r'[%\s]', '', value)
        try:
            return float(cleaned)
        except:
            return None
    
    def _clean_numeric_value(self, value: str) -> Optional[float]:
        """Clean numeric values"""
        if not value:
            return None
        
        cleaned = re.sub(r'[,\s]', '', value)
        try:
            return float(cleaned)
        except:
            return None
    
    def safe_format_currency(self, value, suffix=""):
        """Safely format currency values, handling None"""
        if value is None:
            return "N/A"
        return f"₹{value:,.0f}{suffix}"
    
    def safe_format_percentage(self, value):
        """Safely format percentage values, handling None"""
        if value is None:
            return "N/A"
        return f"{value}%"
    
    def safe_format_number(self, value):
        """Safely format numeric values, handling None"""
        if value is None:
            return "N/A"
        return f"{value}"

    def print_stock_data(self, data: Dict):
        """Pretty print the stock data"""
        if 'error' in data:
            print(f"Error: {data['error']}")
            return
        
        print(f"\n{'='*50}")
        print(f"STOCK DATA FOR {data['symbol']}")
        print(f"{'='*50}")
        
        financial = data.get('financial_metrics', {})
        print("\nFINANCIAL METRICS:")
        print("-" * 30)
        
        print(f"Market Cap: {self.safe_format_currency(financial.get('market_cap'), ' Cr.')}")
        print(f"Current Price: {self.safe_format_currency(financial.get('current_price'))}")
        
        high = financial.get('high')
        low = financial.get('low')
        if high is not None and low is not None:
            print(f"High / Low: ₹{high} / ₹{low}")
        else:
            print(f"High / Low: N/A")
        
        print(f"Stock P/E: {self.safe_format_number(financial.get('pe_ratio'))}")
        print(f"Book Value: {self.safe_format_currency(financial.get('book_value'))}")
        print(f"Dividend Yield: {self.safe_format_percentage(financial.get('dividend_yield'))}")
        print(f"ROCE: {self.safe_format_percentage(financial.get('roce'))}")
        print(f"ROE: {self.safe_format_percentage(financial.get('roe'))}")
        print(f"Face Value: {self.safe_format_currency(financial.get('face_value'))}")
        
        shareholding = data.get('shareholding_pattern', {})
        if shareholding and 'extraction_error' not in shareholding:
            print("\nSHAREHOLDING PATTERN:")
            print("-" * 30)
            for category, percentage in shareholding.items():
                if category != 'extraction_error':
                    print(f"{category.capitalize()}: {self.safe_format_percentage(percentage)}")
        elif shareholding.get('extraction_error'):
            print(f"\nSHAREHOLDING PATTERN: Error - {shareholding['extraction_error']}")
        else:
            print("\nSHAREHOLDING PATTERN: No data found")
        
        print(f"\nScraped at: {data.get('scraped_at', 'Unknown')}")
        print(f"Source: {data.get('url', 'Unknown')}")


# ================================
# PUBLIC API FUNCTIONS
# ================================

def read_symbols_from_file(filename="symbols.txt"):
    """
    Read stock symbols from a text file
    
    Args:
        filename (str): Path to the symbols file
        
    Returns:
        List[str]: List of stock symbols
    """
    symbols = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                symbol = line.strip().upper()
                if symbol and not symbol.startswith('#'):  # Skip empty lines and comments
                    symbols.append(symbol)
        return symbols
    except FileNotFoundError:
        print(f"File {filename} not found. Creating a sample file...")
        # Create a sample symbols.txt file
        sample_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC", "WIPRO", "LT", "BHARTIARTL", "KOTAKBANK", "MARUTI"]
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Add stock symbols one per line\n")
            f.write("# Lines starting with # are ignored\n\n")
            for symbol in sample_symbols:
                f.write(f"{symbol}\n")
        print(f"Sample file created with symbols: {', '.join(sample_symbols)}")
        return sample_symbols
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return []


def scrape_single_stock(symbol: str, verbose=False) -> Dict:
    """
    Scrape data for a single stock symbol
    
    Args:
        symbol (str): Stock symbol to scrape
        verbose (bool): Whether to print progress messages
        
    Returns:
        Dict: Stock data dictionary
    """
    scraper = ScreenerScraper()
    if verbose:
        print(f"Scraping {symbol}...")
    
    data = scraper.get_stock_data(symbol)
    
    if verbose:
        if 'error' in data:
            print(f"✗ Error scraping {symbol}: {data['error']}")
        else:
            print(f" Successfully scraped {symbol}")
    
    return data


def scrape_all_symbols_from_file(filename="symbols.txt", verbose=True, delay=1.0) -> List[Dict]:
    """
    Scrape data for all symbols listed in a file
    
    Args:
        filename (str): Path to the symbols file
        verbose (bool): Whether to print progress messages
        delay (float): Delay in seconds between requests (to be respectful to the server)
        
    Returns:
        List[Dict]: List of stock data dictionaries
    """
    symbols = read_symbols_from_file(filename)
    if not symbols:
        return []
    
    if verbose:
        print(f"\nStarting batch scraping for {len(symbols)} symbols...")
        print("=" * 60)
    
    scraper = ScreenerScraper()
    all_data = []
    successful_scrapes = 0
    
    for i, symbol in enumerate(symbols, 1):
        if verbose:
            print(f"\n[{i}/{len(symbols)}] Scraping {symbol}...")
        
        try:
            data = scraper.get_stock_data(symbol)
            all_data.append(data)
            
            if 'error' not in data:
                successful_scrapes += 1
                if verbose:
                    financial = data.get('financial_metrics', {})
                    market_cap = financial.get('market_cap')
                    current_price = financial.get('current_price')
                    
                    print(f"   Market Cap: {scraper.safe_format_currency(market_cap, ' Cr.')}")
                    print(f"   Current Price: {scraper.safe_format_currency(current_price)}")
            else:
                if verbose:
                    print(f"  ✗ Error: {data['error']}")
                
        except Exception as e:
            error_data = {'symbol': symbol, 'error': str(e)}
            all_data.append(error_data)
            if verbose:
                print(f"  ✗ Exception: {e}")
        
        # Add delay to be respectful to the server
        if i < len(symbols) and delay > 0:
            time.sleep(delay)
    
    if verbose:
        print(f"\nBatch Summary:")
        print(f"Total symbols processed: {len(symbols)}")
        print(f"Successful scrapes: {successful_scrapes}")
        print(f"Failed scrapes: {len(symbols) - successful_scrapes}")
    
    return all_data


def save_data_to_excel(data, filename="stock_data.xlsx"):
    """
    Save stock data to an Excel file with multiple sheets
    
    Args:
        data: Stock data (single dict or list of dicts)
        filename (str): Output filename
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure data is a list
        if isinstance(data, dict):
            data = [data]
        
        # Separate successful and failed data
        successful_data = [item for item in data if 'error' not in item]
        failed_data = [item for item in data if 'error' in item]
        
        # Create Excel writer object
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            
            # Sheet 1: Financial Metrics Summary
            if successful_data:
                financial_rows = []
                for item in successful_data:
                    symbol = item.get('symbol', 'Unknown')
                    financial = item.get('financial_metrics', {})
                    scraped_at = item.get('scraped_at', '')
                    url = item.get('url', '')
                    
                    row = {
                        'Symbol': symbol,
                        'Market Cap (Cr)': financial.get('market_cap'),
                        'Current Price': financial.get('current_price'),
                        'High': financial.get('high'),
                        'Low': financial.get('low'),
                        'P/E Ratio': financial.get('pe_ratio'),
                        'Book Value': financial.get('book_value'),
                        'Dividend Yield (%)': financial.get('dividend_yield'),
                        'ROE (%)': financial.get('roe'),
                        'ROCE (%)': financial.get('roce'),
                        'Face Value': financial.get('face_value'),
                        'Scraped At': scraped_at,
                        'URL': url
                    }
                    financial_rows.append(row)
                
                financial_df = pd.DataFrame(financial_rows)
                financial_df.to_excel(writer, sheet_name='Financial Metrics', index=False)
            
            # Sheet 2: Shareholding Pattern
            if successful_data:
                shareholding_rows = []
                for item in successful_data:
                    symbol = item.get('symbol', 'Unknown')
                    shareholding = item.get('shareholding_pattern', {})
                    
                    # Skip if no shareholding data or if there's an error
                    if not shareholding or 'extraction_error' in shareholding:
                        continue
                    
                    row = {
                        'Symbol': symbol,
                        'Promoters (%)': shareholding.get('promoters'),
                        'FIIs (%)': shareholding.get('fiis'),
                        'DIIs (%)': shareholding.get('diis'),
                        'Government (%)': shareholding.get('government'),
                        'Public (%)': shareholding.get('public')
                    }
                    shareholding_rows.append(row)
                
                if shareholding_rows:
                    shareholding_df = pd.DataFrame(shareholding_rows)
                    shareholding_df.to_excel(writer, sheet_name='Shareholding Pattern', index=False)
            
            # Sheet 3: Failed Scrapes (if any)
            if failed_data:
                failed_rows = []
                for item in failed_data:
                    row = {
                        'Symbol': item.get('symbol', 'Unknown'),
                        'Error': item.get('error', 'Unknown error')
                    }
                    failed_rows.append(row)
                
                failed_df = pd.DataFrame(failed_rows)
                failed_df.to_excel(writer, sheet_name='Failed Scrapes', index=False)
            
            # Sheet 4: Summary Statistics
            if successful_data:
                summary_stats = {
                    'Metric': [
                        'Total Stocks Processed',
                        'Successful Scrapes',
                        'Failed Scrapes',
                        'Success Rate (%)',
                        'Average Market Cap (Cr)',
                        'Average P/E Ratio',
                        'Average ROE (%)',
                        'Average ROCE (%)'
                    ],
                    'Value': []
                }
                
                total_processed = len(data)
                successful_count = len(successful_data)
                failed_count = len(failed_data)
                success_rate = (successful_count / total_processed * 100) if total_processed > 0 else 0
                
                # Calculate averages (excluding None values)
                market_caps = [item.get('financial_metrics', {}).get('market_cap') 
                              for item in successful_data 
                              if item.get('financial_metrics', {}).get('market_cap') is not None]
                avg_market_cap = sum(market_caps) / len(market_caps) if market_caps else 0
                
                pe_ratios = [item.get('financial_metrics', {}).get('pe_ratio') 
                            for item in successful_data 
                            if item.get('financial_metrics', {}).get('pe_ratio') is not None]
                avg_pe = sum(pe_ratios) / len(pe_ratios) if pe_ratios else 0
                
                roes = [item.get('financial_metrics', {}).get('roe') 
                       for item in successful_data 
                       if item.get('financial_metrics', {}).get('roe') is not None]
                avg_roe = sum(roes) / len(roes) if roes else 0
                
                roces = [item.get('financial_metrics', {}).get('roce') 
                        for item in successful_data 
                        if item.get('financial_metrics', {}).get('roce') is not None]
                avg_roce = sum(roces) / len(roces) if roces else 0
                
                summary_stats['Value'] = [
                    total_processed,
                    successful_count,
                    failed_count,
                    round(success_rate, 2),
                    round(avg_market_cap, 2) if avg_market_cap else 'N/A',
                    round(avg_pe, 2) if avg_pe else 'N/A',
                    round(avg_roe, 2) if avg_roe else 'N/A',
                    round(avg_roce, 2) if avg_roce else 'N/A'
                ]
                
                summary_df = pd.DataFrame(summary_stats)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f" Data saved to {filename}")
        return True
        
    except Exception as e:
        print(f"✗ Error saving data to {filename}: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return False


def create_summary_report(all_data, output_file="stock_summary.txt"):
    """
    Create a summary report of scraped stock data
    
    Args:
        all_data (List[Dict]): List of stock data dictionaries
        output_file (str): Output filename for the report
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("STOCK SUMMARY REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            successful_data = [data for data in all_data if 'error' not in data]
            failed_data = [data for data in all_data if 'error' in data]
            
            f.write(f"Total Stocks Processed: {len(all_data)}\n")
            f.write(f"Successful: {len(successful_data)}\n")
            f.write(f"Failed: {len(failed_data)}\n\n")
            
            if failed_data:
                f.write("FAILED SCRAPES:\n")
                f.write("-" * 20 + "\n")
                for data in failed_data:
                    f.write(f"• {data.get('symbol', 'Unknown')}: {data.get('error', 'Unknown error')}\n")
                f.write("\n")
            
            f.write("SUCCESSFUL SCRAPES:\n")
            f.write("=" * 50 + "\n\n")
            
            for data in successful_data:
                symbol = data.get('symbol', 'Unknown')
                financial = data.get('financial_metrics', {})
                
                f.write(f"SYMBOL: {symbol}\n")
                f.write("-" * 20 + "\n")
                
                # Write financial metrics with better formatting
                market_cap = financial.get('market_cap')
                current_price = financial.get('current_price')
                high = financial.get('high')
                low = financial.get('low')
                pe_ratio = financial.get('pe_ratio')
                roe = financial.get('roe')
                roce = financial.get('roce')
                dividend_yield = financial.get('dividend_yield')
                book_value = financial.get('book_value')
                face_value = financial.get('face_value')
                
                f.write(f"Market Cap:      {f'₹{market_cap:,.0f} Cr.' if market_cap else 'N/A'}\n")
                f.write(f"Current Price:   {f'₹{current_price:,.2f}' if current_price else 'N/A'}\n")
                
                if high is not None and low is not None:
                    f.write(f"High / Low:      ₹{high} / ₹{low}\n")
                else:
                    f.write(f"High / Low:      N/A\n")
                
                f.write(f"P/E Ratio:       {pe_ratio if pe_ratio else 'N/A'}\n")
                f.write(f"Book Value:      {f'₹{book_value:,.2f}' if book_value else 'N/A'}\n")
                f.write(f"Dividend Yield:  {f'{dividend_yield}%' if dividend_yield else 'N/A'}\n")
                f.write(f"ROE:             {f'{roe}%' if roe else 'N/A'}\n")
                f.write(f"ROCE:            {f'{roce}%' if roce else 'N/A'}\n")
                f.write(f"Face Value:      {f'₹{face_value}' if face_value else 'N/A'}\n")
                
                # Write shareholding pattern
                shareholding = data.get('shareholding_pattern', {})
                if shareholding and 'extraction_error' not in shareholding and any(v for k, v in shareholding.items() if k != 'extraction_error'):
                    f.write("\nShareholding Pattern:\n")
                    for category, percentage in shareholding.items():
                        if category != 'extraction_error' and percentage is not None:
                            f.write(f"  {category.capitalize():12}: {percentage}%\n")
                else:
                    f.write("\nShareholding Pattern: Data not available\n")
                
                f.write("\n" + "="*50 + "\n\n")
        
        print(f" Summary report saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"✗ Error creating summary report: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return False


def scrape_and_save_all(symbols_file="symbols.txt", 
                       excel_output="stock_data_batch.xlsx", 
                       report_output="stock_summary.txt",
                       verbose=True,
                       delay=1.0):
    """
    Complete workflow: scrape all symbols from file and save results
    
    Args:
        symbols_file (str): Input file with symbols
        excel_output (str): Output Excel file
        report_output (str): Output summary report file
        verbose (bool): Print progress messages
        delay (float): Delay between requests
        
    Returns:
        List[Dict]: All scraped data
    """
    # Scrape all data
    all_data = scrape_all_symbols_from_file(symbols_file, verbose=verbose, delay=delay)
    
    if not all_data:
        if verbose:
            print("No data to save.")
        return []
    
    # Save to Excel
    save_data_to_excel(all_data, excel_output)
    
    # Create summary report
    create_summary_report(all_data, report_output)
    
    return all_data


# ================================
# STANDALONE EXECUTION
# ================================

def main():
    """Main function for standalone execution"""
    scraper = ScreenerScraper()
    
    print("Stock Scraper - Choose an option:")
    print("1. Scrape single stock (manual input)")
    print("2. Scrape stocks from symbols.txt file")
    print("3. Scrape sample stocks")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "1":
        # Single stock scraping
        symbol = input("Enter stock symbol (e.g., RELIANCE): ").strip().upper()
        if not symbol:
            symbol = "RELIANCE"
        
        print(f"Scraping data for {symbol}...")
        data = scraper.get_stock_data(symbol)
        scraper.print_stock_data(data)
        
        # Optionally save to Excel
        save_excel = input("\nSave data to Excel file? (y/n): ").strip().lower()
        if save_excel == 'y':
            filename = f"{symbol}_data.xlsx"
            save_data_to_excel(data, filename)
    
    elif choice == "2":
        # Batch scraping from file
        all_data = scrape_and_save_all()
    
    elif choice == "3":
        # Sample stocks
        sample_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]
        print(f"Scraping sample stocks: {', '.join(sample_symbols)}")
        
        # Create temporary symbols file
        with open("temp_symbols.txt", 'w', encoding='utf-8') as f:
            for symbol in sample_symbols:
                f.write(f"{symbol}\n")
        
        all_data = scrape_and_save_all("temp_symbols.txt", "sample_stock_data.xlsx", "sample_summary.txt")
        
        # Clean up
        try:
            os.remove("temp_symbols.txt")
        except:
            pass
    
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()

