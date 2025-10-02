import json
import pandas as pd
import os
from datetime import datetime
import glob

def process_fundamental_data(data, ticker):
    if data is None:
        print(f"No fundamental data available for {ticker}. Skipping processing.")
        return
    def extract_quarterly_data(metric_name):
        quarterly_data = []
        
        if metric_name in data["facts"]["us-gaap"] and "units" in data["facts"]["us-gaap"][metric_name]:
            metric_data = data["facts"]["us-gaap"][metric_name]
            
            if "USD" in metric_data["units"]:
                used_periods = set()
                
                sorted_entries = sorted(
                    metric_data["units"]["USD"],
                    key=lambda x: x.get("filed", "0000-00-00"),
                    reverse=True
                )
                
                for entry in sorted_entries:
                    if ("fp" in entry and 
                        entry["fp"] is not None and 
                        isinstance(entry["fp"], str) and 
                        (entry["fp"].startswith("Q") or 
                         (entry["fp"] == "FY" and entry.get("form") == "10-K")) and 
                        "fy" in entry and 
                        "val" in entry):
                        
                        period_fp = "Q4" if entry["fp"] == "FY" else entry["fp"]
                        period_key = f"{entry['fy']}-{period_fp}"
                        
                        if period_key in used_periods:
                            continue
                            
                        used_periods.add(period_key)
                        
                        quarterly_data.append({
                            "date": period_key,
                            "value": entry["val"],
                            "metric": metric_name,
                            "filed_date": entry.get("filed", "")
                        })
        
        return quarterly_data

    all_available_metrics = list(data["facts"]["us-gaap"].keys())
    print(f"Found {len(all_available_metrics)} total metrics in the data")

    all_quarterly_data = []
    metrics_with_data = 0

    for i, metric in enumerate(all_available_metrics):
        metric_data = extract_quarterly_data(metric)
        
        if metric_data:
            all_quarterly_data.extend(metric_data)
            metrics_with_data += 1
        

    print(f"Found data for {metrics_with_data} metrics out of {len(all_available_metrics)} total metrics")

    if all_quarterly_data:
        df = pd.DataFrame(all_quarterly_data)
        
        duplicate_check = df.duplicated(subset=['date', 'metric'])
        if duplicate_check.any():
            print(f"Warning: Found {duplicate_check.sum()} duplicate entries. Keeping only the first occurrence.")
            df = df.drop_duplicates(subset=['date', 'metric'])
        
        pivoted_df = df.pivot(index='date', columns='metric', values='value')
        
        pivoted_df = pivoted_df.sort_index()
        
        filename = f"datasets/fundamental/{ticker}_time.csv"
        
        pivoted_df.to_csv(filename)
        print(f"Data saved to {filename}")
        print(f"Saved {len(pivoted_df)} quarters of data for {len(pivoted_df.columns)} metrics")
        
        print(f"\nDataFrame statistics:")
        print(f"Number of quarters: {len(pivoted_df)}")
        print(f"Number of metrics: {len(pivoted_df.columns)}")
        print(f"Number of data points: {pivoted_df.count().sum()}")
        print(f"Data completeness: {(pivoted_df.count().sum() / (len(pivoted_df) * len(pivoted_df.columns)) * 100):.2f}%")
        
        
        return pivoted_df
    return None

def get_fundamental_data_local(ticker, cik=None):
    if cik is None:
        if ticker.upper() == "TSLA":
            cik = "1318605"
        else:
            print(f"CIK untuk ticker {ticker} tidak diketahui. Silakan berikan CIK secara manual.")
            return None
    
    if isinstance(cik, str) and not cik.startswith("CIK"):
        cik_formatted = f"CIK{cik.zfill(10)}"
    elif isinstance(cik, int):
        cik_formatted = f"CIK{str(cik).zfill(10)}"
    else:
        cik_formatted = cik
    
    sec_data_dir = "datasets/fundamental/sec_data"
    
    json_file_path = os.path.join(sec_data_dir, f"{cik_formatted}.json")
    
    if os.path.exists(json_file_path):
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            print(f"Data fundamental untuk {ticker} (CIK: {cik_formatted}) berhasil dimuat dari file lokal.")
            print(f"File: {json_file_path}")
            return data
        except Exception as e:
            print(f"Error membaca file {json_file_path}: {e}")
            return None
    else:
        print(f"File tidak ditemukan: {json_file_path}")
        pattern = os.path.join(sec_data_dir, f"*{cik}*.json")
        matching_files = glob.glob(pattern)
        if matching_files:
            print(f"File yang mungkin cocok ditemukan: {matching_files}")
        return None

def list_available_cik_files():
    sec_data_dir = "datasets/fundamental/sec_data"
    
    if not os.path.exists(sec_data_dir):
        print(f"Direktori {sec_data_dir} tidak ditemukan.")
        return []
    
    json_files = glob.glob(os.path.join(sec_data_dir, "CIK*.json"))
    cik_list = []
    
    for file_path in json_files:
        filename = os.path.basename(file_path)
        cik = filename.replace("CIK", "").replace(".json", "")
        cik_list.append({
            'cik': cik,
            'filename': filename,
            'path': file_path
        })
    
    print(f"Ditemukan {len(cik_list)} file CIK di {sec_data_dir}")
    return cik_list

def get_cik_from_local_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return {
            'cik': data.get('cik'),
            'entityName': data.get('entityName'),
            'file_path': file_path
        }
    except Exception as e:
        print(f"Error membaca file {file_path}: {e}")
        return None

def process_fundamental_data_local(ticker, cik=None, output_dir="datasets/fundamental"):
    data = get_fundamental_data_local(ticker, cik)
    
    if data is None:
        print(f"Gagal memuat data untuk ticker {ticker}")
        return None
    
    try:
        result_path = process_fundamental_data(data, ticker)
        print(f"Data fundamental untuk {ticker} berhasil diproses dari file lokal.")
        return result_path
    except Exception as e:
        print(f"Error memproses data fundamental untuk {ticker}: {e}")
        return None