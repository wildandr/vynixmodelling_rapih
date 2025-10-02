import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

def triple_barrier_method(data, volatility_window=20, upper_barrier_multiplier=1.0, 
                          lower_barrier_multiplier=1.0, time_barrier_days=5):
    data_copy = data.copy()

    if 'date' in data_copy.columns and not isinstance(data_copy.index, pd.DatetimeIndex):
        data_copy['date'] = pd.to_datetime(data_copy['date'])
        data_copy = data_copy.set_index('date')
    
    result = []
    
    returns = data_copy['close'].pct_change().fillna(0)
    
    volatility = returns.rolling(window=volatility_window).std().fillna(method='bfill')
    
    for i in range(len(data_copy) - time_barrier_days - 1):
        decision_date = data_copy.index[i]
        entry_date = data_copy.index[i+1]
        entry_price = data_copy['close'].iloc[i]
        
        upper_barrier = entry_price * (1 + volatility.iloc[i] * upper_barrier_multiplier)
        lower_barrier = entry_price * (1 - volatility.iloc[i] * lower_barrier_multiplier)
        
        data_window = data_copy.iloc[i+1:i+1+time_barrier_days]
        
        if len(data_window) == 0:
            continue
        
        upper_touch_idx = None
        lower_touch_idx = None
        upper_touch_value = None
        lower_touch_value = None
        
        for j in range(len(data_window)):
            if data_window['high'].iloc[j] >= upper_barrier:
                upper_touch_idx = j
                upper_touch_value = upper_barrier
                break
                
        for j in range(len(data_window)):
            if data_window['low'].iloc[j] <= lower_barrier:
                lower_touch_idx = j
                lower_touch_value = lower_barrier
                break
        
        if upper_touch_idx is not None and (lower_touch_idx is None or upper_touch_idx < lower_touch_idx):
            label = 1
            barrier_type = "upper"
            touch_date = data_window.index[upper_touch_idx]
            value_at_barrier = upper_touch_value
        elif lower_touch_idx is not None:
            label = -1
            barrier_type = "lower" 
            touch_date = data_window.index[lower_touch_idx]
            value_at_barrier = lower_touch_value
        else:
            label = 0
            barrier_type = "time"
            touch_date = data_window.index[-1] if len(data_window) > 0 else entry_date
            value_at_barrier = data_window['close'].iloc[-1] if len(data_window) > 0 else entry_price
        
        end_price = data_copy.loc[touch_date, 'close']
        actual_return = (end_price - entry_price) / entry_price
        
        result.append({
            'decision_date': decision_date,
            'entry_date': entry_date,
            'end_date': touch_date,
            'entry_price': entry_price,
            'end_price': end_price,
            'return': actual_return,
            'upper_barrier': upper_barrier,
            'lower_barrier': lower_barrier,
            'barrier_touched': barrier_type,
            'value_at_barrier_touched': value_at_barrier,
            'label': label
        })
    
    return pd.DataFrame(result)

def apply_triple_barrier_labeling(data, 
                                 volatility_window=20, 
                                 upper_barrier_multiplier=1.0, 
                                 lower_barrier_multiplier=1.0, 
                                 time_barrier_days=5,
                                 verbose=True):
    triple_barrier_df = triple_barrier_method(
        data,
        volatility_window=volatility_window,
        upper_barrier_multiplier=upper_barrier_multiplier,
        lower_barrier_multiplier=lower_barrier_multiplier,
        time_barrier_days=time_barrier_days
    )
    
    if verbose:
        print("\n=== Triple Barrier Method Results ===")
        print(f"Total samples generated: {len(triple_barrier_df)}")
        print("\nFirst few rows:")
        print(triple_barrier_df.head())
        
        label_counts = triple_barrier_df['label'].value_counts()
        print("\nLabel Distribution:")
        print(label_counts)
        print(f"Percentage UP (1): {label_counts.get(1, 0)/len(triple_barrier_df)*100:.2f}%")
        print(f"Percentage DOWN (-1): {label_counts.get(-1, 0)/len(triple_barrier_df)*100:.2f}%")
        print(f"Percentage NEUTRAL (0): {label_counts.get(0, 0)/len(triple_barrier_df)*100:.2f}%")
        
        print(f"\nBarrier Touch Statistics:")
        barrier_counts = triple_barrier_df['barrier_touched'].value_counts()
        for barrier_type, count in barrier_counts.items():
            print(f"{barrier_type.capitalize()}: {count} ({count/len(triple_barrier_df)*100:.2f}%)")
    
    return triple_barrier_df