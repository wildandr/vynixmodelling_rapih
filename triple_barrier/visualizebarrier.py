import plotly.graph_objects as go
import pandas as pd
import random
import os
from datetime import datetime

def visualize_triple_barrier_sample(data, triple_barrier_df, label_value, window_size=50):
    label_samples = triple_barrier_df[triple_barrier_df['label'] == label_value]
    
    if len(label_samples) == 0:
        print(f"Tidak ada sampel dengan label {label_value}")
        return None
    
    random_sample = label_samples.sample(1).iloc[0]
    
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'date' in data.columns:
            data = data.copy()
            data['date'] = pd.to_datetime(data['date'])
            data = data.set_index('date')
    
    if 'entry_date' in random_sample:
        entry_date = pd.to_datetime(random_sample['entry_date'])
    elif 'decision_date' in random_sample:
        entry_date = pd.to_datetime(random_sample['decision_date'])
    else:
        date_cols = [col for col in random_sample.index if 'date' in col.lower() and col != 'end_date']
        if date_cols:
            entry_date = pd.to_datetime(random_sample[date_cols[0]])
        else:
            raise KeyError("Tidak dapat menemukan kolom tanggal entry")
    
    entry_idx = data.index.get_indexer([entry_date], method='nearest')[0]
    
    start_window = max(0, entry_idx - window_size)
    end_window = min(len(data), entry_idx + window_size)
    sample_data = data.iloc[start_window:end_window]
    
    end_date = pd.to_datetime(random_sample['end_date'])
    end_idx_rel = sample_data.index.get_indexer([end_date], method='nearest')[0]
    
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=sample_data.index,
        open=sample_data['open'],
        high=sample_data['high'],
        low=sample_data['low'],
        close=sample_data['close'],
        name='Price',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))
    
    fig.add_trace(go.Scatter(
        x=[sample_data.index[0], sample_data.index[-1]],
        y=[random_sample['upper_barrier'], random_sample['upper_barrier']],
        mode='lines',
        line=dict(color='green', width=2, dash='dash'),
        name='Upper Barrier'
    ))
    
    fig.add_trace(go.Scatter(
        x=[sample_data.index[0], sample_data.index[-1]],
        y=[random_sample['lower_barrier'], random_sample['lower_barrier']],
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Lower Barrier'
    ))
    
    entry_idx_rel = sample_data.index.get_indexer([entry_date], method='nearest')[0]
    fig.add_trace(go.Scatter(
        x=[sample_data.index[entry_idx_rel]],
        y=[random_sample['entry_price']],
        mode='markers',
        marker=dict(color='blue', size=10, symbol='circle'),
        name='Entry Point'
    ))
    
    barrier_colors = {'upper': 'green', 'lower': 'red', 'time': 'purple'}
    touch_value = random_sample.get('value_at_barrier_touched', random_sample['end_price'])
    
    fig.add_trace(go.Scatter(
        x=[sample_data.index[end_idx_rel]],
        y=[touch_value],
        mode='markers',
        marker=dict(
            color=barrier_colors[random_sample['barrier_touched']], 
            size=10, 
            symbol='star'
        ),
        name=f"{random_sample['barrier_touched'].capitalize()} Barrier Touch"
    ))
    
    fig.add_annotation(
        x=sample_data.index[entry_idx_rel],
        y=random_sample['entry_price'],
        text=f"Entry: {random_sample['entry_price']:.4f}",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-40
    )
    
    fig.add_annotation(
        x=sample_data.index[end_idx_rel],
        y=touch_value,
        text=f"Touch: {touch_value:.4f}",
        showarrow=True,
        arrowhead=2,
        ax=-40,
        ay=-40
    )
    
    fig.add_shape(
        type="rect",
        x0=sample_data.index[entry_idx_rel],
        x1=sample_data.index[end_idx_rel],
        y0=0,
        y1=1,
        yref="paper",
        fillcolor="lightblue",
        opacity=0.2,
        line_width=0
    )
    
    label_names = {1: "Positif (1)", -1: "Negatif (-1)", 0: "Netral (0)"}
    
    fig.update_layout(
        title=f"Triple Barrier Example - Label: {label_names.get(label_value)} (Barrier: {random_sample['barrier_touched']})",
        height=500,
        width=1000,
        template="plotly_white"
    )
    
    fig.update_xaxes(
        rangeslider_visible=False
    )
    
    return fig

def generate_triple_barrier_visualizations(data, triple_barrier_df, 
                                         output_dir='logs/visualization',
                                         window_size=50, 
                                         save_html=True,
                                         save_png=True,
                                         verbose=True):
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    saved_files = {}
    labels_to_visualize = [1, -1, 0]
    label_names = {1: "positif", -1: "negatif", 0: "netral"}
    
    if verbose:
        print("\n=== Generating Triple Barrier Visualizations ===")
    
    for label_value in labels_to_visualize:
        try:
            fig = visualize_triple_barrier_sample(data, triple_barrier_df, 
                                                 label_value=label_value, 
                                                 window_size=window_size)
            
            if fig:
                label_name = label_names[label_value]
                base_filename = f"visualisasi_{timestamp}_label_{label_name}"
                
                if save_html:
                    html_path = os.path.join(output_dir, f"{base_filename}.html")
                    fig.write_html(html_path)
                    saved_files[f'label_{label_value}_html'] = html_path
                    if verbose:
                        print(f"Saved HTML: {html_path}")
                
                if save_png:
                    try:
                        png_path = os.path.join(output_dir, f"{base_filename}.png")
                        fig.write_image(png_path, width=1200, height=600)
                        saved_files[f'label_{label_value}_png'] = png_path
                        if verbose:
                            print(f"Saved PNG: {png_path}")
                    except Exception as png_error:
                        if verbose:
                            print(f"Warning: Could not save PNG for label {label_value}: {png_error}")
                            print("Note: PNG export requires 'kaleido' package: pip install kaleido")
            
        except Exception as e:
            if verbose:
                print(f"Error saat memvisualisasikan label {label_value}: {e}")
                if not triple_barrier_df.empty:
                    print(f"Kolom yang tersedia dalam triple_barrier_df: {triple_barrier_df.columns.tolist()}")
    
        
        
    
    
    
    return saved_files