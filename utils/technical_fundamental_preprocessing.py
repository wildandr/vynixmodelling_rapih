import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalFundamentalPreprocessor:
    
    def __init__(self):
        self.technical_data = None
        self.fundamental_data = None
        self.combined_data = None
        
    def prepare_technical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Preparing technical data")
            df_clean = df.copy()
            
            required_columns = ['time', 'open', 'high', 'low', 'close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df_clean.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            if 'datetime' not in df_clean.columns:
                df_clean['datetime'] = pd.to_datetime(df_clean['time'], unit='s')
            if 'date' not in df_clean.columns:
                df_clean['date'] = df_clean['datetime'].dt.date
            
            logger.info(f"Technical data prepared: {len(df_clean)} rows, {len(df_clean.columns)} columns")
            
            self.technical_data = df_clean
            return df_clean
            
        except Exception as e:
            logger.error(f"Error preparing technical data: {str(e)}")
            raise ValueError(f"Failed to prepare technical data: {str(e)}")
    
    def prepare_fundamental_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Preparing fundamental data")
            df_clean = df.copy()
            
            if not all(isinstance(idx, str) and '-Q' in str(idx) for idx in df_clean.index[:5]):
                logger.warning("Index might not be in quarter format (YYYY-QX)")
            
            logger.info(f"Fundamental data prepared: {len(df_clean)} quarters, {len(df_clean.columns)} features")
            
            self.fundamental_data = df_clean
            return df_clean
            
        except Exception as e:
            logger.error(f"Error preparing fundamental data: {str(e)}")
            raise ValueError(f"Failed to prepare fundamental data: {str(e)}")
    
    def quarter_to_daily_conversion(self, fundamental_df: pd.DataFrame, 
                                  start_date: str = '2012-04-01', 
                                  end_date: str = '2025-06-30') -> pd.DataFrame:
        try:
            logger.info("Converting quarterly data to daily format with quarter shift mapping")
            
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            quarter_mapping = {}
            for quarter in fundamental_df.index:
                try:
                    year, q = quarter.split('-Q')
                    year = int(year)
                    q = int(q)
                    
                    if q == 4:
                        next_year = year + 1
                        next_q = 1
                    else:
                        next_year = year
                        next_q = q + 1
                    
                    if next_q == 1:
                        period_start = pd.Timestamp(next_year, 1, 1)
                        period_end = pd.Timestamp(next_year, 3, 31)
                    elif next_q == 2:
                        period_start = pd.Timestamp(next_year, 4, 1)
                        period_end = pd.Timestamp(next_year, 6, 30)
                    elif next_q == 3:
                        period_start = pd.Timestamp(next_year, 7, 1)
                        period_end = pd.Timestamp(next_year, 9, 30)
                    else:
                        period_start = pd.Timestamp(next_year, 10, 1)
                        period_end = pd.Timestamp(next_year, 12, 31)
                    
                    quarter_mapping[quarter] = (period_start, period_end)
                    
                    logger.debug(f"Quarter {quarter} data will be used in period {period_start.date()} to {period_end.date()}")
                    
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipping invalid quarter format: {quarter}")
                    continue
            
            daily_data_dict = {}
            
            for col in fundamental_df.columns:
                daily_values = pd.Series(index=date_range, dtype=float)
                
                for quarter, (period_start, period_end) in quarter_mapping.items():
                    if quarter in fundamental_df.index:
                        usage_dates = date_range[(date_range >= period_start) & (date_range <= period_end)]
                        
                        if len(usage_dates) > 0:
                            daily_values.loc[usage_dates] = fundamental_df.loc[quarter, col]
                
                daily_data_dict[col] = daily_values
            
            daily_data = pd.DataFrame(daily_data_dict)
            
            daily_data = daily_data.ffill()
            
            non_null_count = daily_data.notna().sum().sum()
            total_count = len(daily_data) * len(daily_data.columns)
            availability_pct = (non_null_count / total_count) * 100 if total_count > 0 else 0
            
            logger.info(f"Daily conversion completed: {len(daily_data)} days, {len(daily_data.columns)} features")
            logger.info(f"Data availability with quarter shift mapping: {availability_pct:.2f}%")
            
            return daily_data
            
        except Exception as e:
            logger.error(f"Error in quarter to daily conversion: {str(e)}")
            raise ValueError(f"Failed to convert quarterly to daily data: {str(e)}")
    
    def combine_technical_fundamental(self, technical_df: pd.DataFrame, 
                                   fundamental_daily_df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Combining technical and fundamental data")
            
            if 'date' not in technical_df.columns:
                if 'datetime' in technical_df.columns:
                    technical_df['date'] = technical_df['datetime'].dt.date
                elif 'time' in technical_df.columns:
                    technical_df['datetime'] = pd.to_datetime(technical_df['time'], unit='s')
                    technical_df['date'] = technical_df['datetime'].dt.date
                else:
                    raise ValueError("No date/time column found in technical data")
            
            fundamental_daily_df = fundamental_daily_df.copy()
            fundamental_daily_df['date'] = fundamental_daily_df.index.date
            
            combined_df = pd.merge(technical_df, fundamental_daily_df, 
                                 on='date', how='inner', suffixes=('_tech', '_fund'))
            
            if 'date_fund' in combined_df.columns:
                combined_df = combined_df.drop('date_fund', axis=1)
            
            logger.info(f"Data combination completed: {len(combined_df)} rows, {len(combined_df.columns)} columns")
            logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
            
            self.combined_data = combined_df
            return combined_df
            
        except Exception as e:
            logger.error(f"Error combining data: {str(e)}")
            raise ValueError(f"Failed to combine technical and fundamental data: {str(e)}")
    
    def filter_by_period(self, df: pd.DataFrame, 
                        start_period: str = '2012-Q2', 
                        end_period: str = '2025-Q2') -> pd.DataFrame:
        try:
            logger.info(f"Filtering data from {start_period} to {end_period}")
            
            def period_to_date(period: str, is_start: bool = True) -> pd.Timestamp:
                year, q = period.split('-Q')
                year = int(year)
                q = int(q)
                
                if q == 1:
                    return pd.Timestamp(year, 1, 1) if is_start else pd.Timestamp(year, 3, 31)
                elif q == 2:
                    return pd.Timestamp(year, 4, 1) if is_start else pd.Timestamp(year, 6, 30)
                elif q == 3:
                    return pd.Timestamp(year, 7, 1) if is_start else pd.Timestamp(year, 9, 30)
                else:
                    return pd.Timestamp(year, 10, 1) if is_start else pd.Timestamp(year, 12, 31)
            
            start_date = period_to_date(start_period, True)
            end_date = period_to_date(end_period, False)
            
            if 'date' in df.columns:
                df_filtered = df.copy()
                df_filtered['date'] = pd.to_datetime(df_filtered['date'])
                
                mask = (df_filtered['date'] >= start_date) & (df_filtered['date'] <= end_date)
                df_filtered = df_filtered[mask]
                
            elif 'datetime' in df.columns:
                df_filtered = df.copy()
                mask = (df_filtered['datetime'] >= start_date) & (df_filtered['datetime'] <= end_date)
                df_filtered = df_filtered[mask]
                
            else:
                raise ValueError("No date or datetime column found for filtering")
            
            logger.info(f"Filtering completed: {len(df_filtered)} rows remaining")
            
            return df_filtered
            
        except Exception as e:
            logger.error(f"Error filtering data: {str(e)}")
            raise ValueError(f"Failed to filter data by period: {str(e)}")
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> None:
        try:
            logger.info(f"Saving processed data to: {output_path}")
            
            df.to_csv(output_path, index=False)
            
            logger.info(f"Data saved successfully: {len(df)} rows, {len(df.columns)} columns")
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise IOError(f"Failed to save processed data: {str(e)}")
    
    def process_complete_pipeline(self, 
                                technical_df: pd.DataFrame,
                                fundamental_df: pd.DataFrame,
                                output_path: str = None,
                                start_period: str = '2012-Q3',
                                end_period: str = '2025-Q2') -> pd.DataFrame:
        try:
            logger.info("Starting complete preprocessing pipeline with quarter shift mapping")
            
            technical_prepared = self.prepare_technical_data(technical_df)
            fundamental_prepared = self.prepare_fundamental_data(fundamental_df)
            
            start_date = '2012-04-01' if start_period == '2012-Q2' else '2012-01-01'
            end_date = '2025-06-30' if end_period == '2025-Q2' else '2025-12-31'
            
            fundamental_daily = self.quarter_to_daily_conversion(
                fundamental_prepared, start_date, end_date
            )
            
            combined_df = self.combine_technical_fundamental(technical_prepared, fundamental_daily)
            
            filtered_df = self.filter_by_period(combined_df, start_period, end_period)
            
            if output_path:
                self.save_processed_data(filtered_df, output_path)
            
            logger.info("Complete preprocessing pipeline finished successfully")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def preprocess_technical_fundamental_data(technical_df: pd.DataFrame,
                                        fundamental_df: pd.DataFrame,
                                        output_path: str = None,
                                        start_period: str = '2012-Q3',
                                        end_period: str = '2025-Q2') -> pd.DataFrame:
    preprocessor = TechnicalFundamentalPreprocessor()
    return preprocessor.process_complete_pipeline(
        technical_df, fundamental_df, output_path, start_period, end_period
    )


def convert_quarterly_to_daily(fundamental_df: pd.DataFrame,
                             start_date: str = '2012-04-01',
                             end_date: str = '2025-06-30') -> pd.DataFrame:
    preprocessor = TechnicalFundamentalPreprocessor()
    return preprocessor.quarter_to_daily_conversion(fundamental_df, start_date, end_date)


def combine_data(technical_df: pd.DataFrame, 
               fundamental_daily_df: pd.DataFrame) -> pd.DataFrame:
    preprocessor = TechnicalFundamentalPreprocessor()
    return preprocessor.combine_technical_fundamental(technical_df, fundamental_daily_df)


def filter_data_by_period(df: pd.DataFrame,
                         start_period: str = '2012-Q2',
                         end_period: str = '2025-Q2') -> pd.DataFrame:
    preprocessor = TechnicalFundamentalPreprocessor()
    return preprocessor.filter_by_period(df, start_period, end_period)


if __name__ == "__main__":
    try:
        technical_df = pd.read_csv("datasets/technical/TSLA_original.csv")
        fundamental_df = pd.read_csv("datasets/fundamental/TSLA_enhanced_features.csv", index_col=0)
        
        result_df = preprocess_technical_fundamental_data(
            technical_df, fundamental_df
        )
        print(f"Preprocessing completed successfully!")
        print(f"Result shape: {result_df.shape}")
        print(f"Date range: {result_df['date'].min()} to {result_df['date'].max()}")
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")