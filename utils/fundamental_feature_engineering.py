import numpy as np
import pandas as pd
from scipy import stats
import logging

def current_ratio(df):
    return df['AssetsCurrent'] / df['LiabilitiesCurrent']

def quick_ratio(df):
    return (df['AssetsCurrent'] - df.get('InventoryNet', 0)) / df['LiabilitiesCurrent']

def cash_ratio(df):
    return df['CashAndCashEquivalentsAtCarryingValue'] / df['LiabilitiesCurrent']

def working_capital(df):
    return df['AssetsCurrent'] - df['LiabilitiesCurrent']

def accounts_receivable_turnover(df):
    return df['Revenues'] / df['AccountsReceivableNetCurrent']

def days_sales_outstanding(df):
    return (df['AccountsReceivableNetCurrent'] / df['Revenues']) * 365

def inventory_turnover(df):
    return df['CostOfRevenue'] / df.get('InventoryNet', 1)

def days_inventory_outstanding(df):
    return (df.get('InventoryNet', 0) / df['CostOfRevenue']) * 365

def accounts_payable_turnover(df):
    return df['CostOfRevenue'] / df['AccountsPayableCurrent']

def days_payable_outstanding(df):
    return (df['AccountsPayableCurrent'] / df['CostOfRevenue']) * 365

def cash_conversion_cycle(df):
    dso = days_sales_outstanding(df)
    dio = days_inventory_outstanding(df)
    dpo = days_payable_outstanding(df)
    return dso + dio - dpo

def gross_profit_margin(df):
    return (df.get('GrossProfit', df['Revenues'] - df['CostOfRevenue']) / df['Revenues']) * 100

def operating_profit_margin(df):
    return (df['OperatingIncomeLoss'] / df['Revenues']) * 100

def net_profit_margin(df):
    return (df['NetIncomeLoss'] / df['Revenues']) * 100

def return_on_assets(df):
    return (df['NetIncomeLoss'] / df['Assets']) * 100

def return_on_equity(df):
    return (df['NetIncomeLoss'] / df['StockholdersEquity']) * 100

def debt_to_equity_ratio(df):
    return df['Liabilities'] / df['StockholdersEquity']

def debt_to_assets_ratio(df):
    return df['Liabilities'] / df['Assets']

def interest_coverage_ratio(df):
    return df['OperatingIncomeLoss'] / df.get('InvestmentIncomeInterest', 1)

def operating_expense_ratio(df):
    return (df.get('OperatingExpenses', 0) / df['Revenues']) * 100

def rd_to_revenue_ratio(df):
    return (df.get('ResearchAndDevelopmentExpense', 0) / df['Revenues']) * 100

def sga_to_revenue_ratio(df):
    return (df.get('SellingGeneralAndAdministrativeExpense', 0) / df['Revenues']) * 100

def fixed_asset_turnover(df):
    return df['Revenues'] / df['PropertyPlantAndEquipmentNet']

def total_asset_turnover(df):
    return df['Revenues'] / df['Assets']

def capital_expenditure_ratio(df):
    return df.get('PaymentsToAcquirePropertyPlantAndEquipment', 0) / df['NetIncomeLoss']

def compensation_efficiency(df):
    return (df.get('AllocatedShareBasedCompensationExpense', 0) / df['Revenues']) * 100

def warranty_reserve_ratio(df):
    return (df.get('StandardProductWarrantyAccrual', 0) / df['Revenues']) * 100

def depreciation_rate(df):
    return (df.get('AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment', 0) / df.get('PropertyPlantAndEquipmentGross', 1)) * 100

def operating_cash_flow_to_net_income_ratio(df):
    cash_flow = df.get('IncreaseDecreaseInAccountsReceivable', 0) + df.get('IncreaseDecreaseInPrepaidDeferredExpenseAndOtherAssets', 0)
    return cash_flow / df['NetIncomeLoss']

def effective_tax_rate(df):
    return (df.get('IncomeTaxExpenseBenefit', 0) / df.get('IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest', 1)) * 100

def revenue_growth_rate(df):
    return df['Revenues'].pct_change() * 100

def net_income_growth_rate(df):
    return df['NetIncomeLoss'].pct_change() * 100

def asset_growth_rate(df):
    return df['Assets'].pct_change() * 100

def accrual_ratio(df):
    cash_flow = df.get('IncreaseDecreaseInAccountsReceivable', 0) + df.get('IncreaseDecreaseInPrepaidDeferredExpenseAndOtherAssets', 0)
    return (df['NetIncomeLoss'] - cash_flow) / df['Assets']

def cash_flow_to_revenue_ratio(df):
    cash_flow = df.get('IncreaseDecreaseInAccountsReceivable', 0) + df.get('IncreaseDecreaseInPrepaidDeferredExpenseAndOtherAssets', 0)
    return cash_flow / df['Revenues']

def qoq_growth(df, column):
    return df[column].pct_change() * 100

def yoy_quarterly_growth(df, column):
    return df[column].pct_change(periods=4) * 100

def trailing_twelve_months(df, column):
    return df[column].rolling(window=4).sum()

def quarterly_acceleration(df, column):
    qoq = qoq_growth(df, column)
    return qoq.diff()

def seasonal_index(df, column):
    avg_4q = df[column].rolling(window=4).mean()
    return df[column] / avg_4q

def seasonal_growth_rate(df, column):
    return yoy_quarterly_growth(df, column)

def moving_average_4q(df, column):
    return df[column].rolling(window=4).mean()

def quarter_run_rate(df, column):
    return df[column] * 4

def quarterly_operating_leverage(df):
    revenue_growth = qoq_growth(df, 'Revenues')
    operating_growth = qoq_growth(df, 'OperatingIncomeLoss')
    return operating_growth / revenue_growth

def quarterly_cash_burn_rate(df):
    cash_change = df['CashAndCashEquivalentsAtCarryingValue'].diff()
    return -cash_change / 3

def ytd_performance(df, column):
    return df[column].expanding().sum()

def quarterly_volatility(df, column):
    qoq = qoq_growth(df, column)
    return qoq.rolling(window=4).std()

def seasonal_dependency_index(df, column):
    quarterly_avg = df[column].rolling(window=4).mean()
    max_var = df[column].rolling(window=4).max() - df[column].rolling(window=4).min()
    return (max_var / quarterly_avg) * 100

def return_on_invested_capital(df):
    tax_rate = 0.25
    nopat = df['OperatingIncomeLoss'] * (1 - tax_rate)
    invested_capital = df['Assets'] - df['LiabilitiesCurrent']
    return (nopat / invested_capital) * 100

def cash_return_on_capital_invested(df):
    cash_flow = df.get('IncreaseDecreaseInAccountsReceivable', 0) + df.get('IncreaseDecreaseInPrepaidDeferredExpenseAndOtherAssets', 0)
    invested_capital = df['Assets'] - df['LiabilitiesCurrent']
    return cash_flow / invested_capital

def fixed_assets_to_long_term_debt_ratio(df):
    long_term_debt = df['Liabilities'] - df['LiabilitiesCurrent']
    return df['PropertyPlantAndEquipmentNet'] / long_term_debt

def non_current_asset_turnover(df):
    non_current_assets = df['Assets'] - df['AssetsCurrent']
    return df['Revenues'] / non_current_assets

def quarterly_gross_profit_stability(df):
    gross_margin = gross_profit_margin(df)
    return gross_margin.rolling(window=4).std()

def revenue_to_expense_growth_differential(df):
    revenue_growth = qoq_growth(df, 'Revenues')
    expense_growth = qoq_growth(df, 'OperatingExpenses') if 'OperatingExpenses' in df.columns else 0
    return revenue_growth - expense_growth

def quarterly_cash_flow_quality(df):
    cash_flow = df.get('IncreaseDecreaseInAccountsReceivable', 0) + df.get('IncreaseDecreaseInPrepaidDeferredExpenseAndOtherAssets', 0)
    quality_ratio = cash_flow / df['NetIncomeLoss']
    return quality_ratio.rolling(window=4).std()

def dividend_payout_ratio(df):
    dividends = df.get('Dividends_Declared', 0)
    return (dividends / df['NetIncomeLoss']) * 100

def stock_based_compensation_to_operating_expense_ratio(df):
    sbc = df.get('ShareBasedCompensation', 0)
    operating_expenses = df.get('OperatingExpenses', 1)
    return (sbc / operating_expenses) * 100

def long_term_revenue_cagr(df, periods=12):
    if len(df) < periods:
        return np.nan
    current_revenue = df['Revenues'].iloc[-1]
    past_revenue = df['Revenues'].iloc[-periods]
    return ((current_revenue / past_revenue) ** (1/periods) - 1) * 100

def operating_margin_trend(df):
    operating_margin = operating_profit_margin(df)
    x = np.arange(len(operating_margin))
    slope, _, _, _, _ = stats.linregress(x, operating_margin)
    return slope

def financial_leverage_index(df):
    leverage_ratio = debt_to_assets_ratio(df)
    return leverage_ratio / leverage_ratio.shift(1)

def asset_coverage_ratio(df):
    tangible_assets = df['Assets'] - df['LiabilitiesCurrent']
    long_term_debt = df['Liabilities'] - df['LiabilitiesCurrent']
    return tangible_assets / long_term_debt

def quarterly_margin_expansion(df):
    net_margin = net_profit_margin(df)
    return net_margin.diff()

def asset_utilization_ratio(df):
    avg_assets = (df['Assets'] + df['Assets'].shift(1)) / 2
    return df['Revenues'] / avg_assets

def capacity_utilization_proxy(df):
    industry_capacity_factor = 0.8
    return df['CostOfRevenue'] / (df['PropertyPlantAndEquipmentNet'] * industry_capacity_factor)

def altman_z_score(df):
    working_capital = df['working_capital']
    retained_earnings = df.get('RetainedEarningsAccumulatedDeficit', 0)
    z_score = (1.2 * (working_capital / df['Assets']) + 
               1.4 * (retained_earnings / df['Assets']) + 
               3.3 * (df['OperatingIncomeLoss'] / df['Assets']) + 
               0.6 * (df['StockholdersEquity'] / df['Liabilities']) + 
               0.999 * (df['Revenues'] / df['Assets']))
    return z_score

def dupont_analysis_roe(df):
    net_margin = net_profit_margin(df) / 100
    asset_turnover = total_asset_turnover(df)
    equity_multiplier = df['Assets'] / df['StockholdersEquity']
    return net_margin * asset_turnover * equity_multiplier * 100

def economic_value_added(df):
    wacc = 0.10
    tax_rate = 0.25
    nopat = df['OperatingIncomeLoss'] * (1 - tax_rate)
    invested_capital = df['Assets'] - df['LiabilitiesCurrent']
    return nopat - (wacc * invested_capital)

def rd_efficiency_ratio(df):
    gross_profit = df.get('GrossProfit', df['Revenues'] - df['CostOfRevenue'])
    rd_expense = df.get('ResearchAndDevelopmentExpense', 1)
    return gross_profit / rd_expense

def innovation_investment_ratio(df):
    rd_expense = df.get('ResearchAndDevelopmentExpense', 0)
    capex = df.get('PaymentsToAcquirePropertyPlantAndEquipment', 0)
    return ((rd_expense + capex) / df['Revenues']) * 100

def revenue_momentum(df):
    if len(df) < 6:
        return np.nan
    q2_q1_change = df['Revenues'].iloc[-1] - df['Revenues'].iloc[-2]
    q1_q4_change = df['Revenues'].iloc[-2] - df['Revenues'].iloc[-5]
    return q2_q1_change / q1_q4_change if q1_q4_change != 0 else np.nan

def earnings_momentum(df):
    if len(df) < 6:
        return np.nan
    q2_q1_change = df['NetIncomeLoss'].iloc[-1] - df['NetIncomeLoss'].iloc[-2]
    q1_q4_change = df['NetIncomeLoss'].iloc[-2] - df['NetIncomeLoss'].iloc[-5]
    return q2_q1_change / q1_q4_change if q1_q4_change != 0 else np.nan

def quarterly_earnings_quality_index(df):
    operating_net_diff = (df['OperatingIncomeLoss'] - df['NetIncomeLoss']) / df['Revenues']
    return operating_net_diff / operating_net_diff.shift(1)

def non_operating_items_ratio(df):
    non_operating = df['NetIncomeLoss'] - df['OperatingIncomeLoss']
    return (non_operating / df['NetIncomeLoss']) * 100

def apply_feature_engineering(pivoted_df, ticker):

    complete_columns = []
    threshold = 0.7

    for col in pivoted_df.columns:
        completeness = pivoted_df[col].count() / len(pivoted_df)
        if completeness >= threshold:
            complete_columns.append(col)

    logging.info(f'Kolom dengan data lengkap (>= {threshold*100}%): {len(complete_columns)}')

    incomplete_columns = [col for col in pivoted_df.columns if col not in complete_columns]
    if incomplete_columns:
        logging.info(f'Kolom dengan data tidak lengkap (kurang dari {threshold*100}%): {len(incomplete_columns)}')

    filtered_df = pivoted_df[complete_columns].copy()

    filtered_df = filtered_df.fillna(method='ffill').fillna(method='bfill')

    logging.info(f'\nFiltered DataFrame shape: {filtered_df.shape}')
    logging.info(f'Data completeness setelah filtering: {(filtered_df.count().sum() / (len(filtered_df) * len(filtered_df.columns)) * 100):.2f}%')

    enhanced_df = filtered_df.copy()

    feature_functions = [
        ('current_ratio', current_ratio),
        ('quick_ratio', quick_ratio),
        ('cash_ratio', cash_ratio),
        ('working_capital', working_capital),
        ('accounts_receivable_turnover', accounts_receivable_turnover),
        ('days_sales_outstanding', days_sales_outstanding),
        ('inventory_turnover', inventory_turnover),
        ('days_inventory_outstanding', days_inventory_outstanding),
        ('accounts_payable_turnover', accounts_payable_turnover),
        ('days_payable_outstanding', days_payable_outstanding),
        ('cash_conversion_cycle', cash_conversion_cycle),
        ('gross_profit_margin', gross_profit_margin),
        ('operating_profit_margin', operating_profit_margin),
        ('net_profit_margin', net_profit_margin),
        ('return_on_assets', return_on_assets),
        ('return_on_equity', return_on_equity),
        ('debt_to_equity_ratio', debt_to_equity_ratio),
        ('debt_to_assets_ratio', debt_to_assets_ratio),
        ('interest_coverage_ratio', interest_coverage_ratio),
        ('operating_expense_ratio', operating_expense_ratio),
        ('rd_to_revenue_ratio', rd_to_revenue_ratio),
        ('sga_to_revenue_ratio', sga_to_revenue_ratio),
        ('fixed_asset_turnover', fixed_asset_turnover),
        ('total_asset_turnover', total_asset_turnover),
        ('capital_expenditure_ratio', capital_expenditure_ratio),
        ('compensation_efficiency', compensation_efficiency),
        ('warranty_reserve_ratio', warranty_reserve_ratio),
        ('depreciation_rate', depreciation_rate),
        ('operating_cash_flow_to_net_income_ratio', operating_cash_flow_to_net_income_ratio),
        ('effective_tax_rate', effective_tax_rate),
        ('revenue_growth_rate', revenue_growth_rate),
        ('net_income_growth_rate', net_income_growth_rate),
        ('asset_growth_rate', asset_growth_rate),
        ('accrual_ratio', accrual_ratio),
        ('cash_flow_to_revenue_ratio', cash_flow_to_revenue_ratio),
        ('return_on_invested_capital', return_on_invested_capital),
        ('cash_return_on_capital_invested', cash_return_on_capital_invested),
        ('fixed_assets_to_long_term_debt_ratio', fixed_assets_to_long_term_debt_ratio),
        ('non_current_asset_turnover', non_current_asset_turnover),
        ('quarterly_gross_profit_stability', quarterly_gross_profit_stability),
        ('revenue_to_expense_growth_differential', revenue_to_expense_growth_differential),
        ('quarterly_cash_flow_quality', quarterly_cash_flow_quality),
        ('dividend_payout_ratio', dividend_payout_ratio),
        ('stock_based_compensation_to_operating_expense_ratio', stock_based_compensation_to_operating_expense_ratio),
        ('financial_leverage_index', financial_leverage_index),
        ('asset_coverage_ratio', asset_coverage_ratio),
        ('quarterly_margin_expansion', quarterly_margin_expansion),
        ('asset_utilization_ratio', asset_utilization_ratio),
        ('capacity_utilization_proxy', capacity_utilization_proxy),
        ('altman_z_score', altman_z_score),
        ('dupont_analysis_roe', dupont_analysis_roe),
        ('economic_value_added', economic_value_added),
        ('rd_efficiency_ratio', rd_efficiency_ratio),
        ('innovation_investment_ratio', innovation_investment_ratio),
        ('revenue_momentum', revenue_momentum),
        ('earnings_momentum', earnings_momentum),
        ('quarterly_earnings_quality_index', quarterly_earnings_quality_index),
        ('non_operating_items_ratio', non_operating_items_ratio)
    ]

    column_based_functions = [
        ('revenues_qoq_growth', 'Revenues', qoq_growth),
        ('revenues_yoy_growth', 'Revenues', yoy_quarterly_growth),
        ('revenues_ttm', 'Revenues', trailing_twelve_months),
        ('revenues_acceleration', 'Revenues', quarterly_acceleration),
        ('revenues_seasonal_index', 'Revenues', seasonal_index),
        ('revenues_moving_avg', 'Revenues', moving_average_4q),
        ('revenues_run_rate', 'Revenues', quarter_run_rate),
        ('revenues_volatility', 'Revenues', quarterly_volatility),
        ('revenues_seasonal_dependency', 'Revenues', seasonal_dependency_index),
        ('net_income_qoq_growth', 'NetIncomeLoss', qoq_growth),
        ('net_income_yoy_growth', 'NetIncomeLoss', yoy_quarterly_growth),
        ('net_income_ttm', 'NetIncomeLoss', trailing_twelve_months),
        ('net_income_acceleration', 'NetIncomeLoss', quarterly_acceleration),
        ('net_income_seasonal_index', 'NetIncomeLoss', seasonal_index),
        ('net_income_moving_avg', 'NetIncomeLoss', moving_average_4q),
        ('net_income_run_rate', 'NetIncomeLoss', quarter_run_rate),
        ('net_income_volatility', 'NetIncomeLoss', quarterly_volatility),
        ('assets_qoq_growth', 'Assets', qoq_growth),
        ('assets_yoy_growth', 'Assets', yoy_quarterly_growth),
        ('assets_ttm', 'Assets', trailing_twelve_months),
        ('operating_income_qoq_growth', 'OperatingIncomeLoss', qoq_growth),
        ('operating_income_yoy_growth', 'OperatingIncomeLoss', yoy_quarterly_growth),
        ('operating_income_ttm', 'OperatingIncomeLoss', trailing_twelve_months),
        ('cash_qoq_growth', 'CashAndCashEquivalentsAtCarryingValue', qoq_growth),
        ('cash_yoy_growth', 'CashAndCashEquivalentsAtCarryingValue', yoy_quarterly_growth),
        ('equity_qoq_growth', 'StockholdersEquity', qoq_growth),
        ('equity_yoy_growth', 'StockholdersEquity', yoy_quarterly_growth),
        ('liabilities_qoq_growth', 'Liabilities', qoq_growth),
        ('liabilities_yoy_growth', 'Liabilities', yoy_quarterly_growth)
    ]


    for name, func in feature_functions:
        try:
            enhanced_df[name] = func(enhanced_df)
        except Exception as e:
            pass

    for name, column, func in column_based_functions:
        try:
            if column in enhanced_df.columns:
                enhanced_df[name] = func(enhanced_df, column)
            else:
                pass
        except Exception as e:
            pass

    try:
        enhanced_df['quarterly_operating_leverage'] = quarterly_operating_leverage(enhanced_df)
    except Exception as e:
        pass

    try:
        enhanced_df['quarterly_cash_burn_rate'] = quarterly_cash_burn_rate(enhanced_df)
    except Exception as e:
        pass

    ytd_columns = ['Revenues', 'NetIncomeLoss', 'OperatingIncomeLoss']
    for col in ytd_columns:
        try:
            if col in enhanced_df.columns:
                enhanced_df[f'{col.lower()}_ytd'] = ytd_performance(enhanced_df, col)
        except Exception as e:
            pass

    try:
        enhanced_df['long_term_revenue_cagr'] = long_term_revenue_cagr(enhanced_df)
    except Exception as e:
        pass

    try:
        enhanced_df['operating_margin_trend'] = operating_margin_trend(enhanced_df)
    except Exception as e:
        pass

    enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)

    logging.info(f'\nEnhanced DataFrame shape: {enhanced_df.shape}')
    logging.info(f'Jumlah fitur asli: {len(complete_columns)}')
    logging.info(f'Jumlah fitur baru: {enhanced_df.shape[1] - len(complete_columns)}')
    logging.info(f'Total fitur: {enhanced_df.shape[1]}')

    logging.info('\nStatistik Enhanced DataFrame:')
    logging.info(f'Data completeness: {(enhanced_df.count().sum() / (len(enhanced_df) * len(enhanced_df.columns)) * 100):.2f}%')

    incomplete_enhanced_columns = []
    for col in enhanced_df.columns:
        completeness = enhanced_df[col].count() / len(enhanced_df)
        if completeness < 1.0:
            incomplete_enhanced_columns.append(col)

    if incomplete_enhanced_columns:
        logging.info(f'Kolom dengan data tidak lengkap di Enhanced DataFrame: {len(incomplete_enhanced_columns)}')
        for col in incomplete_enhanced_columns:
            empty_quarters = enhanced_df[enhanced_df[col].isna()].index.tolist()
            if empty_quarters:
                filtered_empty_quarters = [q for q in empty_quarters if q > '2012-Q1']
                if filtered_empty_quarters:
                    logging.info(f"q >= '2012-Q1' '{col}' {filtered_empty_quarters}")
                else:
                    pass


    enhanced_filename = f'datasets/fundamental/{ticker}_enhanced_features.csv'
    enhanced_df.to_csv(enhanced_filename)
    logging.info(f'\nEnhanced DataFrame saved to: {enhanced_filename}')
    
    filtered_quarters = [q for q in enhanced_df.index if q > '2012-Q1']
    enhanced_df = enhanced_df.loc[filtered_quarters]
    logging.info(f'DataFrame setelah filter q > 2012-Q1: {enhanced_df.shape}')
    logging.info(f'Data completeness: {(enhanced_df.count().sum() / (len(enhanced_df) * len(enhanced_df.columns)) * 100):.2f}%')

    return enhanced_df
