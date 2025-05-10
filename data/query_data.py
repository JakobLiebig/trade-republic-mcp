"""
Trade Republic Data Query API

This module provides functions to load and query banking and trading data from CSV files.
It supports filtering data by various parameters and includes helper functions for 
common analyses.
"""

import os
import pandas as pd
from datetime import datetime, date
from typing import Optional, List, Dict, Union, Tuple


# Data loading and caching
_banking_df = None  
_trading_df = None


def load_banking_data() -> pd.DataFrame:
    """
    Load banking data from CSV file and properly format columns.
    
    Returns:
        pd.DataFrame: Banking data with properly formatted columns
    """
    global _banking_df
    
    if _banking_df is not None:
        return _banking_df
    
    file_path = os.path.join('data', 'banking_sample_data.csv')
    _banking_df = pd.read_csv(file_path)
    
    # Convert columns to appropriate data types
    _banking_df['bookingDate'] = pd.to_datetime(_banking_df['bookingDate']).dt.date
    _banking_df['amount'] = pd.to_numeric(_banking_df['amount'])
    
    # Ensure mcc is properly handled (it can be NaN)
    _banking_df['mcc'] = _banking_df['mcc'].astype('Int64', errors='ignore')
    
    return _banking_df


def load_trading_data() -> pd.DataFrame:
    """
    Load trading data from CSV file and properly format columns.
    
    Returns:
        pd.DataFrame: Trading data with properly formatted columns
    """
    global _trading_df
    
    if _trading_df is not None:
        return _trading_df
    
    file_path = os.path.join('data', 'trading_sample_data.csv')
    _trading_df = pd.read_csv(file_path)
    
    # Convert columns to appropriate data types
    _trading_df['executedAt'] = pd.to_datetime(_trading_df['executedAt'])
    _trading_df['executionSize'] = pd.to_numeric(_trading_df['executionSize'])
    _trading_df['executionPrice'] = pd.to_numeric(_trading_df['executionPrice'])
    _trading_df['executionFee'] = pd.to_numeric(_trading_df['executionFee'])
    
    # Calculate total transaction amount (including fee)
    _trading_df['totalAmount'] = (_trading_df['executionSize'] * _trading_df['executionPrice']) + _trading_df['executionFee']
    
    return _trading_df


# Main query functions
def get_banking_data(
    user_id: Optional[str] = None,
    start_date: Optional[Union[str, date]] = None,
    end_date: Optional[Union[str, date]] = None,
    side: Optional[str] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    transaction_type: Optional[Union[str, List[str]]] = None,
    currency: Optional[str] = None,
    mcc: Optional[Union[int, List[int]]] = None
) -> pd.DataFrame:
    """
    Query banking data with optional filters.
    
    Args:
        user_id: Filter by specific user ID
        start_date: Filter transactions on or after this date (YYYY-MM-DD)
        end_date: Filter transactions on or before this date (YYYY-MM-DD)
        side: Filter by transaction side (CREDIT/DEBIT)
        min_amount: Filter transactions with amount >= min_amount
        max_amount: Filter transactions with amount <= max_amount
        transaction_type: Filter by transaction type(s) (PAYIN, TRADING, EARNINGS, etc.)
        currency: Filter by currency (e.g., EUR)
        mcc: Filter by MCC code(s) for card transactions
        
    Returns:
        pd.DataFrame: Filtered banking data
    """
    df = load_banking_data()
    
    # Convert date strings to date objects if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).date()
    
    # Apply filters
    if user_id:
        df = df[df['userId'] == user_id]
    
    if start_date:
        df = df[df['bookingDate'] >= start_date]
        
    if end_date:
        df = df[df['bookingDate'] <= end_date]
        
    if side:
        df = df[df['side'] == side]
        
    if min_amount is not None:
        df = df[df['amount'] >= min_amount]
        
    if max_amount is not None:
        df = df[df['amount'] <= max_amount]
        
    if transaction_type:
        if isinstance(transaction_type, list):
            df = df[df['type'].isin(transaction_type)]
        else:
            df = df[df['type'] == transaction_type]
            
    if currency:
        df = df[df['currency'] == currency]
        
    if mcc is not None:
        if isinstance(mcc, list):
            df = df[df['mcc'].isin(mcc)]
        else:
            df = df[df['mcc'] == mcc]
            
    return df


def get_trading_data(
    user_id: Optional[str] = None,
    start_datetime: Optional[Union[str, datetime]] = None,
    end_datetime: Optional[Union[str, datetime]] = None,
    isin: Optional[Union[str, List[str]]] = None,
    direction: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
    min_total: Optional[float] = None,
    max_total: Optional[float] = None,
    transaction_type: Optional[Union[str, List[str]]] = None,
    currency: Optional[str] = None
) -> pd.DataFrame:
    """
    Query trading data with optional filters.
    
    Args:
        user_id: Filter by specific user ID
        start_datetime: Filter transactions on or after this datetime
        end_datetime: Filter transactions on or before this datetime
        isin: Filter by ISIN(s)
        direction: Filter by trade direction (BUY/SELL)
        min_price: Filter by minimum execution price
        max_price: Filter by maximum execution price
        min_size: Filter by minimum execution size
        max_size: Filter by maximum execution size
        min_total: Filter by minimum total transaction amount
        max_total: Filter by maximum total transaction amount
        transaction_type: Filter by transaction type(s) (REGULAR, BONUS, SAVINGSPLAN)
        currency: Filter by currency
        
    Returns:
        pd.DataFrame: Filtered trading data
    """
    df = load_trading_data()
    
    # Convert datetime strings to datetime objects if needed
    if isinstance(start_datetime, str):
        start_datetime = pd.to_datetime(start_datetime)
    if isinstance(end_datetime, str):
        end_datetime = pd.to_datetime(end_datetime)
    
    # Apply filters
    if user_id:
        df = df[df['userId'] == user_id]
    
    if start_datetime:
        df = df[df['executedAt'] >= start_datetime]
        
    if end_datetime:
        df = df[df['executedAt'] <= end_datetime]
        
    if isin:
        if isinstance(isin, list):
            df = df[df['ISIN'].isin(isin)]
        else:
            df = df[df['ISIN'] == isin]
        
    if direction:
        df = df[df['direction'] == direction]
        
    if min_price is not None:
        df = df[df['executionPrice'] >= min_price]
        
    if max_price is not None:
        df = df[df['executionPrice'] <= max_price]
        
    if min_size is not None:
        df = df[df['executionSize'] >= min_size]
        
    if max_size is not None:
        df = df[df['executionSize'] <= max_size]
        
    if min_total is not None:
        df = df[df['totalAmount'] >= min_total]
        
    if max_total is not None:
        df = df[df['totalAmount'] <= max_total]
        
    if transaction_type:
        if isinstance(transaction_type, list):
            df = df[df['type'].isin(transaction_type)]
        else:
            df = df[df['type'] == transaction_type]
            
    if currency:
        df = df[df['currency'] == currency]
            
    return df


def get_user_activity(
    user_id: str,
    start_date: Optional[Union[str, date]] = None,
    end_date: Optional[Union[str, date]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Get combined banking and trading activity for a specific user.
    
    Args:
        user_id: User ID to query
        start_date: Filter transactions on or after this date
        end_date: Filter transactions on or before this date
        
    Returns:
        Dict with keys 'banking' and 'trading' containing the respective data frames
    """
    # Get banking data
    banking_data = get_banking_data(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date
    )
    
    # Convert date strings to datetime objects for trading data query
    start_datetime = None
    if start_date:
        if isinstance(start_date, str):
            start_datetime = pd.to_datetime(start_date)
        else:
            start_datetime = pd.to_datetime(start_date)
    
    end_datetime = None
    if end_date:
        if isinstance(end_date, str):
            end_datetime = pd.to_datetime(end_date)
            # Set to end of day
            end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
        else:
            # Convert date to datetime at end of day
            end_datetime = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)
    
    # Get trading data
    trading_data = get_trading_data(
        user_id=user_id,
        start_datetime=start_datetime,
        end_datetime=end_datetime
    )
    
    return {
        'banking': banking_data,
        'trading': trading_data
    }


# Helper functions for common analyses
def calculate_account_balance(
    user_id: str,
    start_date: Optional[Union[str, date]] = None,
    end_date: Optional[Union[str, date]] = None
) -> pd.DataFrame:
    """
    Calculate account balance over time for a specific user.
    
    Args:
        user_id: User ID to calculate balance for
        start_date: Start date for calculation
        end_date: End date for calculation
        
    Returns:
        pd.DataFrame with date and balance columns
    """
    banking_data = get_banking_data(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date
    )
    
    if banking_data.empty:
        return pd.DataFrame(columns=['date', 'balance'])
    
    # Create a new DataFrame for balance calculation
    balance_df = banking_data.copy()
    
    # Convert amounts: CREDIT is positive, DEBIT is negative
    balance_df['signed_amount'] = balance_df.apply(
        lambda row: row['amount'] if row['side'] == 'CREDIT' else -row['amount'],
        axis=1
    )
    
    # Sort by date
    balance_df = balance_df.sort_values('bookingDate')
    
    # Calculate cumulative balance
    balance_df['balance'] = balance_df['signed_amount'].cumsum()
    
    # Select only relevant columns
    result = balance_df[['bookingDate', 'balance']].copy()
    result.rename(columns={'bookingDate': 'date'}, inplace=True)
    
    return result


def get_largest_transactions(
    user_id: Optional[str] = None,
    transaction_type: Optional[str] = None,
    start_date: Optional[Union[str, date]] = None,
    end_date: Optional[Union[str, date]] = None,
    n: int = 10
) -> pd.DataFrame:
    """
    Get the n largest banking transactions.
    
    Args:
        user_id: Optional user ID filter
        transaction_type: Optional transaction type filter
        start_date: Optional start date filter
        end_date: Optional end date filter
        n: Number of transactions to return (default 10)
        
    Returns:
        pd.DataFrame with the largest transactions by amount
    """
    banking_data = get_banking_data(
        user_id=user_id,
        transaction_type=transaction_type,
        start_date=start_date,
        end_date=end_date
    )
    
    if banking_data.empty:
        return pd.DataFrame()
    
    # Sort by amount in descending order and take the top n
    largest = banking_data.sort_values('amount', ascending=False).head(n)
    
    return largest


def summarize_trading_by_isin(
    user_id: Optional[str] = None,
    start_date: Optional[Union[str, date]] = None,
    end_date: Optional[Union[str, date]] = None
) -> pd.DataFrame:
    """
    Summarize trading activity by security (ISIN).
    
    Args:
        user_id: Optional user ID filter
        start_date: Optional start date filter
        end_date: Optional end date filter
        
    Returns:
        pd.DataFrame with summary statistics by ISIN
    """
    # Convert date to datetime for trading data query
    start_datetime = None
    if start_date:
        if isinstance(start_date, str):
            start_datetime = pd.to_datetime(start_date)
        else:
            start_datetime = pd.to_datetime(start_date)
    
    end_datetime = None
    if end_date:
        if isinstance(end_date, str):
            end_datetime = pd.to_datetime(end_date)
            # Set to end of day
            end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
        else:
            # Convert date to datetime at end of day
            end_datetime = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)
    
    trading_data = get_trading_data(
        user_id=user_id,
        start_datetime=start_datetime,
        end_datetime=end_datetime
    )
    
    if trading_data.empty:
        return pd.DataFrame()
    
    # Group by ISIN and calculate summary statistics
    summary = trading_data.groupby('ISIN').agg({
        'executionSize': 'sum',
        'totalAmount': 'sum',
        'executedAt': ['count', 'min', 'max'],
        'direction': lambda x: (x == 'BUY').sum(),  # Count of BUY orders
    })
    
    # Flatten multi-level columns
    summary.columns = ['total_size', 'total_amount', 'trade_count', 'first_trade', 'last_trade', 'buy_count']
    summary['sell_count'] = summary['trade_count'] - summary['buy_count']
    
    # Add net position
    buy_positions = trading_data[trading_data['direction'] == 'BUY'].groupby('ISIN')['executionSize'].sum()
    sell_positions = trading_data[trading_data['direction'] == 'SELL'].groupby('ISIN')['executionSize'].sum()
    
    # Fill NaN values with 0 for ISINs that don't have both buys and sells
    buy_positions = buy_positions.fillna(0)
    sell_positions = sell_positions.fillna(0)
    
    # Calculate net position
    summary['net_position'] = buy_positions - sell_positions
    
    # Calculate average price
    summary['average_price'] = (summary['total_amount'] - summary['trade_count']) / summary['total_size']
    
    return summary.reset_index()


def group_transactions_by_type(
    user_id: Optional[str] = None,
    start_date: Optional[Union[str, date]] = None,
    end_date: Optional[Union[str, date]] = None
) -> pd.DataFrame:
    """
    Group banking transactions by type and calculate summary statistics.
    
    Args:
        user_id: Optional user ID filter
        start_date: Optional start date filter
        end_date: Optional end date filter
        
    Returns:
        pd.DataFrame with summary by transaction type
    """
    banking_data = get_banking_data(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date
    )
    
    if banking_data.empty:
        return pd.DataFrame()
    
    # Create separate DataFrames for CREDIT and DEBIT
    credit_data = banking_data[banking_data['side'] == 'CREDIT']
    debit_data = banking_data[banking_data['side'] == 'DEBIT']
    
    # Group by type and calculate statistics for CREDIT
    credit_summary = credit_data.groupby('type').agg({
        'amount': ['sum', 'mean', 'count'],
    })
    credit_summary.columns = ['credit_sum', 'credit_mean', 'credit_count']
    
    # Group by type and calculate statistics for DEBIT
    debit_summary = debit_data.groupby('type').agg({
        'amount': ['sum', 'mean', 'count'],
    })
    debit_summary.columns = ['debit_sum', 'debit_mean', 'debit_count']
    
    # Combine the summaries
    summary = pd.concat([credit_summary, debit_summary], axis=1).fillna(0)
    
    # Calculate net flow
    summary['net_flow'] = summary['credit_sum'] - summary['debit_sum']
    summary['total_count'] = summary['credit_count'] + summary['debit_count']
    
    return summary.reset_index()


def get_monthly_summary(
    user_id: Optional[str] = None,
    start_date: Optional[Union[str, date]] = None,
    end_date: Optional[Union[str, date]] = None
) -> pd.DataFrame:
    """
    Get monthly summary of banking transactions.
    
    Args:
        user_id: Optional user ID filter
        start_date: Optional start date filter
        end_date: Optional end date filter
        
    Returns:
        pd.DataFrame with monthly summary
    """
    banking_data = get_banking_data(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date
    )
    
    if banking_data.empty:
        return pd.DataFrame()
    
    # Convert bookingDate to period
    banking_data['year_month'] = pd.to_datetime(banking_data['bookingDate']).dt.to_period('M')
    
    # Create separate DataFrames for CREDIT and DEBIT
    credit_data = banking_data[banking_data['side'] == 'CREDIT']
    debit_data = banking_data[banking_data['side'] == 'DEBIT']
    
    # Group by month and calculate statistics for CREDIT
    credit_monthly = credit_data.groupby('year_month').agg({
        'amount': ['sum', 'count'],
    })
    credit_monthly.columns = ['credit_sum', 'credit_count']
    
    # Group by month and calculate statistics for DEBIT
    debit_monthly = debit_data.groupby('year_month').agg({
        'amount': ['sum', 'count'],
    })
    debit_monthly.columns = ['debit_sum', 'debit_count']
    
    # Combine the summaries
    monthly_summary = pd.concat([credit_monthly, debit_monthly], axis=1).fillna(0)
    
    # Calculate net flow
    monthly_summary['net_flow'] = monthly_summary['credit_sum'] - monthly_summary['debit_sum']
    monthly_summary['total_count'] = monthly_summary['credit_count'] + monthly_summary['debit_count']
    
    # Convert period index to datetime for easier handling
    monthly_summary = monthly_summary.reset_index()
    monthly_summary['year_month'] = monthly_summary['year_month'].dt.to_timestamp()
    
    return monthly_summary


if __name__ == "__main__":
    # Example usage
    print("Loading banking data...")
    banking_df = load_banking_data()
    print(f"Loaded {len(banking_df)} banking transactions")
    
    print("\nLoading trading data...")
    trading_df = load_trading_data()
    print(f"Loaded {len(trading_df)} trading transactions")
    
    # Example query: Get all transactions for a specific user
    user_id = "00909ba7-ad01-42f1-9074-2773c7d3cf2c"
    user_activity = get_user_activity(user_id)
    print(f"\nUser {user_id} has {len(user_activity['banking'])} banking transactions and " 
          f"{len(user_activity['trading'])} trading transactions")
    
    # Example: Calculate balance
    balance = calculate_account_balance(user_id)
    print(f"\nFinal balance for user {user_id}: {balance['balance'].iloc[-1]:.2f} EUR")
