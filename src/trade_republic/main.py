from src.trade_republic.core.server import mcp
from src.trade_republic.features.query_data import (
    load_banking_data,
    load_trading_data,
    get_all_banking_data,
    get_all_trading_data,
    get_banking_data,
    get_trading_data,
    get_user_activity,
    calculate_account_balance,
    get_largest_transactions,
    summarize_trading_by_isin,
    group_transactions_by_type,
    get_monthly_summary
)

import src.trade_republic.features.test

if __name__ == "__main__":
    mcp.run(transport="sse")