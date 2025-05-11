from server import mcp
from features.query_data import (
    get_current_price,
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
    get_monthly_summary,
)
from features.lookups import (
    isin_by_company_name,
    company_name_by_isin,
)

if __name__ == "__main__":
    mcp.run(transport="sse")

