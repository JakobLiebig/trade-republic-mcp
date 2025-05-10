"""
This script creates simplified versions of all tables,

It merges the mcc codes with the banking data to have nl descriptions instead of mcc codes.
It merges the company names with the trading data.
It filters for the user id.
It drops the userId column.
It saves the data to csv files.
"""
import pandas as pd

banking_df = pd.read_csv("./data/banking_sample_data.csv")
trading_df = pd.read_csv("./data/trading_sample_data.csv")
mcc_df = pd.read_csv("./data/mcc_codes.csv")
company_df = pd.read_parquet("./data/company_name.pq")

# first merge mcc and banking to have nl description instead of mcc
banking_df = banking_df.merge(mcc_df, on="mcc", how="left")
# then filter for user id and delete column
banking_df = banking_df[banking_df["userId"] == "18d50f45-6812-4d09-8ea4-57e0cd270646"]
banking_df = banking_df.drop(columns=["userId"])

# first merge company name and trading data
trading_df = trading_df.rename(columns={"ISIN": "isin"})
trading_df = trading_df.merge(company_df, on="isin", how="left")
# then filter for user id
trading_df = trading_df[trading_df["userId"] == "00909ba7-ad01-42f1-9074-2773c7d3cf2c"]
trading_df = trading_df.drop(columns=["userId"])

# save to csv
banking_df.to_csv("./data/banking_simple.csv", index=False)
trading_df.to_csv("./data/trading_simple.csv", index=False)