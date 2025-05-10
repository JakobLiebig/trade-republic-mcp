import pandas as pd
from fuzzywuzzy import fuzz
import Levenshtein

from src.trade_republic.core.server import mcp

df = pd.read_parquet("data/company_name.pq")


def fuzzy_search_column(df, column_name, search_term, threshold=70):
    """
    Search for a term in a specific column using fuzzy matching
    
    Parameters:
    - df: pandas DataFrame
    - column_name: name of the column to search in
    - search_term: term to search for
    - threshold: minimum similarity score (0-100)
    
    Returns:
    - DataFrame with matching rows and their scores
    """
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Calculate similarity scores for each value in the column
    result_df['match_score'] = result_df[column_name].apply(
        lambda x: fuzz.ratio(str(x).lower(), search_term.lower())
    )
    
    # Filter rows that meet the threshold
    result_df = result_df[result_df['match_score'] >= threshold]
    
    # Sort by match score (highest first)
    result_df = result_df.sort_values('match_score', ascending=False)
    
    return result_df

@mcp.tool()
def isin_by_company_name(search_term: str) -> str:
    """
    Fuzzy search to find a ISIN by company name
    If you need a ISIN for a company, to query the trading data for example, use this tool!
    """
    result = fuzzy_search_column(df, "company_name", search_term)

    response = ""
    for index, row in result.iterrows():
        response += f"{row['company_name']} ({row['isin']})\n"

    return response

@mcp.resource("company_name://{isin}")
def company_name_by_isin(isin: str) -> str:
    """
    Returns the company name fo a given ISIN
    """
    result = df[df["isin"] == isin]

    if len(result) == 0:
        return "No company found for this ISIN"

    return result["company_name"].values[0]


if __name__ == "__main__":
    print(isin_by_company_name("Apple"))
    print(company_name_by_isin("XS1269176191"))