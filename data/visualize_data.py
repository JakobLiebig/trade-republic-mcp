"""
Trade Republic Data Visualization

This module provides functions to create visualizations from the banking and trading 
data loaded by query_data.py.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
from datetime import datetime, timedelta

import data.query_data as qd

# Set default styling - switch to dark theme
plt.style.use('dark_background')
sns.set_palette('dark')

# Custom Trade Republic theme for plotly
tr_theme = {
    'layout': {
        'paper_bgcolor': '#121212',
        'plot_bgcolor': '#121212',
        'font': {'color': '#FFFFFF'},
        'title': {'font': {'color': '#FFFFFF', 'size': 20}},
        'legend': {'font': {'color': '#FFFFFF'}},
        'xaxis': {
            'gridcolor': '#333333',
            'zerolinecolor': '#333333',
            'title': {'font': {'color': '#FFFFFF'}}
        },
        'yaxis': {
            'gridcolor': '#333333',
            'zerolinecolor': '#333333',
            'title': {'font': {'color': '#FFFFFF'}}
        }
    }
}

# Trade Republic colors
TR_GREEN = '#00C805'
TR_RED = '#FF4B4B'
TR_GRAY = '#8A8A8A'


def visualize_account_balance(user_id: str, start_date: Optional[str] = None, 
                              end_date: Optional[str] = None, 
                              use_plotly: bool = True) -> Union[plt.Figure, go.Figure]:
    """
    Visualize account balance over time for a specific user.
    
    Args:
        user_id: User ID to visualize balance for
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        use_plotly: Whether to use plotly (interactive) or matplotlib
        
    Returns:
        Figure object with the visualization
    """
    balance_df = qd.calculate_account_balance(user_id, start_date, end_date)
    
    if balance_df.empty:
        raise ValueError(f"No balance data found for user {user_id}")
    
    # Calculate daily change in balance
    balance_df['daily_change'] = balance_df['balance'].diff()
    # Calculate percent change for display
    balance_start = balance_df['balance'].iloc[0]
    balance_end = balance_df['balance'].iloc[-1]
    pct_change = ((balance_end - balance_start) / balance_start) * 100 if balance_start != 0 else 0
    
    if use_plotly:
        # Create interactive plotly visualization
        fig = go.Figure()
        
        # Add main balance line
        fig.add_trace(
            go.Scatter(
                x=balance_df['date'], 
                y=balance_df['balance'],
                mode='lines',
                name='Balance',
                line=dict(color=TR_GREEN if pct_change >= 0 else TR_RED, width=2),
            )
        )
        
        # Add trendline - make it subtle
        fig.add_trace(
            go.Scatter(
                x=balance_df['date'],
                y=balance_df['balance'].rolling(window=7, min_periods=1).mean(),
                mode='lines',
                name='7-day Moving Average',
                line=dict(color='rgba(255,255,255,0.3)', width=1.5, dash='dash'),
                visible='legendonly'  # Hide by default, can be shown from legend
            )
        )
        
        # Apply Trade Republic theme first
        fig.update_layout(**tr_theme['layout'])
        
        # Then apply specific overrides
        fig.update_layout(
            title=None,  # Remove title for cleaner look
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        
        # Add custom header with portfolio value and percent change
        balance_text = f"€{balance_end:.2f}"
        pct_text = f"{pct_change:.2f}%" if pct_change >= 0 else f"{pct_change:.2f}%"
        
        fig.add_annotation(
            text=f"Portfolio<br>{balance_text}",
            xref="paper", yref="paper",
            x=0.01, y=0.99,
            showarrow=False,
            font=dict(color='white', size=22),
            align="left",
        )
        
        fig.add_annotation(
            text=pct_text,
            xref="paper", yref="paper",
            x=0.01, y=0.90,
            showarrow=False,
            font=dict(color=TR_GREEN if pct_change >= 0 else TR_RED, size=16),
            align="left",
        )
        
        # Add time period buttons
        fig.update_layout(
            updatemenus=[{
                'buttons': [
                    {'label': "1D", 'method': "relayout", 'args': [{"xaxis.range": [balance_df['date'].iloc[-1] - pd.Timedelta(days=1), balance_df['date'].iloc[-1]]}]},
                    {'label': "1W", 'method': "relayout", 'args': [{"xaxis.range": [balance_df['date'].iloc[-1] - pd.Timedelta(weeks=1), balance_df['date'].iloc[-1]]}]},
                    {'label': "1M", 'method': "relayout", 'args': [{"xaxis.range": [balance_df['date'].iloc[-1] - pd.Timedelta(days=30), balance_df['date'].iloc[-1]]}]},
                    {'label': "1Y", 'method': "relayout", 'args': [{"xaxis.range": [balance_df['date'].iloc[-1] - pd.Timedelta(days=365), balance_df['date'].iloc[-1]]}]},
                    {'label': "Max", 'method': "relayout", 'args': [{"xaxis.autorange": True}]},
                ],
                'direction': 'right',
                'pad': {'r': 10, 't': 10},
                'showactive': True,
                'type': 'buttons',
                'x': 0.15,
                'y': 1.05,
                'xanchor': 'left',
                'yanchor': 'top',
                'bgcolor': '#121212',
                'bordercolor': '#333333',
                'font': {'color': 'white'}
            }]
        )
        
        return fig
    
    else:
        # Create static matplotlib visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot balance
        ax.plot(balance_df['date'], balance_df['balance'], marker='o', 
                linestyle='-', linewidth=2, markersize=4, label='Daily Balance')
        
        # Add trendline
        ax.plot(balance_df['date'], balance_df['balance'].rolling(window=7, min_periods=1).mean(),
                'b--', linewidth=1.5, label='7-day Moving Average')
        
        # Highlight significant changes
        balance_changes = balance_df[balance_df['daily_change'].abs() > 
                                     balance_df['daily_change'].abs().quantile(0.9)]
        
        for idx, row in balance_changes.iterrows():
            color = 'red' if row['daily_change'] < 0 else 'green'
            ax.annotate(f"€{row['daily_change']:.2f}", 
                        xy=(row['date'], row['balance']),
                        xytext=(0, -30 if row['daily_change'] < 0 else 30),
                        textcoords="offset points",
                        ha='center', va='center',
                        arrowprops=dict(arrowstyle='->', color=color),
                        color=color)
        
        ax.set_title(f"Account Balance Over Time - User {user_id}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Balance (EUR)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig


def visualize_monthly_cashflow(user_id: Optional[str] = None, 
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              use_plotly: bool = True) -> Union[plt.Figure, go.Figure]:
    """
    Visualize monthly income vs expenses and net cash flow.
    
    Args:
        user_id: Optional user ID to filter data
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        use_plotly: Whether to use plotly (interactive) or matplotlib
        
    Returns:
        Figure object with the visualization
    """
    monthly_data = qd.get_monthly_summary(user_id, start_date, end_date)
    
    if monthly_data.empty:
        raise ValueError("No monthly data found for the specified parameters")
    
    # Format dates for display
    monthly_data['month_label'] = monthly_data['year_month'].dt.strftime('%b %Y')
    
    if use_plotly:
        # Create interactive plotly visualization
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bar traces for income and expenses
        fig.add_trace(
            go.Bar(
                x=monthly_data['month_label'],
                y=monthly_data['credit_sum'],
                name='Income',
                marker_color='green',
                opacity=0.7
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Bar(
                x=monthly_data['month_label'],
                y=monthly_data['debit_sum'],
                name='Expenses',
                marker_color='red',
                opacity=0.7
            ),
            secondary_y=False
        )
        
        # Add line trace for net flow
        fig.add_trace(
            go.Scatter(
                x=monthly_data['month_label'],
                y=monthly_data['net_flow'],
                name='Net Flow',
                line=dict(color='blue', width=3),
                mode='lines+markers'
            ),
            secondary_y=True
        )
        
        # Update layout
        title = "Monthly Cashflow"
        if user_id:
            title += f" - User {user_id}"
            
        fig.update_layout(
            title=title,
            barmode='group',
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        
        fig.update_yaxes(title_text="Amount (EUR)", secondary_y=False)
        fig.update_yaxes(title_text="Net Flow (EUR)", secondary_y=True)
        fig.update_xaxes(title_text="Month")
        
        return fig
    
    else:
        # Create static matplotlib visualization
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot bars for income and expenses
        x = np.arange(len(monthly_data))
        width = 0.35
        
        ax1.bar(x - width/2, monthly_data['credit_sum'], width, color='green', 
                alpha=0.7, label='Income')
        ax1.bar(x + width/2, monthly_data['debit_sum'], width, color='red', 
                alpha=0.7, label='Expenses')
        
        # Create second axis for net flow
        ax2 = ax1.twinx()
        ax2.plot(x, monthly_data['net_flow'], 'b-', linewidth=2, marker='o', 
                 label='Net Flow')
        
        # Set labels and title
        title = "Monthly Cashflow"
        if user_id:
            title += f" - User {user_id}"
            
        ax1.set_title(title)
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Amount (EUR)')
        ax2.set_ylabel('Net Flow (EUR)')
        
        # Set x-ticks to months
        ax1.set_xticks(x)
        ax1.set_xticklabels(monthly_data['month_label'], rotation=45)
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        return fig


def visualize_transaction_distribution(user_id: Optional[str] = None,
                                      start_date: Optional[str] = None,
                                      end_date: Optional[str] = None,
                                      use_plotly: bool = True,
                                      chart_type: str = 'pie') -> Union[plt.Figure, go.Figure]:
    """
    Visualize the distribution of transactions by type.
    
    Args:
        user_id: Optional user ID to filter data
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        use_plotly: Whether to use plotly (interactive) or matplotlib
        chart_type: Type of chart to create ('pie', 'donut', or 'stacked_bar')
        
    Returns:
        Figure object with the visualization
    """
    # Get transaction type summary using the correct function name
    type_summary = qd.group_transactions_by_type(user_id, start_date, end_date)
    
    if type_summary.empty:
        raise ValueError("No transaction data found for the specified parameters")
    
    if use_plotly:
        # Create interactive plotly visualization
        if chart_type in ('pie', 'donut'):
            # Prepare data for pie/donut chart
            credit_data = type_summary[['type', 'credit_sum']].copy()
            credit_data['side'] = 'CREDIT'
            credit_data = credit_data.rename(columns={'credit_sum': 'sum'})
            
            debit_data = type_summary[['type', 'debit_sum']].copy()
            debit_data['side'] = 'DEBIT'
            debit_data = debit_data.rename(columns={'debit_sum': 'sum'})
            
            # Combine the data
            combined_data = pd.concat([credit_data, debit_data], ignore_index=True)
            combined_data = combined_data[combined_data['sum'] > 0]
            
            # Create figure
            fig = px.pie(
                combined_data,
                values='sum',
                names='type',
                color='side',
                hole=0.4 if chart_type == 'donut' else 0,
                title="Transaction Distribution by Type",
                color_discrete_map={'CREDIT': 'green', 'DEBIT': 'red'}
            )
            
            # Improve layout
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=50, b=20),
            )
            
        elif chart_type == 'stacked_bar':
            # Get banking data by month and type
            banking_data = qd.get_banking_data(user_id=user_id, start_date=start_date, end_date=end_date)
            
            if banking_data.empty:
                raise ValueError("No banking data found for the specified parameters")
            
            # Add month column
            banking_data['month'] = pd.to_datetime(banking_data['bookingDate']).dt.strftime('%Y-%m')
            
            # Create figure
            fig = px.bar(
                banking_data,
                x='month',
                y='amount',
                color='type',
                barmode='stack',
                facet_row='side',
                title="Monthly Transaction Distribution by Type",
                labels={'month': 'Month', 'amount': 'Amount (EUR)', 'type': 'Transaction Type'},
                height=700
            )
            
            # Update layout
            fig.update_layout(
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=50, b=20),
            )
            
        return fig
    
    else:
        # Create static matplotlib visualization
        if chart_type in ('pie', 'donut'):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
            
            # Plot credit transactions
            credit_data = type_summary[type_summary['credit_sum'] > 0]
            ax1.pie(
                credit_data['credit_sum'],
                labels=credit_data['type'],
                autopct='%1.1f%%',
                startangle=90,
                shadow=False,
                wedgeprops=dict(width=0.4 if chart_type == 'donut' else 1)
            )
            ax1.set_title('Credit Transactions by Type')
            
            # Plot debit transactions
            debit_data = type_summary[type_summary['debit_sum'] > 0]
            ax2.pie(
                debit_data['debit_sum'],
                labels=debit_data['type'],
                autopct='%1.1f%%',
                startangle=90,
                shadow=False,
                wedgeprops=dict(width=0.4 if chart_type == 'donut' else 1)
            )
            ax2.set_title('Debit Transactions by Type')
            
            plt.suptitle("Transaction Distribution by Type", fontsize=16)
            
        elif chart_type == 'stacked_bar':
            # Get banking data
            banking_data = qd.get_banking_data(user_id=user_id, start_date=start_date, end_date=end_date)
            
            if banking_data.empty:
                raise ValueError("No banking data found for the specified parameters")
            
            # Add month column
            banking_data['month'] = pd.to_datetime(banking_data['bookingDate']).dt.strftime('%Y-%m')
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Credit transactions
            credit_data = banking_data[banking_data['side'] == 'CREDIT']
            credit_pivot = pd.pivot_table(
                credit_data,
                values='amount',
                index='month',
                columns='type',
                aggfunc='sum',
                fill_value=0
            )
            credit_pivot.plot(kind='bar', stacked=True, ax=ax1, colormap='Greens')
            ax1.set_title('Credit Transactions by Type')
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Amount (EUR)')
            
            # Debit transactions
            debit_data = banking_data[banking_data['side'] == 'DEBIT']
            debit_pivot = pd.pivot_table(
                debit_data,
                values='amount',
                index='month',
                columns='type',
                aggfunc='sum',
                fill_value=0
            )
            debit_pivot.plot(kind='bar', stacked=True, ax=ax2, colormap='Reds')
            ax2.set_title('Debit Transactions by Type')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Amount (EUR)')
            
            plt.suptitle("Monthly Transaction Distribution by Type", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
        return fig


def visualize_portfolio_composition(user_id: str, 
                                   date: Optional[str] = None,
                                   use_plotly: bool = True) -> Union[plt.Figure, go.Figure]:
    """
    Visualize the composition of a user's portfolio as a treemap.
    
    Args:
        user_id: User ID to visualize portfolio for
        date: Optional date for which to show portfolio (defaults to latest)
        use_plotly: Whether to use plotly (interactive) or matplotlib
        
    Returns:
        Figure object with the visualization
    """
    try:
        # Try to get user's current portfolio
        portfolio = qd.get_user_portfolio(user_id, as_of_date=date)
    except AttributeError:
        # If the function doesn't exist, create a simple fallback message
        if use_plotly:
            fig = go.Figure()
            fig.update_layout(**tr_theme['layout'])
            fig.add_annotation(
                text="Portfolio data not available - function not implemented",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(color='white', size=16)
            )
            return fig
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Portfolio data not available - function not implemented", 
                   ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return fig
    
    if portfolio.empty:
        raise ValueError(f"No portfolio data found for user {user_id}")
    
    # Calculate portfolio value
    portfolio['position_value'] = portfolio['quantity'] * portfolio['price']
    total_value = portfolio['position_value'].sum()
    
    # Calculate percentage of portfolio
    portfolio['percentage'] = (portfolio['position_value'] / total_value) * 100
    
    if use_plotly:
        # Create interactive plotly visualization
        fig = px.treemap(
            portfolio,
            path=['ISIN'],
            values='position_value',
            color='percentage',
            hover_data=['quantity', 'price', 'percentage'],
            color_continuous_scale='Viridis',
            title=f"Portfolio Composition - User {user_id}"
        )
        
        # Apply Trade Republic theme
        fig.update_layout(**tr_theme['layout'])
        
        # Update hover template to show more details
        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>Value: €%{value:.2f}<br>Qty: %{customdata[0]:.2f}<br>Price: €%{customdata[1]:.2f}<br>Portfolio: %{customdata[2]:.1f}%'
        )
        
        # Format the text inside treemap cells
        fig.update_traces(
            texttemplate='<b>%{label}</b><br>€%{value:.2f}<br>%{customdata[2]:.1f}%',
            textposition='middle center'
        )
        
        return fig
    else:
        # Create static matplotlib visualization
        # Note: Matplotlib doesn't have a built-in treemap, so we'll use squarify
        try:
            import squarify
        except ImportError:
            raise ImportError("Please install squarify package to create treemaps with matplotlib")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create colormap
        cmap = plt.cm.viridis
        norm = plt.Normalize(portfolio['percentage'].min(), portfolio['percentage'].max())
        colors = [cmap(norm(value)) for value in portfolio['percentage']]
        
        # Plot treemap
        squarify.plot(
            sizes=portfolio['position_value'],
            label=[f"{row['ISIN']}\n€{row['position_value']:,.2f}\n{row['percentage']:.1f}%" 
                   for _, row in portfolio.iterrows()],
            color=colors,
            alpha=0.8,
            ax=ax
        )
        
        ax.set_title(f"Portfolio Composition - User {user_id}")
        ax.axis('off')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Portfolio Percentage')
        
        return fig


def visualize_trading_activity_heatmap(user_id: Optional[str] = None,
                                      start_date: Optional[str] = None,
                                      end_date: Optional[str] = None,
                                      use_plotly: bool = True) -> Union[plt.Figure, go.Figure]:
    """
    Visualize trading activity as a heatmap showing frequency by day of week and hour.
    
    Args:
        user_id: Optional user ID to filter data
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        use_plotly: Whether to use plotly (interactive) or matplotlib
        
    Returns:
        Figure object with the visualization
    """
    # Get trading data - the function doesn't accept start_date/end_date as arguments
    # So we'll filter after getting the data
    trading_data = qd.get_trading_data(user_id=user_id)
    
    if not trading_data.empty and (start_date or end_date):
        # Convert executedAt to datetime for filtering
        trading_data['executedAt_dt'] = pd.to_datetime(trading_data['executedAt'])
        
        # Apply date filters if provided
        if start_date:
            start_dt = pd.to_datetime(start_date)
            trading_data = trading_data[trading_data['executedAt_dt'] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            trading_data = trading_data[trading_data['executedAt_dt'] <= end_dt]
    
    if trading_data.empty:
        raise ValueError("No trading data found for the specified parameters")
    
    # Extract day of week and hour from executedAt
    trading_data['day_of_week'] = pd.to_datetime(trading_data['executedAt']).dt.day_name()
    trading_data['hour'] = pd.to_datetime(trading_data['executedAt']).dt.hour
    
    # Create a pivot table for the heatmap
    heatmap_data = pd.pivot_table(
        trading_data,
        values='executionSize',
        index='day_of_week',
        columns='hour',
        aggfunc='count',
        fill_value=0
    )
    
    # Reorder days of week
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(days_order)
    
    if use_plotly:
        # Create interactive plotly visualization
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Hour of Day", y="Day of Week", color="Trade Count"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale=[[0, TR_GRAY], [0.5, TR_GREEN], [1, TR_GREEN]]
        )
        
        # Apply Trade Republic theme first
        fig.update_layout(**tr_theme['layout'])
        
        # Then apply specific overrides
        fig.update_layout(
            title="Trading Activity Heatmap by Day and Hour",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            coloraxis_colorbar=dict(title="Trade Count")
        )
        
        # Add annotations with the count values
        for i, day in enumerate(heatmap_data.index):
            for j, hour in enumerate(heatmap_data.columns):
                value = heatmap_data.iloc[i, j]
                if value > 0:
                    fig.add_annotation(
                        x=hour,
                        y=day,
                        text=str(int(value)),
                        showarrow=False,
                        font=dict(color="white" if value > heatmap_data.max().max()/2 else "black")
                    )
        
        return fig
    else:
        # Create static matplotlib visualization
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create heatmap
        heatmap = sns.heatmap(
            heatmap_data,
            cmap="Greens",
            annot=True,
            fmt="d",
            linewidths=.5,
            ax=ax,
            cbar_kws={'label': 'Trade Count'}
        )
        
        # Set title and labels
        title = "Trading Activity Heatmap by Day and Hour"
        if user_id:
            title += f" - User {user_id}"
            
        ax.set_title(title)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Day of Week")
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        return fig


def visualize_trading_vs_price(user_id: str,
                              isin: str,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              use_plotly: bool = True) -> Union[plt.Figure, go.Figure]:
    """
    Visualize a user's trading activity against the price of a specific security.
    
    Args:
        user_id: User ID to visualize trading for
        isin: ISIN of the security to analyze
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        use_plotly: Whether to use plotly (interactive) or matplotlib
        
    Returns:
        Figure object with the visualization
    """
    # Get all trading data for this security - without date filters which aren't supported
    trading_data = qd.get_trading_data(
        user_id=user_id,
        isin=isin
    )
    
    if not trading_data.empty and (start_date or end_date):
        # Convert executedAt to datetime for filtering
        trading_data['executedAt_dt'] = pd.to_datetime(trading_data['executedAt'])
        
        # Apply date filters if provided
        if start_date:
            start_dt = pd.to_datetime(start_date)
            trading_data = trading_data[trading_data['executedAt_dt'] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            trading_data = trading_data[trading_data['executedAt_dt'] <= end_dt]
    
    if trading_data.empty:
        raise ValueError(f"No trading data found for user {user_id} and ISIN {isin}")

    # Sort by date
    trading_data = trading_data.sort_values('executedAt')
    
    # Calculate cumulative position
    trading_data['signed_size'] = np.where(
        trading_data['direction'] == 'BUY',
        trading_data['executionSize'],
        -trading_data['executionSize']
    )
    trading_data['cumulative_position'] = trading_data['signed_size'].cumsum()
    
    # Calculate average price
    trading_data['value'] = trading_data['signed_size'] * trading_data['executionPrice']
    trading_data['cumulative_value'] = trading_data.apply(
        lambda row: row['value'] if row['direction'] == 'BUY' else -row['value'],
        axis=1
    ).cumsum()
    
    # Calculate average price
    trading_data['avg_price'] = np.abs(trading_data['cumulative_value'] / trading_data['cumulative_position'])
    trading_data['avg_price'] = trading_data['avg_price'].fillna(0)

    if use_plotly:
        # Create interactive plotly visualization with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=trading_data['executedAt'],
                y=trading_data['executionPrice'],
                mode='lines',
                name='Price',
                line=dict(color=TR_GRAY, width=2)
            ),
            secondary_y=False
        )
        
        # Add average price line
        fig.add_trace(
            go.Scatter(
                x=trading_data['executedAt'],
                y=trading_data['avg_price'],
                mode='lines',
                name='Avg. Buy Price',
                line=dict(color=TR_GREEN, width=2, dash='dash')
            ),
            secondary_y=False
        )
        
        # Add buy markers
        buys = trading_data[trading_data['direction'] == 'BUY']
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys['executedAt'],
                    y=buys['executionPrice'],
                    mode='markers',
                    name='Buy',
                    marker=dict(
                        color=TR_GREEN,
                        size=buys['executionSize'] * 5,
                        line=dict(width=1, color='darkgreen')
                    )
                ),
                secondary_y=False
            )
        
        # Add sell markers
        sells = trading_data[trading_data['direction'] == 'SELL']
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells['executedAt'],
                    y=sells['executionPrice'],
                    mode='markers',
                    name='Sell',
                    marker=dict(
                        color=TR_RED,
                        size=sells['executionSize'] * 5,
                        symbol='triangle-down',
                        line=dict(width=1, color='darkred')
                    )
                ),
                secondary_y=False
            )
        
        # Add cumulative position line
        fig.add_trace(
            go.Scatter(
                x=trading_data['executedAt'],
                y=trading_data['cumulative_position'],
                mode='lines',
                name='Position Size',
                line=dict(color='rgba(255,255,255,0.5)', width=2)
            ),
            secondary_y=True
        )
        
        # Apply Trade Republic theme
        fig.update_layout(**tr_theme['layout'])
        
        # Update layout
        fig.update_layout(
            title=f"Trading Activity vs Price - {isin}",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Set y-axis titles
        fig.update_yaxes(title_text="Price (EUR)", secondary_y=False)
        fig.update_yaxes(title_text="Position Size", secondary_y=True)
        
        return fig
    else:
        # Create static matplotlib visualization
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot price
        ax1.plot(trading_data['executedAt'], trading_data['executionPrice'], 
                 'w-', linewidth=2, label='Price')
        
        # Plot average price
        ax1.plot(trading_data['executedAt'], trading_data['avg_price'], 
                 'g--', linewidth=2, label='Avg. Buy Price')
        
        # Create second y-axis for position size
        ax2 = ax1.twinx()
        ax2.plot(trading_data['executedAt'], trading_data['cumulative_position'], 
                 color='gray', linewidth=2, alpha=0.5, label='Position Size')
        
        # Plot buy/sell markers
        buys = trading_data[trading_data['direction'] == 'BUY']
        if not buys.empty:
            ax1.scatter(buys['executedAt'], buys['executionPrice'], 
                       color='green', label='Buy', s=buys['executionSize'] * 20,
                       marker='o', edgecolor='darkgreen')
            
        sells = trading_data[trading_data['direction'] == 'SELL']
        if not sells.empty:
            ax1.scatter(sells['executedAt'], sells['executionPrice'], 
                       color='red', label='Sell', s=sells['executionSize'] * 20,
                       marker='v', edgecolor='darkred')
        
        # Set labels and title
        ax1.set_title(f"Trading Activity vs Price - {isin}")
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (EUR)')
        ax2.set_ylabel('Position Size')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.tight_layout()
        return fig


def visualize_largest_transactions(user_id: Optional[str] = None,
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None,
                                  n: int = 10,
                                  use_plotly: bool = True) -> Union[plt.Figure, go.Figure]:
    """
    Visualize the largest transactions (banking and trading).
    
    Args:
        user_id: Optional user ID to filter data
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        n: Number of largest transactions to show
        use_plotly: Whether to use plotly (interactive) or matplotlib
        
    Returns:
        Figure object with the visualization
    """
    # Get banking data
    banking_df = qd.get_banking_data(user_id=user_id, start_date=start_date, end_date=end_date)
    
    if banking_df.empty:
        raise ValueError("No banking data found for the specified parameters")
    
    # Select columns and prepare for visualization
    banking_txs = banking_df[['userId', 'bookingDate', 'side', 'amount', 'type']].copy()
    banking_txs['date'] = pd.to_datetime(banking_txs['bookingDate'])
    banking_txs['transaction_type'] = 'banking'
    
    # Create label column
    banking_txs['label'] = banking_txs.apply(
        lambda row: f"{row['date'].strftime('%Y-%m-%d')} | {row['type']} ({row['side']})",
        axis=1
    )
    
    # Get largest transactions
    largest_txs = banking_txs.nlargest(n, 'amount').copy()
    
    # Add color column based on side
    largest_txs['color'] = largest_txs['side'].apply(
        lambda side: TR_GREEN if side == 'CREDIT' else TR_RED
    )
    
    if use_plotly:
        # Create interactive plotly visualization
        fig = px.bar(
            largest_txs,
            y='label',
            x='amount',
            orientation='h',
            color='side',
            title=f"Largest Transactions (Top {n})",
            labels={'label': '', 'amount': 'Amount (EUR)', 'side': 'Type'},
            color_discrete_map={'CREDIT': TR_GREEN, 'DEBIT': TR_RED}
        )
        
        # Apply Trade Republic theme
        fig.update_layout(**tr_theme['layout'])
        
        # Update layout
        fig.update_layout(
            xaxis_title="Amount (EUR)",
            yaxis_title="",
            margin=dict(l=200)  # Make more room for labels
        )
        
        return fig
    else:
        # Create static matplotlib visualization
        fig, ax = plt.subplots(figsize=(12, max(8, n * 0.4)))  # Adjust height based on number of items
        
        # Plot horizontal bars
        bars = ax.barh(
            largest_txs['label'],
            largest_txs['amount'],
            color=largest_txs['color']
        )
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width * 1.01,
                bar.get_y() + bar.get_height()/2,
                f'€{width:,.2f}',
                va='center'
            )
        
        # Set title and labels
        title = f"Largest Transactions (Top {n})"
        if user_id:
            title += f" - User {user_id}"
            
        ax.set_title(title)
        ax.set_xlabel('Amount (EUR)')
        
        # Remove y-axis label since we have the transaction labels
        ax.set_ylabel('')
        
        # Add a legend for banking vs trading
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=TR_GREEN, label='CREDIT'),
            Patch(facecolor=TR_RED, label='DEBIT')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        return fig


def visualize_user_behavior_clustering(n_clusters: int = 3,
                                      min_transactions: int = 10,
                                      use_plotly: bool = True) -> Union[plt.Figure, go.Figure]:
    """
    Cluster users based on their trading patterns and visualize the results.
    
    Args:
        n_clusters: Number of clusters to identify
        min_transactions: Minimum number of transactions for a user to be included
        use_plotly: Whether to use plotly (interactive) or matplotlib
        
    Returns:
        Figure object with the visualization
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    
    # Get all users with trade data
    trading_df = qd.load_trading_data()
    user_counts = trading_df['userId'].value_counts()
    active_users = user_counts[user_counts >= min_transactions].index.tolist()
    
    if len(active_users) < n_clusters:
        raise ValueError(f"Not enough users with {min_transactions}+ transactions. Found {len(active_users)}, need at least {n_clusters}.")
    
    # Create features for each user
    user_features = []
    
    for user_id in active_users:
        user_trading = qd.get_trading_data(user_id=user_id)
        user_banking = qd.get_banking_data(user_id=user_id)
        
        # Skip if no trading data
        if user_trading.empty:
            continue
        
        # Trading features
        buy_ratio = len(user_trading[user_trading['direction'] == 'BUY']) / max(len(user_trading), 1)
        unique_securities = user_trading['ISIN'].nunique()
        avg_trade_size = user_trading['executionPrice'].mean()
        
        # Calculate trade frequency - handle case where there's only one trade
        if len(user_trading) <= 1:
            trade_frequency = 0
        else:
            trade_range = (user_trading['executedAt'].max() - user_trading['executedAt'].min()).days
            trade_frequency = len(user_trading) / max(trade_range, 1)
        
        # Banking features
        if not user_banking.empty:
            credits = user_banking[user_banking['side'] == 'CREDIT']['amount']
            debits = user_banking[user_banking['side'] == 'DEBIT']['amount']
            
            avg_deposit = credits.mean() if not credits.empty else 0
            avg_withdrawal = debits.mean() if not debits.empty else 0
        else:
            avg_deposit = 0
            avg_withdrawal = 0
        
        user_features.append({
            'userId': user_id,
            'buy_ratio': buy_ratio,
            'unique_securities': unique_securities,
            'avg_trade_size': avg_trade_size,
            'trade_frequency': trade_frequency,
            'avg_deposit': avg_deposit,
            'avg_withdrawal': avg_withdrawal
        })
    
    # Convert to DataFrame
    features_df = pd.DataFrame(user_features)
    
    # Skip if no features
    if features_df.empty:
        raise ValueError("No user features could be calculated")
    
    features_df.set_index('userId', inplace=True)
    
    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    features_scaled = imputer.fit_transform(features_df)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_scaled)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)
    
    # Add cluster to features
    features_df['cluster'] = clusters
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features_scaled)
    
    # Add reduced features to DataFrame
    features_df['x'] = reduced_features[:, 0]
    features_df['y'] = reduced_features[:, 1]
    
    # Get cluster centers
    centroids = kmeans.cluster_centers_
    centroids_reduced = pca.transform(centroids)
    
    # Create a DataFrame for centroids
    centroids_df = pd.DataFrame({
        'cluster': range(n_clusters),
        'x': centroids_reduced[:, 0],
        'y': centroids_reduced[:, 1]
    })
    
    # Define trading styles based on features
    trading_styles = ["Style " + str(i+1) for i in range(n_clusters)]
    
    # Try to interpret cluster patterns and assign descriptive names
    cluster_means = features_df.groupby('cluster').mean()
    
    for i in range(n_clusters):
        cluster_data = cluster_means.iloc[i]
        
        # Interpret cluster patterns - this is a simple heuristic and can be improved
        if cluster_data['buy_ratio'] > 0.7 and cluster_data['trade_frequency'] < 0.3:
            trading_styles[i] = "Buy and Hold"
        elif cluster_data['trade_frequency'] > 0.7:
            trading_styles[i] = "Active Trader"
        elif cluster_data['unique_securities'] > cluster_means['unique_securities'].median() * 1.5:
            trading_styles[i] = "Diversified Investor"
        elif cluster_data['avg_trade_size'] > cluster_means['avg_trade_size'].median() * 1.5:
            trading_styles[i] = "High Roller"
        elif cluster_data['avg_deposit'] > cluster_data['avg_withdrawal'] * 2:
            trading_styles[i] = "Net Depositor"
        elif cluster_data['avg_withdrawal'] > cluster_data['avg_deposit'] * 2:
            trading_styles[i] = "Net Withdrawer"
    
    if use_plotly:
        # Create interactive plotly visualization
        fig = go.Figure()
        
        # Plot each cluster
        for i in range(n_clusters):
            cluster_data = features_df[features_df['cluster'] == i]
            fig.add_trace(
                go.Scatter(
                    x=cluster_data['x'],
                    y=cluster_data['y'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=px.colors.qualitative.Safe[i % len(px.colors.qualitative.Safe)]
                    ),
                    name=trading_styles[i],
                    text=cluster_data.index,
                    hovertemplate='<b>%{text}</b><br>' +
                                  'Buy Ratio: %{customdata[0]:.2f}<br>' +
                                  'Unique Securities: %{customdata[1]}<br>' +
                                  'Avg Trade Size: €%{customdata[2]:.2f}<br>' +
                                  'Trade Frequency: %{customdata[3]:.2f}/day<br>',
                    customdata=cluster_data[['buy_ratio', 'unique_securities', 'avg_trade_size', 'trade_frequency']].values
                )
            )
            
            # Add cluster centroid
            centroid = centroids_df[centroids_df['cluster'] == i]
            fig.add_trace(
                go.Scatter(
                    x=centroid['x'],
                    y=centroid['y'],
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        size=15,
                        color=px.colors.qualitative.Safe[i % len(px.colors.qualitative.Safe)],
                        line=dict(width=2, color='white')
                    ),
                    name=f"{trading_styles[i]} (Centroid)",
                    showlegend=False
                )
            )
        
        # Apply Trade Republic theme
        fig.update_layout(**tr_theme['layout'])
        
        # Update layout
        fig.update_layout(
            title="User Trading Behavior Clustering",
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            legend_title="Trading Style"
        )
        
        return fig
    else:
        # Create static matplotlib visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define colors
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        
        # Plot each cluster
        for i in range(n_clusters):
            cluster_data = features_df[features_df['cluster'] == i]
            ax.scatter(
                cluster_data['x'],
                cluster_data['y'],
                c=[colors[i]],
                label=trading_styles[i],
                alpha=0.7,
                s=80
            )
            
            # Plot centroid
            centroid = centroids_df[centroids_df['cluster'] == i]
            ax.scatter(
                centroid['x'],
                centroid['y'],
                c=[colors[i]],
                marker='*',
                s=300,
                edgecolor='white',
                linewidth=1.5
            )
            
            # Add a text label for the centroid
            ax.annotate(
                trading_styles[i],
                (centroid['x'].iloc[0], centroid['y'].iloc[0]),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=12,
                fontweight='bold'
            )
        
        # Set labels and title
        ax.set_title("User Trading Behavior Clustering")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        
        # Add legend
        ax.legend(title="Trading Style")
        
        plt.tight_layout()
        return fig


def visualize_security_performance(user_id: Optional[str] = None,
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None,
                                  n: int = 5,
                                  use_plotly: bool = True) -> Union[plt.Figure, go.Figure]:
    """
    Visualize performance of the top-N traded securities for a user.
    
    Args:
        user_id: Optional user ID to filter data
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        n: Number of securities to compare
        use_plotly: Whether to use plotly (interactive) or matplotlib
        
    Returns:
        Figure object with the visualization
    """
    # Get trading data without date filters
    trading_df = qd.get_trading_data(user_id=user_id)
    
    # Apply date filtering manually if needed
    if not trading_df.empty and (start_date or end_date):
        trading_df['executedAt_dt'] = pd.to_datetime(trading_df['executedAt'])
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            trading_df = trading_df[trading_df['executedAt_dt'] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            trading_df = trading_df[trading_df['executedAt_dt'] <= end_dt]
    
    if trading_df.empty:
        raise ValueError("No trading data found for the specified parameters")

    # Count trades per security
    isin_counts = trading_df['ISIN'].value_counts()
    valid_isins = isin_counts[isin_counts >= n].index.tolist()
    
    if not valid_isins:
        raise ValueError(f"No securities with at least {n} trades found")
    
    # Calculate performance metrics for each security
    performance_data = []
    
    for isin in valid_isins:
        isin_trades = trading_df[trading_df['ISIN'] == isin].sort_values('executedAt')
        
        # Calculate return
        first_trade = isin_trades.iloc[0]
        last_trade = isin_trades.iloc[-1]
        
        # Calculate buy/sell prices
        buy_prices = isin_trades[isin_trades['direction'] == 'BUY']['executionPrice']
        sell_prices = isin_trades[isin_trades['direction'] == 'SELL']['executionPrice']
        
        if not buy_prices.empty:
            avg_buy_price = buy_prices.mean()
            
            if not sell_prices.empty:
                avg_sell_price = sell_prices.mean()
                returns = (avg_sell_price - avg_buy_price) / avg_buy_price * 100
            else:
                # If no sells, use last price as hypothetical sell
                returns = (last_trade['executionPrice'] - avg_buy_price) / avg_buy_price * 100
        else:
            # Skip if no buys
            continue
        
        # Calculate volatility (standard deviation of price)
        volatility = isin_trades['executionPrice'].std()
        
        # Calculate holding period
        days_held = (last_trade['executedAt'] - first_trade['executedAt']).days
        
        # Add to results
        performance_data.append({
            'ISIN': isin,
            'returns': returns,
            'volatility': volatility,
            'days_held': days_held,
            'trade_count': len(isin_trades),
            'last_price': last_trade['executionPrice'],
            'first_trade': first_trade['executedAt'],
            'last_trade': last_trade['executedAt']
        })
    
    # Convert to DataFrame
    performance_df = pd.DataFrame(performance_data)
    
    if performance_df.empty:
        raise ValueError("Could not calculate performance metrics")
    
    # Calculate risk-adjusted return (simple version)
    performance_df['risk_adjusted_return'] = performance_df['returns'] / performance_df['volatility']
    
    if use_plotly:
        # Create interactive plotly visualization - risk vs return scatterplot
        fig = px.scatter(
            performance_df,
            x='volatility',
            y='returns',
            size='trade_count',
            color='risk_adjusted_return',
            hover_name='ISIN',
            text='ISIN',
            title="Security Performance: Risk vs Return",
            labels={
                'volatility': 'Risk (Price Volatility)',
                'returns': 'Return (%)',
                'trade_count': 'Number of Trades',
                'risk_adjusted_return': 'Risk-Adjusted Return'
            },
            color_continuous_scale=[TR_RED, TR_GRAY, TR_GREEN]
        )
        
        # Add quadrant lines
        fig.add_shape(
            type="line",
            x0=performance_df['volatility'].min(),
            y0=0,
            x1=performance_df['volatility'].max(),
            y1=0,
            line=dict(color=TR_GRAY, width=1, dash="dash")
        )
        
        fig.add_shape(
            type="line",
            x0=performance_df['volatility'].mean(),
            y0=performance_df['returns'].min(),
            x1=performance_df['volatility'].mean(),
            y1=performance_df['returns'].max(),
            line=dict(color=TR_GRAY, width=1, dash="dash")
        )
        
        # Add quadrant annotations
        fig.add_annotation(
            x=performance_df['volatility'].min() * 1.1,
            y=performance_df['returns'].max() * 0.9,
            text="Low Risk<br>High Return",
            showarrow=False,
            font=dict(color=TR_GREEN, size=10)
        )
        
        fig.add_annotation(
            x=performance_df['volatility'].max() * 0.9,
            y=performance_df['returns'].max() * 0.9,
            text="High Risk<br>High Return",
            showarrow=False,
            font=dict(color=TR_GRAY, size=10)
        )
        
        fig.add_annotation(
            x=performance_df['volatility'].min() * 1.1,
            y=performance_df['returns'].min() * 0.9,
            text="Low Risk<br>Low Return",
            showarrow=False,
            font=dict(color=TR_RED, size=10)
        )
        
        fig.add_annotation(
            x=performance_df['volatility'].max() * 0.9,
            y=performance_df['returns'].min() * 0.9,
            text="High Risk<br>Low Return",
            showarrow=False,
            font=dict(color=TR_RED, size=10)
        )
        
        # Apply Trade Republic theme
        fig.update_layout(**tr_theme['layout'])
        
        # Update layout
        fig.update_layout(
            hovermode="closest",
            xaxis_title="Risk (Price Volatility)",
            yaxis_title="Return (%)"
        )
        
        # Format text
        fig.update_traces(
            textposition='top center',
            textfont=dict(size=8, color='white')
        )
        
        return fig
    else:
        # Create static matplotlib visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a scatter plot
        scatter = ax.scatter(
            performance_df['volatility'],
            performance_df['returns'],
            s=performance_df['trade_count'] * 20,
            c=performance_df['risk_adjusted_return'],
            cmap='RdYlGn',
            alpha=0.7
        )
        
        # Add ISIN labels to the points
        for i, row in performance_df.iterrows():
            ax.annotate(
                row['ISIN'],
                (row['volatility'], row['returns']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
        
        # Add quadrant lines
        ax.axhline(y=0, color=TR_GRAY, linestyle='--', alpha=0.5)
        ax.axvline(x=performance_df['volatility'].mean(), color=TR_GRAY, linestyle='--', alpha=0.5)
        
        # Add quadrant labels
        ax.text(
            performance_df['volatility'].min() * 1.1,
            performance_df['returns'].max() * 0.9,
            "Low Risk\nHigh Return",
            fontsize=10,
            color=TR_GREEN
        )
        
        ax.text(
            performance_df['volatility'].max() * 0.9,
            performance_df['returns'].max() * 0.9,
            "High Risk\nHigh Return",
            fontsize=10,
            color=TR_GRAY
        )
        
        ax.text(
            performance_df['volatility'].min() * 1.1,
            performance_df['returns'].min() * 0.9,
            "Low Risk\nLow Return",
            fontsize=10,
            color=TR_RED
        )
        
        ax.text(
            performance_df['volatility'].max() * 0.9,
            performance_df['returns'].min() * 0.9,
            "High Risk\nLow Return",
            fontsize=10,
            color=TR_RED
        )
        
        # Set title and labels
        title = "Security Performance: Risk vs Return"
        if user_id:
            title += f" - User {user_id}"
        
        ax.set_title(title)
        ax.set_xlabel('Risk (Price Volatility)')
        ax.set_ylabel('Return (%)')
        
        # Add a colorbar for risk-adjusted return
        cbar = plt.colorbar(scatter)
        cbar.set_label('Risk-Adjusted Return')
        
        plt.tight_layout()
        return fig


def visualize_transaction_correlation(user_id: Optional[str] = None,
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None,
                                     use_plotly: bool = True) -> Union[plt.Figure, go.Figure]:
    """
    Visualize correlation between different transaction types.
    
    Args:
        user_id: Optional user ID to filter data
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        use_plotly: Whether to use plotly (interactive) or matplotlib
        
    Returns:
        Figure object with the visualization
    """
    # Get banking data
    banking_df = qd.get_banking_data(user_id=user_id, start_date=start_date, end_date=end_date)
    
    if banking_df.empty:
        raise ValueError("No banking data found for the specified parameters")
    
    # Get trading data - without date params which are not supported
    trading_df = qd.get_trading_data(user_id=user_id)
    
    # Apply date filtering manually if needed
    if not trading_df.empty and (start_date or end_date):
        trading_df['executedAt_dt'] = pd.to_datetime(trading_df['executedAt'])
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            trading_df = trading_df[trading_df['executedAt_dt'] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            trading_df = trading_df[trading_df['executedAt_dt'] <= end_dt]
    
    # Prepare data for correlation analysis
    # Group transactions by day and type
    banking_daily = banking_df.copy()
    banking_daily['date'] = pd.to_datetime(banking_daily['bookingDate']).dt.date
    
    # Create pivoted dataframes for banking side+type combinations
    banking_pivot = pd.pivot_table(
        banking_daily,
        values='amount',
        index='date',
        columns=['side', 'type'],
        aggfunc='sum',
        fill_value=0
    )
    
    # Flatten column names
    banking_pivot.columns = [f"{side}_{type_}" for side, type_ in banking_pivot.columns]
    
    # Add trading data if available
    if not trading_df.empty:
        trading_daily = trading_df.copy()
        trading_daily['date'] = pd.to_datetime(trading_daily['executedAt']).dt.date
        
        # Create pivoted dataframe for trading direction
        trading_pivot = pd.pivot_table(
            trading_daily,
            values='executionPrice',
            index='date',
            columns='direction',
            aggfunc='sum',
            fill_value=0
        )
        
        # Add trading columns to banking pivot
        for direction in trading_pivot.columns:
            banking_pivot[f'TRADING_{direction}'] = trading_pivot[direction]
    
    # Calculate correlation matrix
    corr_matrix = banking_pivot.corr()
    
    if use_plotly:
        # Create interactive plotly visualization
        fig = px.imshow(
            corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            color_continuous_scale=[TR_RED, TR_GRAY, TR_GREEN],
            zmin=-1, zmax=1,
            title="Transaction Type Correlation Matrix"
        )
        
        # Apply Trade Republic theme
        fig.update_layout(**tr_theme['layout'])
        
        # Update layout - fixing the colorbar configuration
        fig.update_layout(
            xaxis_title="Transaction Type",
            yaxis_title="Transaction Type",
            coloraxis_colorbar=dict(
                title="Correlation"
            )
        )
        
        # Add correlation values to heatmap
        for i, row in enumerate(corr_matrix.index):
            for j, col in enumerate(corr_matrix.columns):
                value = corr_matrix.iloc[i, j]
                color = "white" if abs(value) > 0.5 else "black"
                fig.add_annotation(
                    x=col,
                    y=row,
                    text=f"{value:.2f}",
                    showarrow=False,
                    font=dict(color=color, size=9)
                )
        
        return fig
    else:
        # Create static matplotlib visualization
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(10, 133, as_cmap=True)
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmin=-1, vmax=1,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            ax=ax,
            cbar_kws={"shrink": 0.8, "label": "Correlation"}
        )
        
        # Set title and labels
        title = "Transaction Type Correlation Matrix"
        if user_id:
            title += f" - User {user_id}"
            
        ax.set_title(title)
        
        # Rotate column labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # Example usage
    user_id = "00909ba7-ad01-42f1-9074-2773c7d3cf2c"
    
    # Account balance visualization
    balance_fig = visualize_account_balance(user_id)
    balance_fig.show()
    
    # Monthly cashflow visualization
    cashflow_fig = visualize_monthly_cashflow(user_id)
    cashflow_fig.show()
    
    # Transaction distribution visualization
    try:
        distribution_fig = visualize_transaction_distribution(user_id, chart_type='donut')
        distribution_fig.show()
    except Exception as e:
        print(f"Could not generate transaction distribution visualization: {str(e)}")
    
    # Portfolio composition visualization
    try:
        portfolio_fig = visualize_portfolio_composition(user_id)
        portfolio_fig.show()
    except Exception as e:
        print(f"Could not generate portfolio visualization: {str(e)}")
    
    # Trading activity heatmap
    try:
        heatmap_fig = visualize_trading_activity_heatmap(user_id)
        heatmap_fig.show()
    except Exception as e:
        print(f"Could not generate trading heatmap: {str(e)}")
    
    # Trading vs price visualization for a specific security
    # First, find a security the user has traded multiple times
    user_trading = qd.get_trading_data(user_id=user_id)
    isin_counts = user_trading['ISIN'].value_counts()
    if any(isin_counts > 1):
        most_traded_isin = isin_counts.index[0]
        trading_price_fig = visualize_trading_vs_price(user_id, most_traded_isin)
        trading_price_fig.show()
    
    # Largest transactions visualization
    largest_tx_fig = visualize_largest_transactions(user_id, n=15)
    largest_tx_fig.show()

    # User behavior clustering
    try:
        clustering_fig = visualize_user_behavior_clustering(n_clusters=3)
        clustering_fig.show()
    except Exception as e:
        print(f"Could not generate user clustering visualization: {str(e)}")
    
    # Security performance comparison
    try:
        performance_fig = visualize_security_performance(user_id)
        performance_fig.show()
    except Exception as e:
        print(f"Could not generate security performance visualization: {str(e)}")
    
    # Transaction correlation analysis
    try:
        correlation_fig = visualize_transaction_correlation(user_id)
        correlation_fig.show()
    except Exception as e:
        print(f"Could not generate transaction correlation visualization: {str(e)}")
