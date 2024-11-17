# Use `pip install -r requirements.txt` command in shell. Alternatively `pip3 install -r requirements.txt` can also be used.
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

import app_functions as uf

### Streamlit App
image = Image.open('title02.png')
st.set_page_config(page_title = "WQU Swing Trading", page_icon="favicon.ico", layout = "wide")
# st.title(f"Swing Trading")
st.image(image)

st.divider()
selected_strategy = st.sidebar.selectbox("Select Strategy", ["Coiled Spring NR7", "Finger Finder", "Power Spike"])

start_date = st.sidebar.date_input("Start Date")
initial_capital_usd = st.sidebar.number_input("Initial Capital ($)", min_value=1000, step=1000)

welcome_statement = """This web app is designed as a part of capstone at WorldQuant University. Please make sure that the start date is atleast 6 months from today. 
Emphasis will be on identifying optimal entry and exit points and following the simple rule of
achieving returns: minimizing risks and maximizing profits."""
st.write(welcome_statement)

if st.sidebar.button("Run Strategy"):
    tab1, tab2 = st.tabs(["Tabular", "Plots"])
    with tab1:
        ## Currency Conversion (free api)
        api_key = "415715a52d797424cb41a1b8"
        url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/USD/INR"

        response = requests.get(url)
        data = response.json()
        current_conversion_rate = data['conversion_rate']

        converted_initial_capital_inr = initial_capital_usd * current_conversion_rate
        currency_convert_statement = f"With the current conversion rate, ${initial_capital_usd} would be equal to ₹{converted_initial_capital_inr}."
        st.write(currency_convert_statement)

        with st.status("Operation in progress. Please wait."):
            ## Data Download
            st.write("Getting tickers (1/7)")
            # NIFTY 50
            nifty_50_link = 'https://en.wikipedia.org/wiki/NIFTY_50'
            nifty_50_df = pd.read_html(nifty_50_link)[1]

            NIFTY_50_TICKER_yfinance = nifty_50_df['Symbol']+'.NS'
            NIFTY_50_TICKER_yfinance = NIFTY_50_TICKER_yfinance.to_list()

            # NIFTY NEXT 50
            nifty_next_50_link = 'https://en.wikipedia.org/wiki/NIFTY_Next_50'
            nifty_next_50_df = pd.read_html(nifty_next_50_link)[2]

            NIFTY_NEXT_50_TICKER_yfinance = nifty_next_50_df['Symbol']+'.NS'
            NIFTY_NEXT_50_TICKER_yfinance = NIFTY_NEXT_50_TICKER_yfinance.to_list()

            nifty_ticker_sector_df = pd.DataFrame(columns=['Ticker', 'Sector'])
            nifty_ticker_sector_df['Ticker'] = pd.concat([nifty_50_df['Symbol'], nifty_next_50_df['Symbol']])
            nifty_ticker_sector_df['Sector'] = pd.concat([nifty_50_df[f'Sector[15]'], nifty_next_50_df['Sector']])
            nifty_ticker_sector_df['Ticker'] = nifty_ticker_sector_df['Ticker']+'.NS'


            for i, n in enumerate(NIFTY_NEXT_50_TICKER_yfinance):
                if n == 'MCDOWELL-N.NS':
                    NIFTY_NEXT_50_TICKER_yfinance[i] = 'UNITDSPR.NS'
                    
            all_stock_tickers = NIFTY_50_TICKER_yfinance + NIFTY_NEXT_50_TICKER_yfinance
            all_data = []

            current_date = date.today()
            st.write(f'Gathering data from {start_date} to {current_date} (2/7)')
            
            for ticker in all_stock_tickers:
                df = uf.get_stock_data(ticker, start_date, current_date)
                # print(f'Data on stock {ticker} has been downloaded.')
                df = uf.add_technical_indicators(df)
                # print(f'Technical indicators added to stock {ticker}.')
                all_data.append(df)
            st.write("Data acquired (3/7)")

            # Combine all DataFrames
            combined_df = pd.concat(all_data, ignore_index=True)

            # Add bb, rsi indicators
            df = uf.add_bb_rsi(combined_df)
            df = uf.fill_nan_values(df)
            st.write("Data cleaned (4/7)")
            st.write("Technical indicators added (5/7)")
        
            strategies = {
                'Coiled Spring NR7': uf.coiled_spring_nr7,
                'Finger Finder': uf.finger_finder,
                'Power Spike': uf.power_spike
            }

            st.write(f"Applying {selected_strategy} strategy (6/7)")

            results = {}
            dataframes = {}

            # Run only the selected strategy
            strategy_func = strategies[selected_strategy]
            strategy_results = {}
            strategy_dataframes = {}

            for df in all_data:
                ticker = df['Ticker'].iloc[0]
                stock_results, stock_df = uf.process_stock(df, strategy_func, capital=converted_initial_capital_inr)
                strategy_results[ticker] = stock_results
                strategy_dataframes[ticker] = stock_df
            st.write("Strategy applied (7/7)")

            results[selected_strategy] = strategy_results
            dataframes[selected_strategy] = strategy_dataframes

            results_df = pd.DataFrame({(ticker): performance 
                                    for ticker, performance in strategy_results.items()})
            

            results_df = results_df.T
            
            # Add a col for profit/loss
            results_df['Profit/Loss (INR)'] = ''
            results_df['Profit/Loss (USD)'] = ''
            
            for i in range(len(results_df)):
                results_df['Profit/Loss (INR)'][i] =  results_df['Final Portfolio Value'][i] - converted_initial_capital_inr
                results_df['Profit/Loss (USD)'][i] =  results_df['Profit/Loss (INR)'][i] / current_conversion_rate

        st.write("Preparing results.")
        st.subheader(f"Results for {selected_strategy}:", divider=True)
        st.table(results_df.round(3))
        st.caption(":red[_Any NA values in the table suggest that there were no trade signals during the selected period._]")

        st.divider()

        st.write("The following tables might give better insights.")
        
        profitable_trades_df = pd.DataFrame(columns=['Ticker', 'Profit (INR)', 'Profit (USD)'])

        # Filter profitable trades
        for ticker, row in results_df.iterrows():
            if row['Profit/Loss (INR)'] > 0:
                profitable_trades_df = profitable_trades_df.append({'Ticker': ticker,
                                        'Profit (INR)': np.round(row['Profit/Loss (INR)'], 2),
                                        'Profit (USD)': np.round(row['Profit/Loss (USD)'], 2)},
                                        ignore_index=True)
        
        hide_index_profitable_trades_df = profitable_trades_df.set_index('Ticker')
        hide_index_profitable_trades_df = hide_index_profitable_trades_df.rename_axis(None)

        sorted_profitable_stocks = profitable_trades_df.sort_values(by='Profit (INR)', ascending=False)
        sorted_profitable_stocks['Sector'] = ''
        for index, row in sorted_profitable_stocks.iterrows():
            ticker = row['Ticker']
            sector = nifty_ticker_sector_df[nifty_ticker_sector_df['Ticker'] == ticker]['Sector'].values[0]
            sorted_profitable_stocks.at[index, 'Sector'] = sector
        hide_index_sorted_profitable_stocks = sorted_profitable_stocks.set_index('Ticker')
        hide_index_sorted_profitable_stocks = hide_index_sorted_profitable_stocks.rename_axis(None)

        sector_counts = sorted_profitable_stocks['Sector'].value_counts()
        sector_appearance_df = pd.DataFrame({'Sector': sector_counts.index, 'Times Appeared': sector_counts.values})
        hide_index_sector_appearance_df = sector_appearance_df.set_index('Sector')
        hide_index_sector_appearance_df = hide_index_sector_appearance_df.rename_axis(None)


        # Create three containers
        container1, container2, container3 = st.columns(3)

        # Display DataFrames in each container
        with container1:
            st.subheader("Profitable Stocks")
            st.caption("Sorted alphabetically (A➡Z)")
            st.table(hide_index_profitable_trades_df.round(3))

        with container2:
            st.subheader("Profitable Stocks")
            st.caption("Sorted by Profit (High➡Low)")
            # st.dataframe(sorted_profitable_stocks)
            st.table(hide_index_sorted_profitable_stocks.round(3))

        with container3:
            st.subheader("Sector Representation")
            st.caption("Sectors of profitable stocks")
            st.table(hide_index_sector_appearance_df.round(3))
        
        st.divider()

    with tab2:
        st.write("This section displays the buy/sell for the stocks with profitable trades for the selected duration.")
    
        st.set_option('deprecation.showPyplotGlobalUse', False)
        uf.filter_dataframes_and_plot(selected_strategy, dataframes, sorted_profitable_stocks)
        
