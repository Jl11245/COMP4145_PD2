# === Additional Strategies ===
def calculate_bollinger_bands(data, window=3, num_std=0.5):
    data['MA3'] = data['Close'].rolling(window=window).mean()
    data['BB_up'] = data['MA3'] + num_std * data['Close'].rolling(window=window).std()
    data['BB_dn'] = data['MA3'] - num_std * data['Close'].rolling(window=window).std()
    return data

def get_trades_bollinger(data):
    # Buy when price crosses below lower band, sell when crosses above upper band
    positions = []
    in_position = False
    for i in range(1, len(data)):
        if not in_position and data['Close'].iloc[i-1] > data['BB_dn'].iloc[i-1] and data['Close'].iloc[i] < data['BB_dn'].iloc[i]:
            buy_date = data.index[i]
            buy_price = data['Close'].iloc[i]
            in_position = True
        elif in_position and data['Close'].iloc[i-1] < data['BB_up'].iloc[i-1] and data['Close'].iloc[i] > data['BB_up'].iloc[i]:
            sell_date = data.index[i]
            sell_price = data['Close'].iloc[i]
            holding_days = (sell_date - buy_date).days
            profit_pct = (sell_price / buy_price - 1) * 100
            positions.append({'BuyDate': buy_date, 'BuyPrice': buy_price, 'SellDate': sell_date, 'SellPrice': sell_price, 'HoldingDays': holding_days, 'ProfitPct': profit_pct, 'SellReason': 'BBands Exit'})
            in_position = False
    return pd.DataFrame(positions)

def calculate_rsi(data, window=2):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def get_trades_rsi(data, lower=48, upper=52):
    # Buy when RSI crosses below lower, sell when RSI crosses above upper
    positions = []
    in_position = False
    for i in range(1, len(data)):
        if not in_position and data['RSI'].iloc[i-1] > lower and data['RSI'].iloc[i] < lower:
            buy_date = data.index[i]
            buy_price = data['Close'].iloc[i]
            in_position = True
        elif in_position and data['RSI'].iloc[i-1] < upper and data['RSI'].iloc[i] > upper:
            sell_date = data.index[i]
            sell_price = data['Close'].iloc[i]
            holding_days = (sell_date - buy_date).days
            profit_pct = (sell_price / buy_price - 1) * 100
            positions.append({'BuyDate': buy_date, 'BuyPrice': buy_price, 'SellDate': sell_date, 'SellPrice': sell_price, 'HoldingDays': holding_days, 'ProfitPct': profit_pct, 'SellReason': 'RSI Exit'})
            in_position = False
    return pd.DataFrame(positions)

def calculate_macd(data, fast=2, slow=5, signal=2):
    data['EMA_fast'] = data['Close'].ewm(span=fast, adjust=False).mean()
    data['EMA_slow'] = data['Close'].ewm(span=slow, adjust=False).mean()
    data['MACD'] = data['EMA_fast'] - data['EMA_slow']
    data['MACD_signal'] = data['MACD'].ewm(span=signal, adjust=False).mean()
    return data

def get_trades_macd(data):
    # Buy when MACD crosses above signal, sell when crosses below
    positions = []
    in_position = False
    for i in range(1, len(data)):
        if not in_position and data['MACD'].iloc[i-1] < data['MACD_signal'].iloc[i-1] and data['MACD'].iloc[i] > data['MACD_signal'].iloc[i]:
            buy_date = data.index[i]
            buy_price = data['Close'].iloc[i]
            in_position = True
        elif in_position and data['MACD'].iloc[i-1] > data['MACD_signal'].iloc[i-1] and data['MACD'].iloc[i] < data['MACD_signal'].iloc[i]:
            sell_date = data.index[i]
            sell_price = data['Close'].iloc[i]
            holding_days = (sell_date - buy_date).days
            profit_pct = (sell_price / buy_price - 1) * 100
            positions.append({'BuyDate': buy_date, 'BuyPrice': buy_price, 'SellDate': sell_date, 'SellPrice': sell_price, 'HoldingDays': holding_days, 'ProfitPct': profit_pct, 'SellReason': 'MACD Exit'})
            in_position = False
    return pd.DataFrame(positions)

def calculate_ema(data, window=3, col_name='EMA'):
    data[col_name] = data['Close'].ewm(span=window, adjust=False).mean()
    return data

def get_trades_ema(data):
    # Buy when price crosses above EMA, sell when crosses below
    positions = []
    in_position = False
    for i in range(1, len(data)):
        if not in_position and data['Close'].iloc[i-1] < data['EMA'].iloc[i-1] and data['Close'].iloc[i] > data['EMA'].iloc[i]:
            buy_date = data.index[i]
            buy_price = data['Close'].iloc[i]
            in_position = True
        elif in_position and data['Close'].iloc[i-1] > data['EMA'].iloc[i-1] and data['Close'].iloc[i] < data['EMA'].iloc[i]:
            sell_date = data.index[i]
            sell_price = data['Close'].iloc[i]
            holding_days = (sell_date - buy_date).days
            profit_pct = (sell_price / buy_price - 1) * 100
            positions.append({'BuyDate': buy_date, 'BuyPrice': buy_price, 'SellDate': sell_date, 'SellPrice': sell_price, 'HoldingDays': holding_days, 'ProfitPct': profit_pct, 'SellReason': 'EMA Exit'})
            in_position = False
    return pd.DataFrame(positions)
import streamlit as st

# =================== IMPORTS ===================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
# =================== INDICATOR PARAMETERS ===================
# Mapping: timeframe -> indicator -> parameters
indicator_params = {
    "Month": {
        "Bollinger Bands": {"window": 5, "num_std": 1.5},
        "RSI": {"window": 7, "lower": 30, "upper": 70},
        "MACD": {"fast": 6, "slow": 13, "signal": 5},
        "EMA": {"fast": 5, "slow": 15},
    },
    "Half Year": {
        "Bollinger Bands": {"window": 10, "num_std": 2},
        "RSI": {"window": 14, "lower": 30, "upper": 70},
        "MACD": {"fast": 12, "slow": 26, "signal": 9},
        "EMA": {"fast": 10, "slow": 30},
    },
    "Year": {
        "Bollinger Bands": {"window": 20, "num_std": 2},
        "RSI": {"window": 14, "lower": 30, "upper": 70},
        "MACD": {"fast": 12, "slow": 26, "signal": 9},
        "EMA": {"fast": 20, "slow": 60},
    },
    "5 Year": {
        "Bollinger Bands": {"window": 30, "num_std": 2.5},
        "RSI": {"window": 21, "lower": 30, "upper": 70},
        "MACD": {"fast": 24, "slow": 52, "signal": 18},
        "EMA": {"fast": 50, "slow": 150},
    },
}

# =================== STRATEGY FUNCTIONS ===================
def get_stock_data(ticker, period="5y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

def calculate_moving_averages(data):
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    return data

def identify_golden_cross(data):
    data['GoldenCross'] = (data['MA50'] > data['MA200']) & (data['MA50'].shift(1) <= data['MA200'].shift(1))
    return data

def get_trades(data):
    positions = []
    data = data.iloc[50:].copy()
    buy_dates = data[data['GoldenCross'] == True].index.tolist()
    for buy_date in buy_dates:
        buy_price = data.loc[buy_date, 'Close']
        target_price = buy_price * 1.15
        max_sell_date = buy_date + pd.Timedelta(days=60)
        sell_period = data.loc[buy_date:max_sell_date].copy()
        target_reached = sell_period[sell_period['Close'] >= target_price]
        if not target_reached.empty:
            sell_date = target_reached.index[0]
            sell_price = target_reached.loc[sell_date, 'Close']
            sell_reason = "Target reached"
        else:
            sell_date_candidates = sell_period.index.tolist()
            if sell_date_candidates:
                sell_date = sell_date_candidates[-1]
                sell_price = data.loc[sell_date, 'Close']
                sell_reason = "Max holding period"
            else:
                continue
        holding_days = (sell_date - buy_date).days
        profit_pct = (sell_price / buy_price - 1) * 100
        positions.append({
            'BuyDate': buy_date,
            'BuyPrice': buy_price,
            'SellDate': sell_date,
            'SellPrice': sell_price,
            'HoldingDays': holding_days,
            'ProfitPct': profit_pct,
            'SellReason': sell_reason
        })
    return pd.DataFrame(positions)

# =================== UI LAYOUT ===================
st.set_page_config(page_title="Trading Strategy Dashboard", layout="wide")

# --- Top bar navigation as horizontal button group ---
nav_options = ["Price Chart", "Trading Strategy Tester", "Portfolio Analysis"]
nav_cols = st.columns(len(nav_options))
if 'top_page' not in st.session_state:
    st.session_state['top_page'] = nav_options[0]
for i, nav in enumerate(nav_options):
    if nav == st.session_state['top_page']:
        nav_cols[i].markdown(f"<div style='background-color:#e0e0e0; border-radius:6px; padding:8px; text-align:center; font-weight:bold'>{nav}</div>", unsafe_allow_html=True)
    else:
        if nav_cols[i].button(nav, key=f"nav_{nav}"):
            st.session_state['top_page'] = nav
top_page = st.session_state['top_page']

# --- Left sidebar for stock selection ---

st.sidebar.header("Stocks")
stock_options = {
    "Alphabet (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Apple (AAPL)": "AAPL",
    "Meta Platforms (META)": "META",
    "Microsoft (MSFT)": "MSFT",
    "Nvidia (NVDA)": "NVDA",
    "Tesla (TSLA)": "TSLA"
}
selected_stock_name = st.sidebar.radio("Select stock", list(stock_options.keys()), key="stock_select")
selected_ticker = stock_options[selected_stock_name]

# Add backtest period dropdown for strategy tester
backtest_period_map = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "3 Years": "3y",
    "5 Years": "5y"
}
if top_page != "Portfolio Analysis":
    backtest_period_label = st.sidebar.selectbox("Backtest Period (Strategy Tester)", list(backtest_period_map.keys()), key="backtest_period_select")
    backtest_period = backtest_period_map[backtest_period_label]

# --- Download raw data for the selected ticker only ---
@st.cache_data(show_spinner=False)
def load_data(ticker, period):
    data = get_stock_data(ticker, period)
    return data

# --- Price Chart Page ---
if top_page == "Price Chart":
    st.markdown(f"### {selected_stock_name} Price Chart")
    # Timeframe selector as horizontal button group
    timeframe_map = {
        "Month": "1mo",
        "Half Year": "6mo",
        "Year": "1y",
        "5 Year": "5y"
    }
    tf_keys = list(timeframe_map.keys())
    if 'timeframe' not in st.session_state:
        st.session_state['timeframe'] = tf_keys[0]
    tf_cols = st.columns(len(tf_keys))
    for i, tf in enumerate(tf_keys):
        if tf == st.session_state['timeframe']:
            tf_cols[i].markdown(f"<div style='background-color:#e0e0e0; border-radius:6px; padding:8px; text-align:center; font-weight:bold'>{tf}</div>", unsafe_allow_html=True)
        else:
            if tf_cols[i].button(tf, key=f"tf_{tf}"):
                st.session_state['timeframe'] = tf
    timeframe = st.session_state['timeframe']
    period = timeframe_map[timeframe]
    st.session_state['current_period'] = period
    data = load_data(selected_ticker, period)
    # Indicator dropdown in sidebar
    indicator_list = ["None", "Bollinger Bands", "RSI", "MACD", "EMA"]
    if 'chart_indicator' not in st.session_state:
        st.session_state['chart_indicator'] = indicator_list[0]
    chart_indicator = st.sidebar.selectbox("Show Indicator on Chart", indicator_list, key="chart_indicator_select")
    st.session_state['chart_indicator'] = chart_indicator
    # Chart
    fig, ax = plt.subplots(figsize=(14, 5.5))  # Larger chart to fit screen
    ax.plot(data.index, data['Close'], label="Stock Price", color="blue")
    # Show selected indicator with saved parameters
    show_rsi_below = False
    if chart_indicator != "None":
        params = indicator_params[timeframe][chart_indicator]
        if chart_indicator == "Bollinger Bands":
            bb_data = calculate_bollinger_bands(data.copy(), window=params["window"], num_std=params["num_std"])
            ax.plot(bb_data.index, bb_data[bb_data.columns[0]], label=f"MA{params['window']}", color="orange")
            ax.plot(bb_data.index, bb_data['BB_up'], label="BB_up", color="green")
            ax.plot(bb_data.index, bb_data['BB_dn'], label="BB_dn", color="red")
        elif chart_indicator == "RSI":
            show_rsi_below = True
            rsi_data = calculate_rsi(data.copy(), window=params["window"])
        elif chart_indicator == "MACD":
            macd_data = calculate_macd(data.copy(), fast=params["fast"], slow=params["slow"], signal=params["signal"])
            ax2 = ax.twinx()
            ax2.plot(macd_data.index, macd_data['MACD'], label="MACD", color="purple", alpha=0.5)
            ax2.plot(macd_data.index, macd_data['MACD_signal'], label="MACD Signal", color="grey", alpha=0.5)
            ax2.set_ylabel("MACD")
        elif chart_indicator == "EMA":
            params = indicator_params[timeframe]["EMA"]
            ema_data = calculate_ema(data.copy(), window=params["fast"], col_name='EMA_fast')
            ema_data = calculate_ema(ema_data, window=params["slow"], col_name='EMA_slow')
            ax.plot(ema_data.index, ema_data['EMA_fast'], label=f"EMA{params['fast']}", color="orange")
            ax.plot(ema_data.index, ema_data['EMA_slow'], label=f"EMA{params['slow']}", color="red")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    # If RSI is selected, show it as a separate chart below
    if show_rsi_below:
        fig_rsi, ax_rsi = plt.subplots(figsize=(14, 2.5))
        ax_rsi.plot(rsi_data.index, rsi_data['RSI'], label="RSI", color="purple")
        ax_rsi.axhline(params["lower"], color="grey", linestyle="--", alpha=0.5)
        ax_rsi.axhline(params["upper"], color="grey", linestyle="--", alpha=0.5)
        ax_rsi.set_ylabel("RSI")
        ax_rsi.set_xlabel("Date")
        ax_rsi.legend()
        st.pyplot(fig_rsi, use_container_width=True)

# --- Trading Strategy Tester Page ---
elif top_page == "Trading Strategy Tester":
    st.title("Trading Strategy Tester")
    # Strategy selection
    strategy = st.selectbox(
        "Select Trading Strategy",
        ["Bollinger Bands", "RSI", "MACD", "EMA"],
        key="strategy_select"
    )
    # Right sidebar for capital and trade size
    with st.sidebar:
        st.header("Strategy Settings")
        initial_capital = st.number_input("Initial Capital", min_value=1000, value=10000, step=1000, key="init_cap")
        trade_size = st.number_input("Trade Size", min_value=1, value=10, step=1, key="trade_size")
        confirm = st.button("Confirm & Run Backtest", key="confirm_btn")
    # Only run backtest if button is pressed
    if confirm:
        st.session_state['run_backtest'] = True
    # Show results only if flag is set (in main area), then reset flag
    if st.session_state.get('run_backtest', False):
        st.success(f"Backtest started for {selected_stock_name} using {strategy} strategy.")
        data_bt = load_data(selected_ticker, backtest_period)
        if strategy == "Bollinger Bands":
            data_bb = calculate_bollinger_bands(data_bt.copy())
            trades_bt = get_trades_bollinger(data_bb)
        elif strategy == "RSI":
            data_rsi = calculate_rsi(data_bt.copy())
            trades_bt = get_trades_rsi(data_rsi)
        elif strategy == "MACD":
            data_macd = calculate_macd(data_bt.copy())
            trades_bt = get_trades_macd(data_macd)
        elif strategy == "EMA":
            data_ema = calculate_ema(data_bt.copy())
            trades_bt = get_trades_ema(data_ema)
        else:
            trades_bt = pd.DataFrame()
        capital_curve = [initial_capital]
        buy_points = []
        sell_points = []
        if not trades_bt.empty:
            for _, row in trades_bt.iterrows():
                # Calculate P&L for this trade using trade size
                shares = trade_size
                buy_cost = row['BuyPrice'] * shares
                sell_value = row['SellPrice'] * shares
                pnl = sell_value - buy_cost
                capital_curve.append(capital_curve[-1] + pnl)
            capital_curve = np.array(capital_curve)
            steps = len(capital_curve)
            # Buy/sell markers for equity curve (clip to valid range)
            for idx, row in trades_bt.iterrows():
                buy_idx = idx + 1
                sell_idx = idx + 2
                if buy_idx < steps:
                    buy_points.append(buy_idx)
                if sell_idx < steps:
                    sell_points.append(sell_idx)
        else:
            steps = 1
            capital_curve = np.array([initial_capital])
        # Calculate Sharpe Ratio and CAGR
        if steps > 1:
            returns = np.diff(capital_curve) / capital_curve[:-1]
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else float('nan')
            years = (steps-1) / 252  # Approximate trading days
            cagr = (capital_curve[-1] / capital_curve[0]) ** (1/years) - 1 if years > 0 else float('nan')
        else:
            sharpe = float('nan')
            cagr = float('nan')
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(range(steps), capital_curve, label="Equity Curve")
        # Plot buy/sell markers
        if buy_points:
            ax2.scatter(buy_points, capital_curve[buy_points], marker='^', color='green', label='Buy', zorder=5)
        if sell_points:
            ax2.scatter(sell_points, capital_curve[sell_points], marker='v', color='red', label='Sell', zorder=5)
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Capital")
        ax2.legend()
        st.pyplot(fig2)
        st.subheader("Backtest Statistics")
        st.write(f"Current stock backtesting: {selected_stock_name} ({selected_ticker})")
        st.write(f"Current Strategy backtesting: {strategy}")
        if not trades_bt.empty:
            total_pnl = capital_curve[-1] - initial_capital
            max_drawdown = np.min(capital_curve) - initial_capital
            total_trades = len(trades_bt)
            profitable_trades = (trades_bt['ProfitPct'] > 0).sum()
            profit_factor = trades_bt[trades_bt['ProfitPct'] > 0]['ProfitPct'].sum() / abs(trades_bt[trades_bt['ProfitPct'] <= 0]['ProfitPct'].sum()) if (trades_bt[trades_bt['ProfitPct'] <= 0]['ProfitPct'].sum()) != 0 else float('inf')
            st.write(f"Total Profit and Loss: ${total_pnl:.2f}")
            st.write(f"Maximum Equity Drawdown: ${max_drawdown:.2f}")
            st.write(f"Total Trades: {total_trades}")
            st.write(f"Profitable Trades: {profitable_trades}")
            st.write(f"Profit Factor: {profit_factor:.2f}")
            st.write(f"Sharpe Ratio: {sharpe:.2f}")
            st.write(f"CAGR: {cagr:.2%}")
            # Download CSV button
            csv = trades_bt.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Trade Results as CSV",
                data=csv,
                file_name=f"{selected_ticker}_{strategy}_trades.csv",
                mime='text/csv'
            )
        else:
            st.write(f"Total Profit and Loss: $0.00")
            st.write(f"Maximum Equity Drawdown: $0.00")
            st.write(f"Total Trades: 0")
            st.write(f"Profitable Trades: 0")
            st.write(f"Profit Factor: N/A")
            st.write(f"Sharpe Ratio: N/A")
            st.write(f"CAGR: N/A")
        st.subheader(f"Trade List ({strategy})")
        if not trades_bt.empty:
            st.dataframe(trades_bt)
        else:
            st.info("No trades for this strategy and period.")
        # Reset flag so results only show after pressing Confirm
        st.session_state['run_backtest'] = False

# --- Portfolio Analysis Page ---
elif top_page == "Portfolio Analysis":
    st.title("Portfolio Analysis")
    st.markdown("#### Analyze multiple stocks together: combined returns, correlations, and risk metrics.")
    # Multi-select stocks and analysis period in sidebar
    portfolio_stock_names = st.sidebar.multiselect(
        "Select stocks for portfolio analysis",
        list(stock_options.keys()),
        default=[list(stock_options.keys())[0], list(stock_options.keys())[1]]
    )
    portfolio_period_label = st.sidebar.selectbox("Portfolio Analysis Period", list(backtest_period_map.keys()), key="portfolio_period_select")
    confirm_portfolio = st.sidebar.button("Confirm & Analyze Portfolio", key="confirm_portfolio_btn")
    portfolio_tickers = [stock_options[name] for name in portfolio_stock_names]
    portfolio_period = backtest_period_map[portfolio_period_label]
    if confirm_portfolio:
        st.session_state['run_portfolio_analysis'] = True
    if st.session_state.get('run_portfolio_analysis', False):
        if len(portfolio_tickers) < 2:
            st.warning("Select at least two stocks for portfolio analysis.")
        else:
            # Download data for all selected stocks
            price_data = {}
            for ticker in portfolio_tickers:
                price_data[ticker] = load_data(ticker, portfolio_period)["Close"]
            # Combine into DataFrame
            df_prices = pd.DataFrame(price_data)
            df_returns = df_prices.pct_change().dropna()
            # Portfolio: equal-weighted
            portfolio_returns = df_returns.mean(axis=1)
            cumulative_portfolio = (1 + portfolio_returns).cumprod()
            # Correlation matrix
            corr_matrix = df_returns.corr()
            # Risk metrics
            mean_return = portfolio_returns.mean()
            volatility = portfolio_returns.std()
            sharpe_ratio = mean_return / volatility if volatility != 0 else float('nan')
            # Show metrics
            st.subheader("Portfolio Cumulative Return")
            fig_port, ax_port = plt.subplots(figsize=(10, 4))
            ax_port.plot(cumulative_portfolio.index, cumulative_portfolio, label="Portfolio")
            ax_port.set_ylabel("Cumulative Return")
            ax_port.set_xlabel("Date")
            ax_port.legend()
            st.pyplot(fig_port)
            st.subheader("Correlation Matrix")
            st.dataframe(corr_matrix)
            st.subheader("Portfolio Risk Metrics")
            st.write(f"Mean Daily Return: {mean_return:.5f}")
            st.write(f"Daily Volatility: {volatility:.5f}")
            st.write(f"Sharpe Ratio (daily): {sharpe_ratio:.2f}")
        # Reset flag so results only show after pressing Confirm
        st.session_state['run_portfolio_analysis'] = False

