import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import math

# plotly import (선택적)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    st.warning("⚠️ plotly가 설치되지 않았습니다. 차트 기능이 비활성화됩니다.")


# 포트폴리오 구성 (티커: 비중)
PORTFOLIO = {
    "QQQM": 0.15,
    "SPY": 0.20,
    "JEPQ": 0.10,
    "BRK-B": 0.15,
    "IEF": 0.15,
    "TLT": 0.10,
    "GLD": 0.10,
    "PDBC": 0.05
}

# yfinance에서 사용할 티커 리스트
# BRK-B는 yfinance에서 "BRK-B" 또는 "BRK.B" 둘 다 사용 가능
TICKERS = ["QQQM", "SPY", "JEPQ", "BRK-B", "IEF", "TLT", "GLD", "PDBC"]
TICKER_MAPPING = {
    "BRK-B": "BRK-B",  # 표시용 이름 (동일)
    "BRK.B": "BRK-B"   # 대체 티커 매핑
}


def get_current_prices(tickers):
    """현재 가격 조회 (장중 가격 우선)"""
    prices = {}
    for ticker in tickers:
        price = None
        
        # BRK-B는 여러 티커 형식으로 시도
        if ticker == "BRK-B":
            alt_tickers = ["BRK-B", "BRK.B"]
            for alt_ticker in alt_tickers:
                try:
                    t = yf.Ticker(alt_ticker)
                    # 1) 장중 가격(fast_info) 우선 조회
                    try:
                        price = t.fast_info.get("last_price")
                    except:
                        pass
                    
                    # 2) fast_info 실패 시 history 사용 (최근 종가)
                    if price is None or price == 0:
                        hist = t.history(period="1d")
                        if not hist.empty:
                            price = hist["Close"].iloc[-1]
                    
                    if price and price > 0:
                        break
                except Exception as e:
                    continue
            
            if price is None or price == 0:
                st.warning(f"BRK-B 가격 조회 실패 (BRK-B 및 BRK.B 모두 시도했으나 실패)")
        else:
            # 다른 티커는 일반 방식
            try:
                t = yf.Ticker(ticker)
                # 1) 장중 가격(fast_info) 우선 조회
                price = t.fast_info.get("last_price")
                
                # 2) fast_info 실패 시 history 사용 (최근 종가)
                if price is None or price == 0:
                    hist = t.history(period="1d")
                    if not hist.empty:
                        price = hist["Close"].iloc[-1]
            except Exception as e:
                st.warning(f"{ticker} 가격 조회 실패: {e}")
        
        prices[ticker] = price
    return prices


def calculate_target_shares(total_balance, prices):
    """목표 주식 수 계산"""
    target_shares = {}
    for ticker, allocation in PORTFOLIO.items():
        # yfinance 티커는 그대로 사용 (BRK-B는 BRK-B로 조회)
        yf_ticker = ticker
        price = prices.get(yf_ticker)
        
        if price and price > 0:
            target_value = total_balance * allocation
            shares = target_value / price
            target_shares[ticker] = {
                "target_value": target_value,
                "target_shares": shares,
                "current_price": price
            }
        else:
            target_shares[ticker] = {
                "target_value": total_balance * allocation,
                "target_shares": None,
                "current_price": None
            }
    return target_shares


def calculate_rebalancing(target_shares, current_holdings, prices):
    """리밸런싱 계산"""
    rebalancing = {}
    
    for ticker, target_data in target_shares.items():
        current_shares = current_holdings.get(ticker, 0)
        target_shares_count = target_data["target_shares"]
        price = target_data["current_price"]
        
        if target_shares_count is not None and price is not None:
            shares_diff = target_shares_count - current_shares
            value_diff = shares_diff * price
            
            rebalancing[ticker] = {
                "current_shares": current_shares,
                "target_shares": target_shares_count,
                "shares_to_buy": max(0, shares_diff) if shares_diff > 0 else 0,
                "shares_to_sell": abs(min(0, shares_diff)) if shares_diff < 0 else 0,
                "value_to_buy": max(0, value_diff) if value_diff > 0 else 0,
                "value_to_sell": abs(min(0, value_diff)) if value_diff < 0 else 0,
                "current_value": current_shares * price,
                "target_value": target_data["target_value"],
                "current_price": price
            }
        else:
            rebalancing[ticker] = {
                "current_shares": current_holdings.get(ticker, 0),
                "target_shares": None,
                "shares_to_buy": None,
                "shares_to_sell": None,
                "value_to_buy": None,
                "value_to_sell": None,
                "current_value": None,
                "target_value": target_data["target_value"],
                "current_price": None
            }
    
    return rebalancing


def display_portfolio_table(rebalancing):
    """포트폴리오 테이블 표시"""
    data = []
    
    for ticker, data_dict in rebalancing.items():
        row = {
            "티커": ticker,
            "목표 비중": f"{PORTFOLIO[ticker]*100:.1f}%",
            "현재 가격": f"${data_dict['current_price']:,.2f}" if data_dict['current_price'] else "N/A",
            "목표 주식 수": f"{data_dict['target_shares']:.2f}" if data_dict['target_shares'] else "N/A",
            "현재 보유 수": f"{data_dict['current_shares']:.2f}",
            "구매 필요": f"{data_dict['shares_to_buy']:.2f}" if data_dict['shares_to_buy'] is not None else "N/A",
            "매도 필요": f"{data_dict['shares_to_sell']:.2f}" if data_dict['shares_to_sell'] is not None else "N/A",
            "목표 평가액": f"${data_dict['target_value']:,.2f}",
            "현재 평가액": f"${data_dict['current_value']:,.2f}" if data_dict['current_value'] is not None else "N/A",
            "구매 금액": f"${data_dict['value_to_buy']:,.2f}" if data_dict['value_to_buy'] is not None else "N/A",
            "매도 금액": f"${data_dict['value_to_sell']:,.2f}" if data_dict['value_to_sell'] is not None else "N/A"
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


# ==== 백테스트 함수들 ====

def run_portfolio_backtest(portfolio_weights, start_date="2020-01-01", end_date=None):
    """
    포트폴리오 백테스트 실행 (월별 리밸런싱)
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # 티커 변환 (BRK-B -> BRK.B for yfinance)
    tickers_for_download = []
    ticker_mapping = {}
    for ticker in portfolio_weights.keys():
        if ticker == "BRK-B":
            tickers_for_download.append("BRK.B")
            ticker_mapping["BRK.B"] = "BRK-B"
        else:
            tickers_for_download.append(ticker)
            ticker_mapping[ticker] = ticker
    
    # 데이터 다운로드
    with st.spinner("과거 데이터를 다운로드하는 중..."):
        data = yf.download(tickers_for_download, start=start_date, end=end_date, progress=False)["Adj Close"]
        data.index = data.index.tz_localize(None)
    
    # 티커 이름 매핑 복원
    data.columns = [ticker_mapping.get(col, col) for col in data.columns]
    
    # 월별 리밸런싱
    monthly_data = data.resample("M").last().dropna()
    
    if len(monthly_data) < 2:
        return None, None, None
    
    # 포트폴리오 가치 계산 (시작값 = 100)
    portfolio_value = pd.Series(index=monthly_data.index, dtype=float)
    portfolio_value.iloc[0] = 100.0
    
    for i in range(1, len(monthly_data)):
        prev_value = portfolio_value.iloc[i-1]
        
        # 각 자산의 월간 수익률 계산
        monthly_returns = {}
        for ticker in portfolio_weights.keys():
            if ticker in monthly_data.columns:
                prev_price = monthly_data.iloc[i-1][ticker]
                curr_price = monthly_data.iloc[i][ticker]
                if not pd.isna(prev_price) and not pd.isna(curr_price) and prev_price > 0:
                    monthly_returns[ticker] = (curr_price / prev_price) - 1
                else:
                    monthly_returns[ticker] = 0
        
        # 포트폴리오 수익률 = 가중 평균
        portfolio_return = 0
        for ticker, weight in portfolio_weights.items():
            if ticker in monthly_returns:
                portfolio_return += weight * monthly_returns[ticker]
        
        portfolio_value.iloc[i] = prev_value * (1 + portfolio_return)
    
    return portfolio_value, monthly_data, start_date


def calculate_performance_metrics(portfolio_value):
    """성과 지표 계산"""
    if portfolio_value is None or len(portfolio_value) < 2:
        return None
    
    # 일별 수익률 근사 (월별 데이터를 일별로 보간)
    # 월별 데이터를 일별로 확장하여 계산
    daily_index = pd.date_range(start=portfolio_value.index[0], end=portfolio_value.index[-1], freq='D')
    daily_value = portfolio_value.reindex(daily_index).interpolate(method='linear')
    daily_returns = daily_value.pct_change().dropna()
    
    # 기간 계산
    years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
    
    # CAGR
    total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
    cagr = ((1 + total_return) ** (1 / years) - 1) if years > 0 else 0
    
    # 연환산 표준편차
    annual_vol = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
    
    # MDD (최대 낙폭)
    cumulative = portfolio_value
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    mdd = drawdown.min()
    
    # 샤프지수 (무위험 수익률 0% 가정)
    sharpe = (cagr / annual_vol) if annual_vol > 0 else 0
    
    return {
        "CAGR": cagr * 100,
        "연환산 표준편차": annual_vol * 100,
        "MDD": mdd * 100,
        "샤프지수": sharpe,
        "총 수익률": total_return * 100,
        "기간(년)": years,
        "시작일": portfolio_value.index[0].strftime('%Y-%m-%d'),
        "종료일": portfolio_value.index[-1].strftime('%Y-%m-%d')
    }


def calculate_yearly_returns(portfolio_value):
    """연도별 수익률 계산"""
    if portfolio_value is None or len(portfolio_value) < 2:
        return None
    
    yearly = portfolio_value.resample("YE").last()
    yearly_returns = yearly.pct_change().dropna() * 100
    return yearly_returns


def calculate_monthly_returns(portfolio_value):
    """월별 수익률 계산"""
    if portfolio_value is None or len(portfolio_value) < 2:
        return None
    
    monthly_returns = portfolio_value.pct_change().dropna() * 100
    return monthly_returns


# ==== Streamlit 앱 메인 ====
st.set_page_config(
    page_title="포트폴리오 리밸런싱 계산기",
    page_icon="📊",
    layout="wide"
)

st.title("📊 포트폴리오 리밸런싱 계산기")
st.markdown("---")

# 포트폴리오 구성 표시
with st.expander("📋 포트폴리오 구성", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    portfolio_items = list(PORTFOLIO.items())
    for i, (ticker, allocation) in enumerate(portfolio_items):
        with col1 if i < 2 else col2 if i < 4 else col3 if i < 6 else col4:
            st.metric(ticker, f"{allocation*100:.1f}%")

st.markdown("---")

# 사이드바에 입력 필드
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 자동 계산 모드 설정
    auto_calculate = st.checkbox(
        "🔄 자동 계산 모드",
        value=False,
        help="총 금액 또는 보유 수량 입력 시 자동으로 계산합니다."
    )
    
    st.markdown("---")
    
    # 총 금액 입력 (자동 계산 모드일 때 on_change 추가)
    total_balance = st.number_input(
        "총 금액 (평가금 + 예수금)",
        min_value=0.0,
        value=10000.0,
        step=1000.0,
        format="%.2f",
        help="보유하고 있는 총 자산 금액을 입력하세요.",
        key="total_balance_input",
        on_change=lambda: st.session_state.update({'auto_calc_trigger': True}) if auto_calculate else None
    )
    
    st.markdown("---")
    st.subheader("📦 현재 보유 주식 수")
    
    current_holdings = {}
    for ticker in PORTFOLIO.keys():
        current_holdings[ticker] = st.number_input(
            f"{ticker} 보유 수량",
            min_value=0.0,
            value=0.0,
            step=0.01,
            format="%.2f",
            key=f"holding_{ticker}",
            on_change=lambda: st.session_state.update({'auto_calc_trigger': True}) if auto_calculate else None
        )
    
    st.markdown("---")
    
    # 자동 계산 모드일 때 자동 계산
    if auto_calculate and total_balance > 0:
        # 총 금액이나 보유 수량이 변경되었거나, 아직 계산되지 않은 경우
        if 'auto_calc_trigger' in st.session_state or 'calculate' not in st.session_state:
            st.session_state['total_balance'] = total_balance
            st.session_state['current_holdings'] = current_holdings
            st.session_state['calculate'] = True
            if 'auto_calc_trigger' in st.session_state:
                del st.session_state['auto_calc_trigger']
    
    if st.button("🚀 계산하기", type="primary", use_container_width=True):
        if total_balance <= 0:
            st.error("총 금액은 0보다 커야 합니다.")
        else:
            st.session_state['total_balance'] = total_balance
            st.session_state['current_holdings'] = current_holdings
            st.session_state['calculate'] = True
    
    if st.button("🔄 초기화", use_container_width=True):
        # 모든 계산 관련 세션 상태 초기화
        keys_to_remove = [
            'calculate', 
            'total_balance', 
            'current_holdings',
            'auto_calc_trigger',
            'backtest_results'
        ]
        for key in keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    # ==== 사이드바에 설정 정보 표시 ====
    if st.session_state.get('calculate', False):
        st.markdown("---")
        st.subheader("📊 설정 정보")
        current_date = datetime.now()
        st.metric("기준 날짜", current_date.strftime('%Y-%m-%d'))
        st.metric("총 자산", f"${st.session_state.get('total_balance', 0):,.2f}")

# 메인 영역에 결과 표시
if st.session_state.get('calculate', False):
    total_balance = st.session_state.get('total_balance', 0)
    current_holdings = st.session_state.get('current_holdings', {})
    current_date = datetime.now()
    
    with st.spinner("현재 가격을 조회하는 중..."):
        prices = get_current_prices(TICKERS)
    
    # 목표 주식 수 계산
    target_shares = calculate_target_shares(total_balance, prices)
    
    # 리밸런싱 계산
    rebalancing = calculate_rebalancing(target_shares, current_holdings, prices)
    
    # ==== 기준 날짜 및 설정 정보 표시 ====
    st.subheader("📊 설정 정보")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("기준 날짜", current_date.strftime('%Y-%m-%d'))
    with col2:
        st.metric("총 자산", f"${total_balance:,.2f}")
    
    st.markdown("---")
    
    # 요약 정보
    st.subheader("📈 요약 정보")
    col1, col2, col3 = st.columns(3)
    
    total_target_value = sum([data["target_value"] for data in target_shares.values()])
    total_current_value = sum([
        data["current_value"] for data in rebalancing.values() 
        if data["current_value"] is not None
    ])
    total_buy_value = sum([
        data["value_to_buy"] for data in rebalancing.values() 
        if data["value_to_buy"] is not None
    ])
    total_sell_value = sum([
        data["value_to_sell"] for data in rebalancing.values() 
        if data["value_to_sell"] is not None
    ])
    
    with col1:
        st.metric("목표 평가액 합계", f"${total_target_value:,.2f}")
        st.metric("현재 평가액 합계", f"${total_current_value:,.2f}" if total_current_value else "N/A")
    
    with col2:
        st.metric("총 구매 필요 금액", f"${total_buy_value:,.2f}" if total_buy_value else "$0.00")
        st.metric("총 매도 필요 금액", f"${total_sell_value:,.2f}" if total_sell_value else "$0.00")
    
    with col3:
        net_rebalance = total_buy_value - total_sell_value
        st.metric("순 리밸런싱 금액", f"${net_rebalance:,.2f}" if net_rebalance else "$0.00")
        # 현재 비중 vs 목표 비중 편차 계산
        if total_current_value and total_current_value > 0:
            deviation = ((total_current_value - total_target_value) / total_target_value) * 100
            st.metric("비중 편차", f"{deviation:+.2f}%")
    
    st.markdown("---")
    
    # ==== 현재 비중 vs 목표 비중 비교 ====
    st.subheader("📊 현재 비중 vs 목표 비중 비교")
    comparison_data = []
    priority_data = []  # 우선순위용 데이터
    
    for ticker in PORTFOLIO.keys():
        target_weight = PORTFOLIO[ticker] * 100
        current_value = rebalancing[ticker].get("current_value", 0)
        current_weight = (current_value / total_current_value * 100) if total_current_value and total_current_value > 0 else 0
        weight_diff = current_weight - target_weight
        abs_weight_diff = abs(weight_diff)
        
        comparison_data.append({
            "티커": ticker,
            "목표 비중": f"{target_weight:.1f}%",
            "현재 비중": f"{current_weight:.1f}%" if current_value else "0.0%",
            "편차": f"{weight_diff:+.1f}%",
            "상태": "✅" if abs_weight_diff < 1 else ("⬆️" if weight_diff > 0 else "⬇️")
        })
        
        # 우선순위용 데이터 (편차가 큰 순서)
        priority_data.append({
            "티커": ticker,
            "목표 비중": target_weight,
            "현재 비중": current_weight,
            "편차": weight_diff,
            "절대 편차": abs_weight_diff,
            "구매 필요": rebalancing[ticker].get("value_to_buy", 0) or 0,
            "매도 필요": rebalancing[ticker].get("value_to_sell", 0) or 0
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # ==== 포트폴리오 비중 차트 ====
    if HAS_PLOTLY:
        st.markdown("---")
        st.subheader("📊 포트폴리오 비중 비교 차트")
        
        chart_data = []
        for ticker in PORTFOLIO.keys():
            target_weight = PORTFOLIO[ticker] * 100
            current_value = rebalancing[ticker].get("current_value", 0)
            current_weight = (current_value / total_current_value * 100) if total_current_value and total_current_value > 0 else 0
            
            chart_data.append({
                "티커": ticker,
                "목표 비중": target_weight,
                "현재 비중": current_weight
            })
        
        chart_df = pd.DataFrame(chart_data)
        
        # 막대 차트 생성
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='목표 비중',
            x=chart_df['티커'],
            y=chart_df['목표 비중'],
            marker_color='lightblue',
            text=chart_df['목표 비중'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='현재 비중',
            x=chart_df['티커'],
            y=chart_df['현재 비중'],
            marker_color='lightcoral',
            text=chart_df['현재 비중'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside'
        ))
        
        fig.update_layout(
            title="포트폴리오 비중 비교",
            xaxis_title="티커",
            yaxis_title="비중 (%)",
            barmode='group',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 차트 기능을 사용하려면 plotly 패키지가 필요합니다. `pip install plotly`로 설치해주세요.")
    
    # ==== 리밸런싱 우선순위 표시 ====
    st.markdown("---")
    st.subheader("🎯 리밸런싱 우선순위 (편차 큰 순서)")
    
    priority_df = pd.DataFrame(priority_data)
    priority_df = priority_df.sort_values('절대 편차', ascending=False)
    
    priority_display = []
    for _, row in priority_df.iterrows():
        if row['절대 편차'] > 0.1:  # 편차가 0.1% 이상인 것만 표시
            action = "구매" if row['구매 필요'] > 0 else ("매도" if row['매도 필요'] > 0 else "유지")
            priority_display.append({
                "순위": len(priority_display) + 1,
                "티커": row['티커'],
                "편차": f"{row['편차']:+.1f}%",
                "액션": action,
                "금액": f"${max(row['구매 필요'], row['매도 필요']):,.2f}" if max(row['구매 필요'], row['매도 필요']) > 0 else "-"
            })
    
    if priority_display:
        priority_display_df = pd.DataFrame(priority_display)
        st.dataframe(priority_display_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ 모든 자산이 목표 비중에 근접해 있습니다!")
    
    st.markdown("---")
    
    # 상세 테이블
    st.subheader("📊 상세 리밸런싱 정보")
    df = display_portfolio_table(rebalancing)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # CSV 다운로드
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 CSV로 다운로드",
        data=csv,
        file_name=f"portfolio_rebalancing_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    # 리밸런싱 필요 항목만 필터링
    st.markdown("---")
    st.subheader("🔄 리밸런싱 필요 항목")
    
    needs_rebalancing = []
    for ticker, data in rebalancing.items():
        if data["shares_to_buy"] and data["shares_to_buy"] > 0.01:
            needs_rebalancing.append({
                "티커": ticker,
                "액션": "구매",
                "수량": f"{data['shares_to_buy']:.2f}",
                "금액": f"${data['value_to_buy']:,.2f}"
            })
        if data["shares_to_sell"] and data["shares_to_sell"] > 0.01:
            needs_rebalancing.append({
                "티커": ticker,
                "액션": "매도",
                "수량": f"{data['shares_to_sell']:.2f}",
                "금액": f"${data['value_to_sell']:,.2f}"
            })
    
    if needs_rebalancing:
        rebalancing_df = pd.DataFrame(needs_rebalancing)
        st.dataframe(rebalancing_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ 모든 포트폴리오가 목표 비중에 맞게 구성되어 있습니다!")
    
    # ==== 포트폴리오 백테스트 ====
    st.markdown("---")
    st.subheader("📊 포트폴리오 효용성 분석")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date_input = st.date_input(
            "백테스트 시작일",
            value=datetime(2020, 1, 1).date(),
            min_value=datetime(2010, 1, 1).date(),
            max_value=datetime.now().date()
        )
    
    with col2:
        if st.button("🔍 백테스트 실행", type="primary", use_container_width=True):
            with st.spinner("백테스트를 실행하는 중..."):
                portfolio_value, monthly_data, start_date = run_portfolio_backtest(
                    PORTFOLIO, 
                    start_date=start_date_input.strftime("%Y-%m-%d")
                )
                
                if portfolio_value is not None:
                    metrics = calculate_performance_metrics(portfolio_value)
                    yearly_returns = calculate_yearly_returns(portfolio_value)
                    monthly_returns = calculate_monthly_returns(portfolio_value)
                    
                    st.session_state['backtest_results'] = {
                        'metrics': metrics,
                        'yearly_returns': yearly_returns,
                        'monthly_returns': monthly_returns,
                        'portfolio_value': portfolio_value
                    }
                else:
                    st.error("백테스트 실행 실패: 데이터가 부족합니다.")
    
    # 백테스트 결과 표시
    if 'backtest_results' in st.session_state:
        results = st.session_state['backtest_results']
        
        if results['metrics']:
            # 성과 지표 표시
            st.markdown("---")
            st.subheader("📈 성과 지표")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("CAGR", f"{results['metrics']['CAGR']:.2f}%")
            with col2:
                st.metric("연환산 표준편차", f"{results['metrics']['연환산 표준편차']:.2f}%")
            with col3:
                st.metric("MDD", f"{results['metrics']['MDD']:.2f}%")
            with col4:
                st.metric("샤프지수", f"{results['metrics']['샤프지수']:.2f}")
            with col5:
                st.metric("총 수익률", f"{results['metrics']['총 수익률']:.2f}%")
            
            # 기간 정보
            st.caption(f"기간: {results['metrics']['시작일']} ~ {results['metrics']['종료일']} ({results['metrics']['기간(년)']:.2f}년)")
            
            # 연도별 수익률 표
            if results['yearly_returns'] is not None and len(results['yearly_returns']) > 0:
                st.markdown("---")
                st.subheader("📅 연도별 수익률")
                yearly_df = results['yearly_returns'].to_frame("수익률 (%)")
                yearly_df.index = yearly_df.index.year
                yearly_df = yearly_df.round(2)
                st.dataframe(yearly_df, use_container_width=True, height=300)
            
            # 월별 수익률 표
            if results['monthly_returns'] is not None and len(results['monthly_returns']) > 0:
                st.markdown("---")
                st.subheader("📅 월별 수익률")
                monthly_df = results['monthly_returns'].to_frame("수익률 (%)")
                monthly_df.index = monthly_df.index.strftime("%Y-%m")
                monthly_df = monthly_df.round(2)
                st.dataframe(monthly_df, use_container_width=True, height=400)
            
            # 포트폴리오 가치 차트
            if HAS_PLOTLY and results['portfolio_value'] is not None:
                st.markdown("---")
                st.subheader("📈 포트폴리오 가치 추이")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=results['portfolio_value'].index,
                    y=results['portfolio_value'].values,
                    mode='lines',
                    name='포트폴리오 가치',
                    line=dict(color='#1f77b4', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(31, 119, 180, 0.1)'
                ))
                fig.update_layout(
                    title="포트폴리오 가치 추이 (시작값 = 100)",
                    xaxis_title="날짜",
                    yaxis_title="포트폴리오 가치",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
        
else:
    st.info("👈 왼쪽 사이드바에서 총 금액과 현재 보유 주식 수를 입력하고 '계산하기' 버튼을 클릭하세요.")
    
    # 예시 표시
    st.markdown("### 💡 사용 예시")
    st.markdown("""
    1. **총 금액 입력**: 보유하고 있는 총 자산 금액(평가금 + 예수금)을 입력합니다.
    2. **현재 보유 수량 입력**: 각 자산별로 현재 보유하고 있는 주식 수를 입력합니다.
    3. **계산하기 클릭**: 목표 비중에 맞춰 필요한 주식 수와 리밸런싱 정보를 확인합니다.
    
    **포트폴리오 구성:**
    - QQQM: 15%
    - SPY: 20%
    - JEPQ: 10%
    - BRK-B: 15%
    - IEF: 15%
    - TLT: 10%
    - GLD: 10%
    - PDBC: 5%
    """)

