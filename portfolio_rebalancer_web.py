import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime


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
    
    total_balance = st.number_input(
        "총 금액 (평가금 + 예수금)",
        min_value=0.0,
        value=10000.0,
        step=1000.0,
        format="%.2f",
        help="보유하고 있는 총 자산 금액을 입력하세요."
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
            key=f"holding_{ticker}"
        )
    
    st.markdown("---")
    
    if st.button("🚀 계산하기", type="primary", use_container_width=True):
        if total_balance <= 0:
            st.error("총 금액은 0보다 커야 합니다.")
        else:
            st.session_state['total_balance'] = total_balance
            st.session_state['current_holdings'] = current_holdings
            st.session_state['calculate'] = True
    
    if st.button("🔄 초기화", use_container_width=True):
        if 'calculate' in st.session_state:
            del st.session_state['calculate']
        st.rerun()

# 메인 영역에 결과 표시
if st.session_state.get('calculate', False):
    total_balance = st.session_state.get('total_balance', 0)
    current_holdings = st.session_state.get('current_holdings', {})
    
    with st.spinner("현재 가격을 조회하는 중..."):
        prices = get_current_prices(TICKERS)
    
    # 목표 주식 수 계산
    target_shares = calculate_target_shares(total_balance, prices)
    
    # 리밸런싱 계산
    rebalancing = calculate_rebalancing(target_shares, current_holdings, prices)
    
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
        st.metric("총 자산", f"${total_balance:,.2f}")
        st.metric("목표 평가액 합계", f"${total_target_value:,.2f}")
    
    with col2:
        st.metric("현재 평가액 합계", f"${total_current_value:,.2f}" if total_current_value else "N/A")
        st.metric("총 구매 필요 금액", f"${total_buy_value:,.2f}" if total_buy_value else "$0.00")
    
    with col3:
        st.metric("총 매도 필요 금액", f"${total_sell_value:,.2f}" if total_sell_value else "$0.00")
        net_rebalance = total_buy_value - total_sell_value
        st.metric("순 리밸런싱 금액", f"${net_rebalance:,.2f}" if net_rebalance else "$0.00")
    
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

