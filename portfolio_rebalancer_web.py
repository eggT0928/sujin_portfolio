import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
# SPY와 SPYM은 합산하여 20% 비중을 차지
PORTFOLIO = {
    "QQQM": 0.15,
    "SPY+SPYM": 0.20,  # SPY와 SPYM 합산 비중
    "JEPQ": 0.10,
    "BRK-B": 0.15,
    "IEF": 0.15,
    "TLT": 0.10,
    "GLD": 0.10,
    "PDBC": 0.05
}

# S&P 500 관련 티커 (SPY와 SPYM 합산)
SNP_TICKERS = ["SPY", "SPYM"]

# yfinance에서 사용할 티커 리스트
# BRK-B는 yfinance에서 "BRK-B" 또는 "BRK.B" 둘 다 사용 가능
TICKERS = ["QQQM", "SPY", "SPYM", "JEPQ", "BRK-B", "IEF", "TLT", "GLD", "PDBC"]
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
        # SPY+SPYM의 경우 특별 처리 - 목표 총 가치만 계산 (개별 목표 주식 수는 나중에 현재 보유 비중 기준으로 계산)
        if ticker == "SPY+SPYM":
            spy_price = prices.get("SPY")
            spym_price = prices.get("SPYM")
            target_value = total_balance * allocation
            
            # 평균 가격 계산 (표시용)
            avg_price = None
            if spy_price and spy_price > 0 and spym_price and spym_price > 0:
                avg_price = (spy_price + spym_price) / 2
            elif spy_price and spy_price > 0:
                avg_price = spy_price
            elif spym_price and spym_price > 0:
                avg_price = spym_price
            
            target_shares["SPY+SPYM"] = {
                "target_value": target_value,  # 목표 총 가치 (SPY + SPYM 합산)
                "target_shares": None,  # 개별 주식 수는 현재 보유 비중 기준으로 나중에 계산
                "current_price": avg_price,
                "spy_target_value": None,  # 나중에 현재 보유 비중 기준으로 계산
                "spy_target_shares": None,  # 나중에 현재 보유 비중 기준으로 계산
                "spy_price": spy_price,
                "spym_target_value": None,  # 나중에 현재 보유 비중 기준으로 계산
                "spym_target_shares": None,  # 나중에 현재 보유 비중 기준으로 계산
                "spym_price": spym_price
            }
        else:
            # 일반 티커 처리
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
        # SPY+SPYM의 경우 특별 처리
        if ticker == "SPY+SPYM":
            spy_current_shares = current_holdings.get("SPY", 0)
            spym_current_shares = current_holdings.get("SPYM", 0)
            
            spy_price = target_data.get("spy_price")
            spym_price = target_data.get("spym_price")
            
            # 현재 가치 계산 (사용자가 입력한 보유 수량 기준)
            spy_current_value = spy_current_shares * spy_price if spy_price and spy_price > 0 else 0
            spym_current_value = spym_current_shares * spym_price if spym_price and spym_price > 0 else 0
            total_current_value = spy_current_value + spym_current_value
            
            # 목표 총 가치 (SPY + SPYM 합산)
            target_total_value = target_data["target_value"]
            
            # SPY는 현재 보유 수량 유지 (수수료 때문에 더 늘리지 않음)
            # SPY 목표 가치 = 현재 SPY 가치 (변경 없음)
            spy_target_value = spy_current_value
            spy_target_shares = spy_current_shares  # 현재 보유 수량 유지
            
            # SPYM 목표 가치 = 목표 총 가치 - SPY 현재 가치
            spym_target_value = target_total_value - spy_target_value
            
            # SPYM 목표 주식 수 계산
            spym_target_shares = spym_target_value / spym_price if spym_price and spym_price > 0 else None
            
            # SPY 리밸런싱 계산 (SPY는 현재 보유 수량 유지, 구매/매도 없음)
            spy_rebalancing = {}
            if spy_price and spy_price > 0:
                # SPY는 현재 보유 수량 유지 (수수료 때문에 더 늘리지 않음)
                spy_rebalancing = {
                    "current_shares": spy_current_shares,
                    "target_shares": spy_target_shares,  # 현재 보유 수량과 동일
                    "shares_to_buy": 0,  # 구매 없음
                    "shares_to_sell": 0,  # 매도 없음
                    "value_to_buy": 0,  # 구매 없음
                    "value_to_sell": 0,  # 매도 없음
                    "current_value": spy_current_value,
                    "target_value": spy_target_value,  # 현재 가치와 동일
                    "current_price": spy_price
                }
            else:
                spy_rebalancing = {
                    "current_shares": spy_current_shares,
                    "target_shares": spy_target_shares,
                    "shares_to_buy": 0,
                    "shares_to_sell": 0,
                    "value_to_buy": 0,
                    "value_to_sell": 0,
                    "current_value": spy_current_value,
                    "target_value": spy_target_value,
                    "current_price": spy_price
                }
            
            # SPYM 리밸런싱 계산
            spym_rebalancing = {}
            if spym_price and spym_price > 0:
                spym_shares_diff = (spym_target_shares - spym_current_shares) if spym_target_shares is not None else 0
                spym_value_diff = spym_shares_diff * spym_price
                spym_rebalancing = {
                    "current_shares": spym_current_shares,
                    "target_shares": spym_target_shares,
                    "shares_to_buy": max(0, spym_shares_diff) if spym_shares_diff > 0 else 0,
                    "shares_to_sell": abs(min(0, spym_shares_diff)) if spym_shares_diff < 0 else 0,
                    "value_to_buy": max(0, spym_value_diff) if spym_value_diff > 0 else 0,
                    "value_to_sell": abs(min(0, spym_value_diff)) if spym_value_diff < 0 else 0,
                    "current_value": spym_current_value,
                    "target_value": spym_target_value,
                    "current_price": spym_price
                }
            else:
                spym_rebalancing = {
                    "current_shares": spym_current_shares,
                    "target_shares": None,
                    "shares_to_buy": None,
                    "shares_to_sell": None,
                    "value_to_buy": None,
                    "value_to_sell": None,
                    "current_value": spym_current_value,
                    "target_value": spym_target_value,
                    "current_price": spym_price
                }
            
            # 합산 정보 저장 (SPY는 구매/매도 없으므로 SPYM만 합산)
            rebalancing["SPY+SPYM"] = {
                "current_shares": None,  # 개별 주식 수는 별도로 관리
                "target_shares": None,
                "shares_to_buy": None,
                "shares_to_sell": None,
                "value_to_buy": spym_rebalancing.get("value_to_buy", 0) or 0,  # SPYM만 구매
                "value_to_sell": spym_rebalancing.get("value_to_sell", 0) or 0,  # SPYM만 매도
                "current_value": total_current_value,
                "target_value": target_total_value,
                "current_price": target_data.get("current_price"),
                "spy": spy_rebalancing,
                "spym": spym_rebalancing
            }
        else:
            # 일반 티커 처리
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


def display_portfolio_table(rebalancing, total_balance=0):
    """포트폴리오 테이블 표시"""
    data = []
    
    for ticker, data_dict in rebalancing.items():
        # SPY+SPYM의 경우 SPY와 SPYM을 각각 표시
        if ticker == "SPY+SPYM":
            spy_data = data_dict.get("spy", {})
            spym_data = data_dict.get("spym", {})
            target_weight = PORTFOLIO[ticker] * 100  # 합산 목표 비중 20%
            
            # SPY 목표 비중 계산 (목표 가치 기준)
            spy_target_value = spy_data.get("target_value", 0) or 0
            spy_target_weight = (spy_target_value / total_balance * 100) if total_balance > 0 else 0
            
            # SPY 행
            spy_row = {
                "티커": "SPY",
                "목표 비중": f"{spy_target_weight:.1f}%",  # 목표 가치 기준 비중
                "현재 가격": f"${spy_data.get('current_price', 0):,.2f}" if spy_data.get('current_price') else "N/A",
                "목표 주식 수": f"{spy_data.get('target_shares', 0):.2f}" if spy_data.get('target_shares') is not None else "N/A",
                "현재 보유 수": f"{spy_data.get('current_shares', 0):.2f}",
                "구매 필요": f"{spy_data.get('shares_to_buy', 0):.2f}" if spy_data.get('shares_to_buy') is not None else "N/A",
                "매도 필요": f"{spy_data.get('shares_to_sell', 0):.2f}" if spy_data.get('shares_to_sell') is not None else "N/A",
                "목표 평가액": f"${spy_data.get('target_value', 0):,.2f}",
                "현재 평가액": f"${spy_data.get('current_value', 0):,.2f}" if spy_data.get('current_value') is not None else "N/A",
                "구매 금액": f"${spy_data.get('value_to_buy', 0):,.2f}" if spy_data.get('value_to_buy') is not None else "N/A",
                "매도 금액": f"${spy_data.get('value_to_sell', 0):,.2f}" if spy_data.get('value_to_sell') is not None else "N/A"
            }
            data.append(spy_row)
            
            # SPYM 목표 비중 계산 (목표 가치 기준)
            spym_target_value = spym_data.get("target_value", 0) or 0
            spym_target_weight = (spym_target_value / total_balance * 100) if total_balance > 0 else 0
            
            # SPYM 행
            spym_row = {
                "티커": "SPYM",
                "목표 비중": f"{spym_target_weight:.1f}%",  # 목표 가치 기준 비중
                "현재 가격": f"${spym_data.get('current_price', 0):,.2f}" if spym_data.get('current_price') else "N/A",
                "목표 주식 수": f"{spym_data.get('target_shares', 0):.2f}" if spym_data.get('target_shares') is not None else "N/A",
                "현재 보유 수": f"{spym_data.get('current_shares', 0):.2f}",
                "구매 필요": f"{spym_data.get('shares_to_buy', 0):.2f}" if spym_data.get('shares_to_buy') is not None else "N/A",
                "매도 필요": f"{spym_data.get('shares_to_sell', 0):.2f}" if spym_data.get('shares_to_sell') is not None else "N/A",
                "목표 평가액": f"${spym_data.get('target_value', 0):,.2f}",
                "현재 평가액": f"${spym_data.get('current_value', 0):,.2f}" if spym_data.get('current_value') is not None else "N/A",
                "구매 금액": f"${spym_data.get('value_to_buy', 0):,.2f}" if spym_data.get('value_to_buy') is not None else "N/A",
                "매도 금액": f"${spym_data.get('value_to_sell', 0):,.2f}" if spym_data.get('value_to_sell') is not None else "N/A"
            }
            data.append(spym_row)
            
            # 합계 행 (선택적)
            total_row = {
                "티커": "SPY+SPYM 합계",
                "목표 비중": f"{target_weight:.1f}%",
                "현재 가격": "-",
                "목표 주식 수": "-",
                "현재 보유 수": "-",
                "구매 필요": "-",
                "매도 필요": "-",
                "목표 평가액": f"${data_dict['target_value']:,.2f}",
                "현재 평가액": f"${data_dict['current_value']:,.2f}" if data_dict['current_value'] is not None else "N/A",
                "구매 금액": f"${data_dict['value_to_buy']:,.2f}" if data_dict['value_to_buy'] is not None else "N/A",
                "매도 금액": f"${data_dict['value_to_sell']:,.2f}" if data_dict['value_to_sell'] is not None else "N/A"
            }
            data.append(total_row)
        else:
            # 일반 티커 처리
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

def convert_portfolio_for_backtest(portfolio_weights):
    """
    백테스트용 포트폴리오 변환 (SPY+SPYM을 SPY와 SPYM으로 분리)
    """
    backtest_weights = {}
    for ticker, weight in portfolio_weights.items():
        if ticker == "SPY+SPYM":
            # SPY와 SPYM을 각각 50%씩 분배
            backtest_weights["SPY"] = weight / 2
            backtest_weights["SPYM"] = weight / 2
        else:
            backtest_weights[ticker] = weight
    return backtest_weights


def get_latest_listing_date(portfolio_weights):
    """
    각 티커의 첫 거래일을 확인하고 가장 늦은 날짜를 반환
    """
    listing_dates = {}
    
    # 백테스트용 포트폴리오로 변환 (SPY+SPYM 분리)
    backtest_weights = convert_portfolio_for_backtest(portfolio_weights)
    
    # 티커 변환 (BRK-B -> BRK.B for yfinance)
    ticker_mapping = {}
    for ticker in backtest_weights.keys():
        if ticker == "BRK-B":
            ticker_mapping["BRK.B"] = "BRK-B"
        else:
            ticker_mapping[ticker] = ticker
    
    for ticker in backtest_weights.keys():
        try:
            # yfinance 티커 변환 (BRK-B는 여러 형식 시도)
            if ticker == "BRK-B":
                yf_tickers = ["BRK.B", "BRK-B"]
            else:
                yf_tickers = [ticker]
            
            first_date = None
            for yf_ticker in yf_tickers:
                try:
                    # 최근 10년 데이터로 첫 거래일 확인
                    ticker_obj = yf.Ticker(yf_ticker)
                    hist = ticker_obj.history(period="10y", interval="1d")
                    
                    if not hist.empty:
                        first_date = hist.index[0].date()
                        listing_dates[ticker] = first_date
                        break  # 성공하면 루프 종료
                except:
                    continue  # 다음 티커 시도
            
        except Exception as e:
            # 실패 시 기본값 사용하지 않고 스킵
            continue
    
    if listing_dates:
        # 가장 늦은 상장일 찾기
        latest_date = max(listing_dates.values())
        # 한 달 여유를 두고 설정
        latest_date = latest_date + timedelta(days=30)
        return latest_date
    else:
        # 모든 티커 조회 실패 시 기본값
        return datetime(2022, 1, 1).date()


def run_portfolio_backtest(portfolio_weights, start_date="2020-01-01", end_date=None):
    """
    포트폴리오 백테스트 실행 (월별 리밸런싱)
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # 백테스트용 포트폴리오로 변환 (SPY+SPYM 분리)
    backtest_weights = convert_portfolio_for_backtest(portfolio_weights)
    
    # 티커 변환 및 개별 다운로드 (BRK-B는 BRK.B로 시도)
    ticker_data_frames = []
    ticker_mapping = {}
    
    for ticker in backtest_weights.keys():
        if ticker == "BRK-B":
            # BRK-B는 BRK.B로 시도
            yf_tickers = ["BRK.B", "BRK-B"]
        else:
            yf_tickers = [ticker]
        
        ticker_downloaded = False
        for yf_ticker in yf_tickers:
            try:
                ticker_obj = yf.Ticker(yf_ticker)
                hist = ticker_obj.history(start=start_date, end=end_date, auto_adjust=False)
                
                if not hist.empty and "Adj Close" in hist.columns:
                    adj_close = hist["Adj Close"].rename(ticker)
                    ticker_data_frames.append(adj_close)
                    ticker_mapping[ticker] = ticker
                    ticker_downloaded = True
                    break
            except:
                continue
        
        if not ticker_downloaded:
            st.warning(f"{ticker} 데이터 다운로드 실패")
    
    # 데이터 다운로드
    data = None
    try:
        with st.spinner("과거 데이터를 다운로드하는 중..."):
            if len(ticker_data_frames) == 0:
                st.error("다운로드된 티커 데이터가 없습니다.")
                return None, None, None
            
            # 모든 티커 데이터를 하나의 DataFrame으로 합치기
            data = pd.concat(ticker_data_frames, axis=1)
            
            data.index = data.index.tz_localize(None)
            
    except Exception as e:
        st.error(f"데이터 다운로드 실패: {str(e)}")
        import traceback
        st.error(f"상세 오류: {traceback.format_exc()}")
        return None, None, None
    
    if data is None or data.empty:
        st.error("다운로드된 데이터가 없습니다. 시작일을 조정해보세요.")
        return None, None, None
    
    # 사용 가능한 티커 확인 (NaN이 모두가 아닌 티커)
    available_tickers = []
    for t in backtest_weights.keys():
        if t in data.columns:
            # 해당 티커의 데이터가 있는 행이 하나라도 있으면 사용 가능
            if not data[t].isna().all():
                available_tickers.append(t)
    
    if len(available_tickers) == 0:
        st.error(f"사용 가능한 티커 데이터가 없습니다. 다운로드된 컬럼: {list(data.columns)}, 요청한 티커: {list(backtest_weights.keys())}")
        return None, None, None
    
    if len(available_tickers) < len(backtest_weights):
        missing = [t for t in backtest_weights.keys() if t not in available_tickers]
        st.warning(f"일부 티커 데이터가 없습니다: {', '.join(missing)}. 사용 가능한 티커만으로 백테스트를 진행합니다.")
    
    # 사용 가능한 티커만으로 가중치 재조정
    available_weights = {t: backtest_weights[t] for t in available_tickers}
    total_weight = sum(available_weights.values())
    if total_weight > 0:
        available_weights = {t: w / total_weight for t, w in available_weights.items()}
    
    # 월별 리밸런싱 (사용 가능한 티커만)
    monthly_data = data[available_tickers].resample("M").last()
    
    # NaN이 있는 행 제거 (모든 티커가 NaN인 경우만)
    monthly_data = monthly_data.dropna(how='all')
    
    if len(monthly_data) < 2:
        st.error(f"월별 데이터가 부족합니다 (필요: 최소 2개월, 현재: {len(monthly_data)}개월). 시작일을 조정해보세요.")
        return None, None, None
    
    # 포트폴리오 가치 계산 (시작값 = 100)
    portfolio_value = pd.Series(index=monthly_data.index, dtype=float)
    portfolio_value.iloc[0] = 100.0
    
    for i in range(1, len(monthly_data)):
        prev_value = portfolio_value.iloc[i-1]
        
        # 각 자산의 월간 수익률 계산
        monthly_returns = {}
        total_weight_used = 0
        
        for ticker in available_tickers:
            if ticker in monthly_data.columns:
                prev_price = monthly_data.iloc[i-1][ticker]
                curr_price = monthly_data.iloc[i][ticker]
                if not pd.isna(prev_price) and not pd.isna(curr_price) and prev_price > 0:
                    monthly_returns[ticker] = (curr_price / prev_price) - 1
                    total_weight_used += available_weights.get(ticker, 0)
                else:
                    monthly_returns[ticker] = 0
        
        # 포트폴리오 수익률 = 가중 평균
        portfolio_return = 0
        for ticker, weight in available_weights.items():
            if ticker in monthly_returns:
                portfolio_return += weight * monthly_returns[ticker]
        
        portfolio_value.iloc[i] = prev_value * (1 + portfolio_return)
    
    return portfolio_value, monthly_data, start_date


def get_risk_free_rate(start_date: str = None, end_date: str = None):
    """
    무위험 수익률 조회 (미국 10년 국채 수익률)
    start_date와 end_date가 제공되면 해당 기간의 평균을 사용,
    없으면 최근 1개월 값을 사용
    """
    try:
        # 미국 10년 국채 수익률 조회 (^TNX)
        ticker = yf.Ticker("^TNX")
        
        if start_date and end_date:
            # 백테스트 기간 전체의 평균 사용
            hist = ticker.history(start=start_date, end=end_date)
            if not hist.empty:
                # 기간 전체의 평균 수익률 (이미 % 단위이므로 100으로 나눔)
                avg_rate = hist["Close"].mean() / 100.0
                return avg_rate
        else:
            # 최근 1개월 값 사용 (기존 방식)
            hist = ticker.history(period="1mo")
            if not hist.empty:
                current_rate = hist["Close"].iloc[-1] / 100.0
                return current_rate
    except:
        pass
    
    # 조회 실패 시 보수적인 기본값 사용 (최근 10년 국채 평균 약 2.5%)
    return 0.025


def calculate_performance_metrics(portfolio_value, risk_free_rate=None, backtest_start_date=None, backtest_end_date=None):
    """성과 지표 계산"""
    if portfolio_value is None or len(portfolio_value) < 2:
        return None
    
    # 무위험 수익률 설정 (백테스트 기간 전체의 평균 사용)
    if risk_free_rate is None:
        # 실제 백테스트 시작일/종료일이 제공되면 사용, 없으면 portfolio_value의 기간 사용
        if backtest_start_date and backtest_end_date:
            start_date_str = backtest_start_date.strftime('%Y-%m-%d') if hasattr(backtest_start_date, 'strftime') else str(backtest_start_date)
            end_date_str = backtest_end_date.strftime('%Y-%m-%d') if hasattr(backtest_end_date, 'strftime') else str(backtest_end_date)
        else:
            # portfolio_value는 월별 데이터이므로, 첫 월의 시작일과 마지막 월의 종료일 사용
            start_date_str = portfolio_value.index[0].strftime('%Y-%m-%d')
            end_date_str = portfolio_value.index[-1].strftime('%Y-%m-%d')
        risk_free_rate = get_risk_free_rate(start_date=start_date_str, end_date=end_date_str)
    
    # 기간 계산
    years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
    
    # CAGR
    total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
    cagr = ((1 + total_return) ** (1 / years) - 1) if years > 0 else 0
    
    # 월별 수익률 계산 (portfolio_value는 월별 데이터)
    monthly_returns = portfolio_value.pct_change().dropna()
    
    # 연환산 표준편차 (월별 수익률의 표준편차 * sqrt(12))
    # 월별 수익률을 연환산하려면 sqrt(12)를 곱해야 함
    if len(monthly_returns) > 1:
        monthly_vol = monthly_returns.std()
        annual_vol = monthly_vol * np.sqrt(12)
    else:
        annual_vol = 0
    
    # MDD (최대 낙폭)
    cumulative = portfolio_value
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    mdd = drawdown.min()
    
    # 샤프지수 (무위험 수익률 반영)
    # Sharpe Ratio = (CAGR - Risk-Free Rate) / Annual Volatility
    sharpe = ((cagr - risk_free_rate) / annual_vol) if annual_vol > 0 else 0
    
    return {
        "CAGR": cagr * 100,
        "연환산 표준편차": annual_vol * 100,
        "MDD": mdd * 100,
        "샤프지수": sharpe,
        "무위험 수익률": risk_free_rate * 100,
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


def create_monthly_heatmap_data(monthly_returns):
    """월별 수익률 히트맵 데이터 생성 (연도 x 월)"""
    if monthly_returns is None or len(monthly_returns) == 0:
        return None
    
    # 연도와 월로 분리
    monthly_returns.index = pd.to_datetime(monthly_returns.index)
    monthly_returns_df = monthly_returns.to_frame("return")
    monthly_returns_df['year'] = monthly_returns_df.index.year
    monthly_returns_df['month'] = monthly_returns_df.index.month
    
    # 피벗 테이블 생성 (연도 x 월)
    heatmap_data = monthly_returns_df.pivot_table(
        values='return',
        index='year',
        columns='month',
        aggfunc='first'
    )
    
    # 컬럼 이름을 월 이름으로 변경 (있는 월만)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_dict = {i: month_names[i-1] for i in range(1, 13)}
    
    # 실제 데이터가 있는 월만 선택
    available_months = [month_dict[i] for i in heatmap_data.columns if i in month_dict]
    heatmap_data.columns = [month_dict[i] if i in month_dict else f"Month_{i}" 
                          for i in heatmap_data.columns]
    
    # 있는 월만 유지
    heatmap_data = heatmap_data[[col for col in heatmap_data.columns if col in month_names]]
    
    # 연도 순서 역순 (최신 연도가 아래로)
    heatmap_data = heatmap_data.sort_index(ascending=False)
    
    # 평균 행 계산 (NaN 값 제외하고 계산)
    monthly_avg = heatmap_data.mean(axis=0, skipna=True)
    avg_row = pd.DataFrame([monthly_avg.values], index=['평균'], columns=heatmap_data.columns)
    
    # 평균 행을 맨 앞에 추가 (Y축 역순이므로 맨 앞이 차트 하단에 표시됨)
    heatmap_data = pd.concat([avg_row, heatmap_data])
    
    return heatmap_data


def calculate_drawdown_events(portfolio_value):
    """드로우다운 이벤트 계산"""
    if portfolio_value is None or len(portfolio_value) < 2:
        return None, None
    
    # 드로우다운 계산
    cumulative = portfolio_value
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    
    # 드로우다운 이벤트 찾기 (새로운 드로우다운 시작)
    drawdown_events = []
    in_drawdown = False
    drawdown_start = None
    drawdown_start_value = None
    max_drawdown = 0
    
    for i, (date, dd_value) in enumerate(drawdown.items()):
        if dd_value < 0 and not in_drawdown:
            # 드로우다운 시작
            in_drawdown = True
            drawdown_start = date
            drawdown_start_value = cumulative.loc[date]
            max_drawdown = dd_value
        elif dd_value < max_drawdown and in_drawdown:
            # 더 깊은 드로우다운
            max_drawdown = dd_value
        elif dd_value >= 0 and in_drawdown:
            # 드로우다운 종료
            # 최대 드로우다운 시점 찾기
            drawdown_period = drawdown.loc[drawdown_start:date]
            trough_date = drawdown_period.idxmin()
            trough_value = drawdown_period.min()
            
            drawdown_events.append({
                'start': drawdown_start,
                'trough': trough_date,
                'end': date,
                'drawdown': trough_value
            })
            in_drawdown = False
            max_drawdown = 0
    
    # 진행 중인 드로우다운 처리
    if in_drawdown:
        drawdown_period = drawdown.loc[drawdown_start:]
        trough_date = drawdown_period.idxmin()
        trough_value = drawdown_period.min()
        drawdown_events.append({
            'start': drawdown_start,
            'trough': trough_date,
            'end': portfolio_value.index[-1],
            'drawdown': trough_value
        })
    
    # 드로우다운 크기순으로 정렬
    drawdown_events.sort(key=lambda x: x['drawdown'])
    
    return drawdown, drawdown_events


def create_monthly_distribution(monthly_returns):
    """월별 수익률 분포 히스토그램 데이터 생성"""
    if monthly_returns is None or len(monthly_returns) == 0:
        return None
    
    # 히스토그램 구간 설정 (-10% ~ 10%, 2% 간격)
    bins = np.arange(-10, 12, 2)  # -10, -8, -6, ..., 8, 10
    hist, bin_edges = np.histogram(monthly_returns.values, bins=bins)
    
    # 중간값 계산
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return pd.DataFrame({
        'bin_center': bin_centers,
        'count': hist
    })


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
        if ticker == "SPY+SPYM":
            # SPY와 SPYM을 각각 입력받음
            current_holdings["SPY"] = st.number_input(
                f"SPY 보유 수량",
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.2f",
                key=f"holding_SPY",
                on_change=lambda: st.session_state.update({'auto_calc_trigger': True}) if auto_calculate else None
            )
            current_holdings["SPYM"] = st.number_input(
                f"SPYM 보유 수량",
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.2f",
                key=f"holding_SPYM",
                on_change=lambda: st.session_state.update({'auto_calc_trigger': True}) if auto_calculate else None
            )
        else:
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
        
        # SPY+SPYM의 경우 티커명을 "SPY+SPYM"으로 표시
        display_ticker = ticker
        if ticker == "SPY+SPYM":
            display_ticker = "SPY+SPYM"
        
        comparison_data.append({
            "티커": display_ticker,
            "목표 비중": f"{target_weight:.1f}%",
            "현재 비중": f"{current_weight:.1f}%" if current_value else "0.0%",
            "편차": f"{weight_diff:+.1f}%",
            "상태": "✅" if abs_weight_diff < 1 else ("⬆️" if weight_diff > 0 else "⬇️")
        })
        
        # 우선순위용 데이터 (편차가 큰 순서)
        priority_data.append({
            "티커": display_ticker,
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
    df = display_portfolio_table(rebalancing, total_balance)
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
        # SPY+SPYM의 경우 SPY와 SPYM을 각각 처리
        if ticker == "SPY+SPYM":
            spy_data = data.get("spy", {})
            spym_data = data.get("spym", {})
            
            # SPY 처리
            if spy_data.get("shares_to_buy") and spy_data["shares_to_buy"] > 0.01:
                needs_rebalancing.append({
                    "티커": "SPY",
                    "액션": "구매",
                    "수량": f"{spy_data['shares_to_buy']:.2f}",
                    "금액": f"${spy_data['value_to_buy']:,.2f}"
                })
            if spy_data.get("shares_to_sell") and spy_data["shares_to_sell"] > 0.01:
                needs_rebalancing.append({
                    "티커": "SPY",
                    "액션": "매도",
                    "수량": f"{spy_data['shares_to_sell']:.2f}",
                    "금액": f"${spy_data['value_to_sell']:,.2f}"
                })
            
            # SPYM 처리
            if spym_data.get("shares_to_buy") and spym_data["shares_to_buy"] > 0.01:
                needs_rebalancing.append({
                    "티커": "SPYM",
                    "액션": "구매",
                    "수량": f"{spym_data['shares_to_buy']:.2f}",
                    "금액": f"${spym_data['value_to_buy']:,.2f}"
                })
            if spym_data.get("shares_to_sell") and spym_data["shares_to_sell"] > 0.01:
                needs_rebalancing.append({
                    "티커": "SPYM",
                    "액션": "매도",
                    "수량": f"{spym_data['shares_to_sell']:.2f}",
                    "금액": f"${spym_data['value_to_sell']:,.2f}"
                })
        else:
            # 일반 티커 처리
            if data.get("shares_to_buy") and data["shares_to_buy"] > 0.01:
                needs_rebalancing.append({
                    "티커": ticker,
                    "액션": "구매",
                    "수량": f"{data['shares_to_buy']:.2f}",
                    "금액": f"${data['value_to_buy']:,.2f}"
                })
            if data.get("shares_to_sell") and data["shares_to_sell"] > 0.01:
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
    
    # 가장 늦은 상장일을 기본값으로 설정
    if 'default_start_date' not in st.session_state:
        with st.spinner("티커 상장일을 확인하는 중..."):
            default_start_date = get_latest_listing_date(PORTFOLIO)
            st.session_state['default_start_date'] = default_start_date
    
    default_start_date = st.session_state.get('default_start_date', datetime(2022, 1, 1).date())
    
    col1, col2 = st.columns(2)
    with col1:
        start_date_input = st.date_input(
            "백테스트 시작일",
            value=default_start_date,
            min_value=datetime(2010, 1, 1).date(),
            max_value=datetime.now().date(),
            help="각 티커의 상장일을 확인하여 가장 늦은 날짜를 기본값으로 설정했습니다."
        )
    
    with col2:
        if st.button("🔍 백테스트 실행", type="primary", use_container_width=True):
            try:
                with st.spinner("백테스트를 실행하는 중..."):
                    portfolio_value, monthly_data, start_date = run_portfolio_backtest(
                        PORTFOLIO, 
                        start_date=start_date_input.strftime("%Y-%m-%d")
                    )
                    
                    if portfolio_value is not None and len(portfolio_value) > 0:
                        # 무위험 수익률 조회 (백테스트 기간 전체의 평균 사용)
                        with st.spinner("무위험 수익률을 조회하는 중..."):
                            # 실제 백테스트 시작일과 종료일 전달
                            backtest_start = datetime.strptime(start_date, "%Y-%m-%d") if isinstance(start_date, str) else start_date
                            backtest_end = datetime.now()
                            risk_free_rate = None  # None으로 전달하면 함수 내부에서 기간 평균 계산
                        
                        metrics = calculate_performance_metrics(
                            portfolio_value, 
                            risk_free_rate, 
                            backtest_start_date=backtest_start,
                            backtest_end_date=backtest_end
                        )
                        yearly_returns = calculate_yearly_returns(portfolio_value)
                        monthly_returns = calculate_monthly_returns(portfolio_value)
                        monthly_heatmap = create_monthly_heatmap_data(monthly_returns)
                        drawdown_series, drawdown_events = calculate_drawdown_events(portfolio_value)
                        monthly_distribution = create_monthly_distribution(monthly_returns)
                        
                        if metrics:
                            st.session_state['backtest_results'] = {
                                'metrics': metrics,
                                'yearly_returns': yearly_returns,
                                'monthly_returns': monthly_returns,
                                'portfolio_value': portfolio_value,
                                'monthly_heatmap': monthly_heatmap,
                                'drawdown_series': drawdown_series,
                                'drawdown_events': drawdown_events,
                                'monthly_distribution': monthly_distribution
                            }
                            st.success("백테스트가 완료되었습니다!")
                        else:
                            st.error("성과 지표 계산에 실패했습니다.")
                    else:
                        # 에러 메시지는 run_portfolio_backtest 내부에서 이미 표시됨
                        pass
            except Exception as e:
                st.error(f"백테스트 실행 중 오류 발생: {str(e)}")
                import traceback
                st.error(f"상세 오류: {traceback.format_exc()}")
    
    # 백테스트 결과 표시
    if 'backtest_results' in st.session_state:
        results = st.session_state['backtest_results']
        
        if results['metrics']:
            # 성과 지표 표시
            st.markdown("---")
            st.subheader("📈 성과 지표")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
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
            with col6:
                if '무위험 수익률' in results['metrics']:
                    st.metric("무위험 수익률", f"{results['metrics']['무위험 수익률']:.2f}%")
            
            # 기간 정보
            st.caption(f"기간: {results['metrics']['시작일']} ~ {results['metrics']['종료일']} ({results['metrics']['기간(년)']:.2f}년)")
            
            # ==== 연도별 수익률 차트 ====
            if results['yearly_returns'] is not None and len(results['yearly_returns']) > 0:
                st.markdown("---")
                st.subheader("📊 연도별 수익률 (%)")
                
                if HAS_PLOTLY:
                    yearly_df = results['yearly_returns'].to_frame("수익률")
                    yearly_df.index = yearly_df.index.year
                    
                    # 색상 설정 (양수: 초록, 음수: 빨강)
                    colors = ['#d32f2f' if x < 0 else '#2e7d32' for x in yearly_df['수익률']]
                    
                    # 연도 레이블을 "2022년" 형식으로 변경
                    year_labels = [f"{int(year)}년" for year in yearly_df.index]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=year_labels,
                        y=yearly_df['수익률'],
                        marker_color=colors,
                        text=[f"{x:.1f}%" for x in yearly_df['수익률']],
                        textposition='outside',
                        name='연도별 수익률'
                    ))
                    fig.update_layout(
                        xaxis_title="연도",
                        yaxis_title="수익률 (%)",
                        height=400,
                        showlegend=False,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    yearly_df = results['yearly_returns'].to_frame("수익률 (%)")
                    yearly_df.index = [f"{int(year)}년" for year in yearly_df.index.year]
                    yearly_df = yearly_df.round(2)
                    st.dataframe(yearly_df, use_container_width=True, height=300)
            
            # ==== 월별 수익률 히트맵 ====
            if results.get('monthly_heatmap') is not None and not results['monthly_heatmap'].empty:
                st.markdown("---")
                st.subheader("📅 월별 수익률 (%)")
                
                if HAS_PLOTLY:
                    heatmap_data = results['monthly_heatmap']
                    
                    # Y축 레이블 생성 (연도는 정수로만, 평균은 그대로)
                    y_labels = []
                    y_positions = []
                    for pos, idx in enumerate(heatmap_data.index):
                        if idx == '평균':
                            y_labels.append('평균')
                        else:
                            # 연도를 정수로 변환하여 표시
                            try:
                                year_int = int(float(idx))
                                y_labels.append(str(year_int))
                            except:
                                y_labels.append(str(idx))
                        y_positions.append(pos)
                    
                    # 평균 행에 다른 색상 적용을 위한 z 값 준비
                    z_values = heatmap_data.values.copy()
                    
                    # 색상 스케일 설정 (빨강 -> 흰색 -> 초록)
                    fig = go.Figure(data=go.Heatmap(
                        z=z_values,
                        x=heatmap_data.columns,
                        y=y_positions,  # 위치 인덱스 사용
                        colorscale=[
                            [0, '#d32f2f'],      # 빨강 (음수)
                            [0.5, '#ffffff'],   # 흰색 (0)
                            [1, '#2e7d32']      # 초록 (양수)
                        ],
                        text=[[f"{val:.1f}%" if not pd.isna(val) else "" for val in row] 
                              for row in heatmap_data.values],
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        colorbar=dict(title="수익률 (%)"),
                        ygap=2  # 행 간격
                    ))
                    fig.update_layout(
                        height=400 + len(heatmap_data) * 30,
                        xaxis_title="월",
                        yaxis_title="연도",
                        yaxis=dict(
                            autorange='reversed',  # Y축 역순 (최신 연도가 아래)
                            tickmode='array',
                            tickvals=y_positions,  # 정확한 위치에만 틱 표시
                            ticktext=y_labels,     # 커스텀 레이블 사용
                            dtick=None             # 자동 틱 생성 비활성화
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.dataframe(results['monthly_heatmap'].round(2), use_container_width=True, height=400)
            
            # ==== 월별 수익률 분포 히스토그램 ====
            if results.get('monthly_distribution') is not None:
                st.markdown("---")
                st.subheader("📊 월별 수익률 분포")
                
                if HAS_PLOTLY:
                    dist_data = results['monthly_distribution']
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=dist_data['bin_center'],
                        y=dist_data['count'],
                        marker_color='#2e7d32',
                        name='빈도'
                    ))
                    fig.update_layout(
                        xaxis_title="수익률 (%)",
                        yaxis_title="빈도",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # ==== MDD 차트 ====
            if results.get('drawdown_series') is not None and HAS_PLOTLY:
                st.markdown("---")
                st.subheader("📉 최대 손실폭 (MDD)")
                
                drawdown = results['drawdown_series']
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.metric("현재 MDD", f"{results['metrics']['MDD']:.2f}%")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode='lines',
                    fill='tozeroy',
                    fillcolor='rgba(211, 47, 47, 0.3)',
                    line=dict(color='#d32f2f', width=2),
                    name='드로우다운'
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(
                    xaxis_title="날짜",
                    yaxis_title="드로우다운 (%)",
                    height=400,
                    showlegend=False,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # ==== 드로우다운 이벤트 테이블 ====
            if results.get('drawdown_events') is not None and len(results['drawdown_events']) > 0:
                st.markdown("---")
                st.subheader("📋 포트폴리오 드로우다운")
                
                events = results['drawdown_events'][:10]  # 상위 10개만
                events_data = []
                for i, event in enumerate(events, 1):
                    events_data.append({
                        '순위': i,
                        '시작': event['start'].strftime('%Y/%m'),
                        '종료': event['end'].strftime('%Y/%m'),
                        '드로우다운': f"{event['drawdown']:.1f}%"
                    })
                
                events_df = pd.DataFrame(events_data)
                st.dataframe(events_df, use_container_width=True, hide_index=True)
            
            # ==== 포트폴리오 가치 추이 ====
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
    - SPY+SPYM: 20% (SPY와 SPYM 합산)
    - JEPQ: 10%
    - BRK-B: 15%
    - IEF: 15%
    - TLT: 10%
    - GLD: 10%
    - PDBC: 5%
    """)
