import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Technology Stock Market Dashboard',page_icon='📈',layout='wide')
TICKERS=['AAPL','MSFT','GOOGL','AMZN','NVDA']
LABELS={'AAPL':'Apple','MSFT':'Microsoft','GOOGL':'Alphabet','AMZN':'Amazon','NVDA':'NVIDIA'}
START='2018-01-01'; END='2023-12-31'

@st.cache_data(show_spinner=False)
def download_stocks():
    import yfinance as yf
    frames=[]
    for ticker in TICKERS:
        d=yf.download(ticker,start=START,end=END,auto_adjust=False,progress=False)
        if d.empty: continue
        if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
        d=d.reset_index(); d['Ticker']=ticker; frames.append(d)
    if not frames: raise ValueError('No data was returned by Yahoo Finance.')
    return pd.concat(frames,ignore_index=True)

def clean_data(data):
    d=data.copy(); d.columns=[str(c).strip() for c in d.columns]
    required={'Date','Ticker','Close'}
    if not required.issubset(d.columns): raise ValueError('Data needs Date, Ticker, and Close columns.')
    d['Date']=pd.to_datetime(d['Date'],errors='coerce',utc=True).dt.tz_localize(None)
    d['Ticker']=d['Ticker'].astype(str).str.upper().str.strip()
    for col in ['Open','High','Low','Close','Adj Close','Volume']:
        if col in d.columns: d[col]=pd.to_numeric(d[col],errors='coerce')
    d=d.dropna(subset=['Date','Ticker','Close'])
    d=d[d['Close']>0].drop_duplicates(subset=['Date','Ticker'],keep='last')
    d=d.sort_values(['Ticker','Date']).reset_index(drop=True)
    if d.empty: raise ValueError('No valid price observations after cleaning.')
    return d

def compute(d):
    d=d.sort_values(['Ticker','Date']).copy()
    d['Daily_Return']=d.groupby('Ticker')['Close'].pct_change()*100
    d['Normalized']=d['Close']/d.groupby('Ticker')['Close'].transform('first')*100
    d['Year']=d['Date'].dt.year
    return d

def price_returns(d):
    t=d.groupby('Ticker')['Close'].agg(first='first',last='last')
    return ((t['last']/t['first'])-1)*100

def annual_returns(d):
    t=d.groupby(['Ticker','Year'])['Close'].agg(first='first',last='last')
    return ((t['last']/t['first'])-1).mul(100).rename('Annual Return (%)').reset_index()

def risk_return(d):
    t=d.groupby('Ticker')['Daily_Return'].agg(['mean','std'])
    return pd.DataFrame({'Annualized Return (%)':t['mean']*252,'Annualized Volatility (%)':t['std']*np.sqrt(252)}).reset_index()

def chart(fig):
    fig.update_layout(template='plotly_white',margin=dict(l=10,r=10,t=50,b=10),legend_title_text='Stock')
    st.plotly_chart(fig,use_container_width=True)

st.title('📈 Technology Stock Market Performance Dashboard')
st.caption('Comparative analysis of five technology stocks | 2018–2023 | Price-based returns, excluding dividends')
with st.sidebar:
    st.header('Data source')
    uploaded=st.file_uploader('Upload raw_stock_data.csv',type=['csv'])
    st.caption('Without a CSV, the app downloads historical prices from Yahoo Finance (internet required).')
try:
    raw=pd.read_csv(uploaded) if uploaded is not None else download_stocks()
    full=compute(clean_data(raw))
except Exception as exc:
    st.error(f'Unable to load stock data: {exc}')
    st.info('Upload the raw_stock_data.csv created by your notebook, or check your internet connection and try again.')
    st.stop()
with st.sidebar:
    st.header('Filters')
    available=sorted(full['Ticker'].unique())
    selected=st.multiselect('Stocks',available,default=available)
    minimum=full['Date'].min().date(); maximum=full['Date'].max().date()
    dates=st.date_input('Date range',value=(minimum,maximum),min_value=minimum,max_value=maximum)
    if st.button('Reset filters'):
        st.rerun()
if not selected:
    st.warning('Select at least one stock.'); st.stop()
if len(dates)!=2:
    st.info('Select a start and end date.'); st.stop()
start,end=dates
if start>end:
    st.warning('Start date must be before end date.'); st.stop()
filtered=full[full['Ticker'].isin(selected)&full['Date'].between(pd.Timestamp(start),pd.Timestamp(end))].copy()
if filtered.empty:
    st.warning('No observations match the selected filters.'); st.stop()
# Recompute relative measures on the filtered interval, so comparisons start at the chosen date.
d=compute(filtered.drop(columns=['Daily_Return','Normalized','Year']))
returns=price_returns(d); rr=risk_return(d); annual=annual_returns(d)
c1,c2,c3,c4=st.columns(4)
c1.metric('Stocks selected',len(selected))
c2.metric('Trading observations',f'{len(d):,}')
c3.metric('Best cumulative return',f'{returns.max():.2f}%',returns.idxmax())
c4.metric('Lowest volatility',f'{rr["Annualized Volatility (%)"].min():.2f}%',rr.loc[rr['Annualized Volatility (%)'].idxmin(),'Ticker'])
st.caption('Returns and risk statistics are recalculated for the selected date range. Annualized return = mean daily % return × 252; volatility = standard deviation of daily % returns × √252.')
tabs=st.tabs(['Overview','Price trends','Returns','Risk analysis','Data explorer'])
with tabs[0]:
    st.subheader('Normalized price performance (base = 100)')
    chart(px.line(d,x='Date',y='Normalized',color='Ticker',title='Relative price performance'))
    st.subheader('Total price return by stock')
    chart(px.bar(returns.sort_values().reset_index(name='Return (%)'),x='Ticker',y='Return (%)',color='Ticker',title='Cumulative price returns',text_auto='.1f'))
with tabs[1]:
    chart(px.line(d,x='Date',y='Close',color='Ticker',title='Historical closing prices (USD)'))
    st.info('Closing prices are unadjusted, matching the original notebook. Stock splits may affect cross-period comparisons; use adjusted prices for a total-return-oriented analysis.')
    if 'Volume' in d.columns:
        chart(px.line(d,x='Date',y='Volume',color='Ticker',title='Trading volume'))
with tabs[2]:
    chart(px.bar(annual,x='Year',y='Annual Return (%)',color='Ticker',barmode='group',title='Within-year first-to-last closing-price returns'))
    chart(px.line(d,x='Date',y='Daily_Return',color='Ticker',title='Daily percentage returns'))
    chart(px.histogram(d.dropna(subset=['Daily_Return']),x='Daily_Return',color='Ticker',nbins=70,barmode='overlay',opacity=.55,title='Distribution of daily returns'))
    st.dataframe(annual.pivot(index='Year',columns='Ticker',values='Annual Return (%)').round(2),use_container_width=True)
with tabs[3]:
    fig=px.scatter(rr,x='Annualized Volatility (%)',y='Annualized Return (%)',text='Ticker',color='Ticker',size_max=20,title='Risk versus return')
    fig.update_traces(marker_size=14,textposition='top center'); chart(fig)
    pivot=d.pivot(index='Date',columns='Ticker',values='Daily_Return')
    correlation=pivot.corr(min_periods=10)
    if correlation.notna().any().any():
        chart(px.imshow(correlation,text_auto='.2f',zmin=-1,zmax=1,color_continuous_scale='RdBu_r',title='Daily-return correlation heatmap'))
    st.dataframe(rr.round(2),hide_index=True,use_container_width=True)
with tabs[4]:
    st.subheader('Filtered dataset')
    st.dataframe(d.sort_values(['Date','Ticker'],ascending=[False,True]),use_container_width=True,hide_index=True)
    st.download_button('Download filtered, cleaned CSV',data=d.to_csv(index=False).encode('utf-8'),file_name='cleaned_filtered_stock_data.csv',mime='text/csv')
    st.subheader('Data quality checks')
    a,b,c=st.columns(3)
    a.metric('Rows',len(d));b.metric('Missing closing prices',int(d['Close'].isna().sum()));c.metric('Duplicate date/ticker pairs',int(d.duplicated(['Date','Ticker']).sum()))
    st.write('Descriptive statistics for closing prices')
    st.dataframe(d.groupby('Ticker')['Close'].describe().round(2),use_container_width=True)
st.divider()
st.caption('Academic project dashboard. Historical price performance is not a forecast or investment recommendation. Data source: uploaded notebook CSV or Yahoo Finance via yfinance.')
