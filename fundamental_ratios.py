import pandas as pd
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from tabulate import tabulate

def fundamental_metric(soup, metric):
    return soup.find(text = metric).find_next(class_='snapshot-td2').text

def get_fundamental_data(df):
    for symbol in df.index:
        try:
            url = ("http://finviz.com/quote.ashx?t=" + symbol.lower())
            req = Request(url=url,headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'}) 
            response = urlopen(req)
            soup = BeautifulSoup(response, features='lxml')
            for m in df.columns:                
                df.loc[symbol,m] = fundamental_metric(soup,m)                
        except Exception as e:
            print (symbol, 'not found')
    return df

stock_list = ['AMZN','GOOG','PG','KO','IBM','DG','XOM','KO','PEP','MT','NL','LPL', 'META', 'TSLA', 'NVDA']

metric = ['P/B',
'P/E',
'Forward P/E',
'PEG',
'Debt/Eq',
'EPS (ttm)',
'Dividend %',
'ROE',
'ROI',
'EPS Q/Q',
'Insider Own'
]
print("----- PORTFOLIO-----")
print(stock_list, '\n')
print("----- METRICS -----")
print(metric, '\n')


df = pd.DataFrame(index=stock_list,columns=metric)
df = get_fundamental_data(df)
print("----- RAW DATA -----")
print(tabulate(df, headers=metric, tablefmt='psql'))
print()

df['Dividend %'] = df['Dividend %'].str.replace('%', '')
df['ROE'] = df['ROE'].str.replace('%', '')
df['ROI'] = df['ROI'].str.replace('%', '')
df['EPS Q/Q'] = df['EPS Q/Q'].str.replace('%', '')
df['Insider Own'] = df['Insider Own'].str.replace('%', '')
df = df.apply(pd.to_numeric, errors='coerce')
print("----- CLEAN DATA -----")
print(tabulate(df, headers=metric, tablefmt='psql'))
print()

# companies quoted at low valuations
df_filtered = df[(df['P/E'].astype(float)<15) & (df['P/B'].astype(float) < 1)]
print("----- LOW VALUATIONS -----")
print(tabulate(df_filtered, headers=metric, tablefmt='psql'))
print()

# companies which have demonstrated earning power
df_filtered = df_filtered[df_filtered['EPS Q/Q'].astype(float) > 10]
print("----- HIGH EARNIG POWER -----")
print(tabulate(df_filtered, headers=metric, tablefmt='psql'))
print()

# management having substantial ownership in the business
df = df[df['Insider Own'].astype(float) > 30]
print("----- INSIDER OWNERSHIP -----")
print(tabulate(df, headers=metric, tablefmt='psql'))

