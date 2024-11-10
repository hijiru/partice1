 
import numpy as np  
import matplotlib.pyplot as plt  
from vmdpy import VMD  
import numpy as np
import ccxt
import pandas as pd
import pywt
import talib as ta


#市場資料來源
def get_binance_data():
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1m', limit=500) 
    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('datetime', inplace=True)
    data = data[['open', 'high', 'low', 'close', 'volume']]
    return data

data = get_binance_data()
close_data = data["close"].values
high_data = data["high"].values
low_data = data["low"].values
open_data = data["open"].values
volume = data["volume"].values
close_data = (close_data - np.mean(close_data)) / np.std(close_data)
high_data = (high_data - np.mean(high_data)) / np.std(high_data)
low_data = (low_data - np.mean(low_data)) / np.std(low_data)
open_data = (open_data - np.mean(open_data)) / np.std(open_data)
volume = (volume - np.mean(volume)) / np.std(volume)


#計算技術指標
sma = ta.SMA(close_data,9)
rsi = ta.RSI(close_data,14)
kd = ta.STOCH(high_data,low_data,close_data,fastk_period=9, slowk_period=3, slowd_period=3)
macd = ta.MACD(close_data,fastperiod=12, slowperiod=26, signalperiod=9)
bbnds = ta.BBANDS(close_data, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
cci = ta.CCI(high_data,low_data,close_data,14)
adx = ta.ADX(close_data,high_data,low_data,14)
mfi =ta.MFI(high_data,low_data,close_data,volume,14)
roc = ta.ROC(close_data,14)
mdi = ta.MINUS_DI(high_data,low_data,close_data,14)
pdi = ta.PLUS_DI(high_data,low_data,close_data,14)
mom = ta.MOM(close_data,14)
#進行VMD多模態分析
alpha = 2000       # moderate bandwidth constraint  
tau = 0.0           # noise-tolerance (no strict fidelity enforcement)  
K = 3            # 3 modes  
DC = 0             # no DC part imposed  
init = 1           # initialize omegas uniformly  
tol = 1e-7  
 
u, u_hat, omega = VMD(close_data, alpha, tau, K, DC, init, tol) 
m= pd.DataFrame(u.T)
m= m.iloc[:,1]


#將高頻信號丟進cwt捕捉異常特徵
scales = np.arange(1, 300)  # 可以調整尺度範圍
wavelet = 'morl'  # 使用 Morlet 小波（適合連續性信號）
coef, freqs = pywt.cwt(m.values, scales, wavelet)
    

#%%將計算完的特徵加入資料庫%%
# # 建立連接
# from pymongo import MongoClient
# client = MongoClient("mongodb://localhost:27017/")

# # 選擇資料庫
# db = client.my_database

# # 選擇集合（相當於關係資料庫中的表）
# collection = db.my_collection

# # 插入一條資料
# data = {"name": "Alice", "age": 25, "city": "New York"}
# collection.insert_one(data)

# # 插入多條資料
# data_list = [
#     {"name": "Bob", "age": 30, "city": "Chicago"},
#     {"name": "Charlie", "age": 35, "city": "San Francisco"}
# ]
# collection.insert_many(data_list)

#%%特徵篩選

