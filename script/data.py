 #%%模組安裝
import numpy as np  
import matplotlib.pyplot as plt  
from vmdpy import VMD  
import numpy as np
import ccxt
import pandas as pd
import pywt
import talib as ta


#%%市場資料來源
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
volume_ = (volume - np.mean(volume)) / np.std(volume)


#計算技術指標
sma_ = ta.SMA(close_data,9)
rsi_ = ta.RSI(close_data,14)
# kd_ = ta.STOCH(high_data,low_data,close_data,fastk_period=9, slowk_period=3, slowd_period=3)
# macd_ = ta.MACD(close_data,fastperiod=12, slowperiod=26, signalperiod=9)
# bbnds_ = ta.BBANDS(close_data, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
cci_ = ta.CCI(high_data,low_data,close_data,14)
adx_ = ta.ADX(close_data,high_data,low_data,14)
mfi_ =ta.MFI(high_data,low_data,close_data,volume,14)
roc_ = ta.ROC(close_data,14)
mdi_ = ta.MINUS_DI(high_data,low_data,close_data,14)
pdi_ = ta.PLUS_DI(high_data,low_data,close_data,14)
mom_ = ta.MOM(close_data,14)
#進行VMD多模態分析
alpha = 2000       # moderate bandwidth constraint  
tau = 0.0           # noise-tolerance (no strict fidelity enforcement)  
K = 3            # 3 modes  
DC = 0             # no DC part imposed  
init = 1           # initialize omegas uniformly  
tol = 1e-7  
 
u, u_hat, omega = VMD(close_data, alpha, tau, K, DC, init, tol) 
signals= pd.DataFrame(u.T)
signal_= signals.iloc[:,0]


#將高頻信號丟進cwt捕捉異常特徵
scales = np.arange(1, 300)  # 可以調整尺度範圍
wavelet = 'morl'  # 使用 Morlet 小波（適合連續性信號）
# coef_, freqs_ = pywt.cwt(signal_.values, scales, wavelet)


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

#%%資料預處裡
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 数据标准化


def init(a, b):
    scaler = StandardScaler()
    if not isinstance(a, (pd.Series, pd.DataFrame)):
        
        item = pd.DataFrame(a, columns=[b])
        item[b] = item[b].fillna(item[b].mean())
        item = scaler.fit_transform(item)
        item = pd.DataFrame(item, columns=[b])
        return item
    elif isinstance(a, pd.Series):
        item=a.to_frame(name=b) 
        item = scaler.fit_transform(item)
        item = pd.DataFrame(item, columns=[b])
        return item # 转换为 DataFrame 并使用给定的列名
    else:
        return a

    
glb = globals()
index = dir()
# print(glb['signal'])
df_index= [i for i in index if i.endswith('_') and '__' not in i]
df_list = [init(glb[i], i) for i in df_index]
df = pd.concat(df_list,axis=1)
cols = df.columns.tolist()

# 交换col1和col2的位置
cols[0], cols[8] = cols[8], cols[0]

# 根据新顺序重新排列列
df = df[cols]

# df['target'] = df['signal_'].copy()
# df.drop(df['signal_'])
# print(df.isna().sum())
print(df)


# print(glb['signal_'])
# print(df['signal_'])
#%%特徵篩選

import pymrmr

# # 选择10个最佳特征
selected_features = pymrmr.mRMR(df, 'MIQ', 10)

print("Selected features:", selected_features)


#%%woe芬箱&關聯分析