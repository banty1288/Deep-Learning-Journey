# import pandas as pd
# data={
#     "Date":["2023-01-01","2026-01-02","2026-01-03"],
#     "Value":[10,15,12]
# }
# df=pd.DataFrame(data)
# df["Date"]=pd.to_datetime(df["Date"])
# import matplotlib.pyplot as plt
# plt.plot(df["Date"],df["Value"])
# plt.xlabel("Date")
# plt.ylabel("Value")
# plt.title("Time Series")
# plt.show()
# df["MA_3"]=df["Value"].rolling(window=3).mean()
# df.set_index("Date", inplace=True)
# monthly = df.resample("Y").mean()
# print(df)
# Lag Features
# import pandas as pd
# df=pd.DataFrame({
#     "Value":[10,15,12,18,20]
# })
# df["Lag_1"] = df["Value"].shift(1)
# df["Lag_2"] = df["Value"].shift(2)
# df["Lag_3"] = df["Value"].shift(3)
# df=df.dropna()
# print(df)
# import pandas as pd
# values = [10,15,12,18,20,25,22]
# df = pd.DataFrame({"Value":values})
# df["Lag_1"] = df["Value"].shift(1)
# df["Lag_2"] = df["Value"].shift(2)
# df["Rolling_Mean_3"] = df["Value"].rolling(3).mean()
# df["Rolling_STD_3"] = df["Value"].rolling(3).std()
# df = df.dropna()
# print(df)
# Split
# import pandas as pd
# df=pd.DataFrame({
#     "Values":[10,15,12,18,20,25,22]
# })
# split=int(len(df)*0.8)
# train=df[:split]
# test=df[split:]
# print(train)
# print(test)

