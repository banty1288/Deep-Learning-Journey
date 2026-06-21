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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
tscv=TimeSeriesSplit(n_splits=3)
scores = []
values=[10,15,12,18,20,25,22,28,30,35]
df=pd.DataFrame({"Value":values})
df["Lag_1"]=df["Value"].shift(1)
df["Lag_2"]=df["Value"].shift(2)
df["Rolling_Mean_3"]=df["Value"].rolling(3).mean()
df=df.dropna()
X=df[["Lag_1","Lag_2","Rolling_Mean_3"]]
Y=df["Value"]

for train_idx, test_idx in tscv.split(X):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    Y_train = Y.iloc[train_idx]
    Y_test = Y.iloc[test_idx]

    model = XGBRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, Y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(Y_test, preds)

    scores.append(mae)

print("MAE Scores:", scores)
print("Average MAE:", np.mean(scores))

