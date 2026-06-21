import pandas as pd
values=[10,15,12,18,20,25,22,28,30,35]
df=pd.DataFrame({"Value":values})
df["Lag_1"]=df["Value"].shift(1)
df["Lag_2"]=df["Value"].shift(2)
df["Rolling_Mean_3"]=df["Value"].rolling(3).mean()
df=df.dropna()
X=df[["Lag_1","Lag_2","Rolling_Mean_3"]]
Y=df["Value"]
split=int(len(df)*0.8)
X_train=X[:split]
X_test=X[split:]
Y_train=Y[:split]
Y_test=Y[split:]
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)
from xgboost import XGBRegressor
model=XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train,Y_train)
prediction=model.predict(X_test)
from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(Y_test,prediction)
print(prediction)
print(mae)
importance=pd.DataFrame({
    "Feature":X.columns,
    "Importance":model.feature_importances_
})
print(importance.sort_values("Importance",ascending=False))
