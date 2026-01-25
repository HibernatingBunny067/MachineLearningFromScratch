import numpy as np
from models import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,root_mean_squared_error
import matplotlib.pyplot as plt

X,y = make_regression( # type: ignore
    n_samples=1000,
    n_features=10,
    noise=20.0,
    bias=5.0,
    random_state=42
)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True)

model = LinearRegression(tol=1e-5,verbose=False)

model.fit(X_train,y_train)

y_hat = model.predict(X_test)

r2 = r2_score(y_test,y_hat)
rmse = root_mean_squared_error(y_test,y_hat)

print("="*5 + "MODEL PERFORMANCE" + "="*5)
print(f'R2: {r2}, RMSE: {rmse}')

fig,ax = plt.subplots(figsize=(6,6))
ax.scatter(y_test,y_hat,s=2,color='blue')
plt.show()