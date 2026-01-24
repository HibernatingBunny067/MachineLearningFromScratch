from models import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X,y = make_classification(
    n_samples=1000,
    n_features=10,
    n_classes=2,
    random_state=42
)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True,random_state=42)


model = LogisticRegression(lr=0.001,tol=1e-6,verbose=True)
model.fit(X_train,y_train)

p = model.predict(X_test)
accuracy = accuracy_score(y_test,p)

print('MODEL PERFORMANCE')
print('Acc: {accuracy}'.format(accuracy=accuracy))

fig,ax = plt.subplots()