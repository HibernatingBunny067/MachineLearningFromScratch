from models import DecisionTreeClassifier,LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score

X,y = make_classification(
    n_samples=1000,
    n_features=10,
    n_classes=2,
    n_informative=3,
    random_state=42
)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True,random_state=42)

model1 = DecisionTreeClassifier(max_depth=6)
model2 = LogisticRegression(tol=1e-5)

model1.fit(X_train,y_train)
model2.fit(X_train,y_train)

predictions1 = model1.predict(X_test)
predictions2 = model2.predict(X_test)

accuracy1 = accuracy_score(y_test,predictions1)
accuracy2 = accuracy_score(y_test,predictions2)

print(accuracy1)
print(accuracy2)