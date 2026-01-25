from models import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score

X,y = make_classification(
    n_samples=1000,
    n_features=30,
    n_classes=2,
    n_informative=3,
    random_state=42
)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True,random_state=42)

model = RandomForestClassifier(n_trees=500,min_samples_split=6,max_depth=10,verbose=True)

model.fit(X_train,y_train)

predictions = model.predict(X_test)

acc = accuracy_score(y_test,predictions)

print(acc)