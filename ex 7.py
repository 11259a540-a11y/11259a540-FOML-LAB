from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=0
)

tree = DecisionTreeRegressor(random_state=1)
tree.fit(x_train, y_train)

forest = RandomForestClassifier(n_estimators=100, random_state=1)
forest.fit(x_train, y_train)

print("one tree accuracy:", round(tree.score(x_train, y_train), 3))
print("one forest accuracy:", round(forest.score(x_train, y_train), 3))

names = data.target_names
new_tumour = [x_test[0]]
result = forest.predict(new_tumour)[0]
print("Diagnosis:",names[result])

import matplotlib.pyplot as plt
importances = forest.feature_importances_
names = data.feature_names
top = sorted(zip(importances,names),reverse=True)[:5]
vals = [x[0] for x in top];labels = [x[1] for x in top]
plt.bar(labels[::-1],vals[::-1],color="#2F49D1")
plt.xlabel("importances");plt.title("top 5 most important features")
plt.show()
