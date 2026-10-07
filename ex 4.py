import pandas as pd                                      # tables
from sklearn.model_selection import train_test_split     # splits data
from sklearn.preprocessing import StandardScaler         # resizes numbers
from sklearn.neighbors import KNeighborsClassifier       # the KNN model
from sklearn.svm import SVC                               # the SVM model

data = pd.read_csv("Social_Network_Ads.csv")             # load the CSV file
X = data[["Age","EstimatedSalary"]]                      # inputs: age and salary
y = data["Purchased"]                                    # answer: 1 buy, 0 no buy

X_train, X_test, y_train, y_test = train_test_split(     # split learn + exam
    X, y, test_size=0.25, random_state=1)

scaler = StandardScaler()                                # create the resizer
X_train = scaler.fit_transform(X_train)                  # resize training numbers
X_test  = scaler.transform(X_test)                       # resize test numbers the same way

knn = KNeighborsClassifier(n_neighbors=5)                # make KNN
knn.fit(X_train, y_train)                                # train KNN
print("KNN accuracy:", round(knn.score(X_test, y_test), 2))  # KNN score

svm = SVC()                                              # make SVM
svm.fit(X_train, y_train)                                # train SVM
print("SVM accuracy:", round(svm.score(X_test, y_test), 2))  # SVM score

new_person = scaler.transform(pd.DataFrame([[40, 90000]],columns = ['Age','EstimatedSalary']))       # new visitor: age 40, salary 90000 (resized)
answer = knn.predict(new_person)[0]                      # ask KNN (returns 1 or 0)
print("Will they buy? (1=yes, 0=no):", answer)           # show the answer