import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt


data_glass = pd.read_csv('glass.csv')
data_glass.columns = ['ri','na','mg','al','si','k','va','ba','fe','glass_type']

# print(data_glass.glass_type.value_counts().sort_index())

data_glass['household'] = data_glass.glass_type.apply(lambda x: int(x/4))

# print(data_glass.household.value_counts())

plt.scatter(data_glass.al, data_glass.household)
plt.xlabel('al')
plt.ylabel('household')

# plt.show()


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

features_cols= ['al']
X = data_glass[features_cols]
y = data_glass.household

# train
logreg.fit(X,y)
# predict
pred = logreg.predict(X)

# logreg.predict_proba(X)[0:5]

data_glass['household_pred_prob'] = logreg.predict_proba(X)[:, 1]
glass_sorted = data_glass.sort_values(by='household_pred_prob', ascending=True)

# print(glass_sorted)

plt.scatter(glass_sorted.al, glass_sorted.household)
plt.plot(glass_sorted.al, glass_sorted.household_pred_prob, color='red')
plt.xlabel('al')
plt.ylabel('probabilty of household')

plt.show()

# # print(logreg.predict_proba([[1]]))
# # print(logreg.predict_proba([[2]]))
# # print(logreg.predict_proba([[3]]))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=99)

model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)
print(accuracy)

