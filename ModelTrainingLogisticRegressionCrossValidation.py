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
# nn
estimator = LogisticRegression()
parameters = {'C' : np.linspace(0.1,10,20)}

from sklearn.model_selection import GridSearchCV
features_cols= ['al','na','fe','mg']
X = data_glass[features_cols]
y = data_glass.household

clf = GridSearchCV(estimator, parameters)

clf.fit(X,y)

print(clf.score(X,y))



