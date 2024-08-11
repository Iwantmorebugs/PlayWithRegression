import numpy as np
import pandas as pd
from sklearn import linear_model, model_selection, metrics
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

data_glass = pd.read_csv('glass.csv')
data_glass.columns = ['RI','na','mg','al','si','k','va','ba','fe','glass_type']

glass_pretrain, glass_test = train_test_split(data_glass, test_size=0.2, random_state=42) 
glass_train, glass_validation = train_test_split(glass_pretrain, test_size=0.25, random_state=42) 

from sklearn.linear_model import Ridge

def get_model_performance(feature_cols, alpha):
    X = glass_train[feature_cols]
    y = glass_train.RI

    model = Ridge(alpha=alpha)

    model.fit(X, y)

    y_pred = model.predict(glass_validation[feature_cols])
    y_true = glass_validation.RI

    return np.sqrt(metrics.mean_squared_error(y_pred,y_true))

feature_cols = ['al']

# print(get_model_performance(feature_cols, 1))

results = {}
for alpha in np.linspace(0.5, 2, 10):
  for feature_cols in [['na','mg'], ['na','mg', 'al'], ['al','si','k']]:
    results[(alpha, tuple(feature_cols))] = get_model_performance(feature_cols, alpha=alpha)

print(results)

# print(min(results, key=results.get))
best_key= min(results, key=results.get)

# Print the best alpha, feature_cols, and the corresponding RMSE
print(f"Best alpha: {best_key[0]}")
print(f"Best feature columns: {best_key[1]}")
print(f"Best RMSE: {results[best_key]}")
    
feature_cols = ['al', 'si', 'k' ]
X = glass_train[feature_cols]
y = glass_train.RI

model = Ridge(alpha=2)
model.fit(X,y)

y_pred = model.predict(glass_test[feature_cols])
y_true = glass_test.RI

print(np.sqrt(metrics.mean_squared_error(y_pred, y_true)))