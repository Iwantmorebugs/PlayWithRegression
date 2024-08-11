
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection, metrics

import warnings
warnings.filterwarnings('ignore')

data_glass = pd.read_csv('glass.csv')
data_glass.columns = ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type']

# print(data_glass.head(3))

sns.set_theme(font_scale=1.5)
sns.lmplot(data=data_glass,  y='RI', x='Al',)
plt.show()


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()

feature_cols = ['Al']
X = data_glass[feature_cols]
y = data_glass.RI


linreg.fit(X, y, sample_weight=0.1)

y_pred= linreg.predict(X)
data_glass['y_pred'] = y_pred

print(data_glass.head())


