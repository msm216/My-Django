import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from etl import X, y

c_value = 1

# Split train test datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=1
)
# Train and save model
mod_path = os.path.abspath(r'../..') + '\\model\\svc.pickle'
model = SVC(C=c_value).fit(X_train.values, y_train.values)
pd.to_pickle(model, mod_path)

# Make prediction
y_pred = model.predict(X_test.values)

# Evaluate
score_train = np.mean(cross_val_score(model, X_train.values, y_train.values, cv=5))
score_test = model.score(X_test.values, y_test)
conf_matrix = confusion_matrix(y_test, y_pred)

# Get DataFrame of confusion matrix
df_cm = pd.DataFrame(
    conf_matrix,
    index=[
        'Iris-setosa',
        'Iris-versicolor',
        'Iris-virginica'
    ],
    columns=[
        'Iris-setosa',
        'Iris-versicolor',
        'Iris-virginica'
    ]
)


fig_path = '..\\..\\static\\images\\cm'

# Save confusion matrix as heatmap
fig = sns.heatmap(
    df_cm,
    cmap='Blues',
    square=True, linewidths=1,
    linecolor='black',
    annot=True
)
img = fig.get_figure()
img.savefig(fig_path, dpi=400)


if __name__ == '__main__':

    print("Training score: %.2f" % score_train)
    print("Testing score: %.2f" % score_test)
    #print(accuracy_score(y_test, y_pred))
    print('='*10, "Confusion Matrix", '='*10)
    print(df_cm)
