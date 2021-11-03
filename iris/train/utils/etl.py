import os
import pandas as pd


# Load dataset
#df = pd.read_csv(os.getcwd() + '\\data\\iris.csv')
df = pd.read_csv(r"../data/iris.csv")    # 字符串前r表示不转义

# Split into training data and test data
X = df[['sepal_length','sepal_width','petal_length','petal_width']]
y = df['classification']


if __name__ == '__main__':

    print(X.tail(5))
    print('='*30)
    print(y.tail(5))
