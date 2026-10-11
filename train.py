from model.linearRegression import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


if __name__ == '__main__':
    

    df = pd.read_csv('./csv/data.csv')
    X = df.values[:, 0:1]   # km
    y = df.values[:, 1]     # price
    
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr.save_model(scalar.mean_[0], scalar.scale_[0])
    predict = lr.predict(X_test)

    # sort by km so the line is drawn left-to-right

    X_scaled = scalar.transform(X)
    line = lr.predict(X_scaled)

    order = np.argsort(X.flatten())
    X_sorted = X.flatten()[order]
    y_sorted = y[order]
    line_sorted = line[order]

    lr.show_plot(X_sorted, y_sorted, line_sorted)

