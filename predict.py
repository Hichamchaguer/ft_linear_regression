import pandas as pd


model = pd.read_csv('./csv/model.csv')

theta0 = model["theta00"].iloc[0]
theta1 = model["theta01"].iloc[0]
mean = model["mean"].iloc[0]
std = model["std"].iloc[0]

mileage = float(input('enter mileage : '))

mileage_scaled = (mileage - mean) / std
price = theta0 + (theta1 * mileage_scaled)
print(f'the price estimated is : {price}')