import pandas as pd

try:
    model = pd.read_csv('./csv/model.csv')
    theta0 = model["theta00"].iloc[0]
    theta1 = model["theta01"].iloc[0]
    mean   = model["mean"].iloc[0]
    std    = model["std"].iloc[0]
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

while True:
    raw = input("enter mileage (or 'q' to quit): ").strip()

    if raw.lower() == 'q':
        break

    # Validate input
    try:
        mileage = float(raw)
    except ValueError:
        print("Invalid input please enter a number.")
        continue

    # Predict
    if model_loaded and mileage >= 0:
        mileage_scaled = (mileage - mean) / std
        price = theta0 + (theta1 * mileage_scaled)
        if price < 0:
            price = 0
    else:
        price = 0

    print(f"The price estimated is: {price:.2f}")