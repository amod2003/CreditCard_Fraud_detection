import pandas as pd

data = pd.read_csv("creditcard.csv")

for i in range(0, 5):
    sample = data.sample(5000)
    sample.to_csv(f"Test_{(i+1)}.csv", index=False)