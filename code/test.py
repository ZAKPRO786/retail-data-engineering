import pandas as pd

df = pd.read_csv("../data/raw/retail_data1.csv")

print(df["transaction_date"].sample(30).tolist())