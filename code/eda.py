import pandas as pd

df = pd.read_csv(
    "../data/processed/cleaned_retail_data.csv"
)

print(df.isnull().sum())