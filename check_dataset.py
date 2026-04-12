import pandas as pd

df = pd.read_csv("Medical_QA_Cleaned.csv")

print(df.head())
print(df.columns)
print("Total data:", len(df))