import pandas as pd

df = pd.read_csv("data/youtube_data.csv")

# Remove duplicates
df = df.drop_duplicates()

# Convert titles to lowercase
df["title"] = df["title"].str.lower()

# Remove empty rows
df = df.dropna()

df.to_csv("data/cleaned_youtube.csv", index=False)

print("Data Cleaned Successfully")
print(df.head())
