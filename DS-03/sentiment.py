from transformers import pipeline
import pandas as pd

classifier = pipeline(
    "sentiment-analysis"
)

df = pd.read_csv(
    "data/clean_data.csv"
)

sentiments = []

for text in df["clean_title"]:

    result = classifier(text)[0]

    sentiments.append(
        result["label"]
    )

df["sentiment"] = sentiments

df.to_csv(
    "data/sentiment.csv",
    index=False
)

print("Sentiment Complete")