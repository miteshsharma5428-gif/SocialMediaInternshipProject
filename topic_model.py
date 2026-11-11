from bertopic import BERTopic
import pandas as pd

df = pd.read_csv("data/sentiment.csv")

docs = df["clean_title"].astype(str).tolist()

topic_model = BERTopic(
    min_topic_size=2,
    verbose=True
)

topics, probs = topic_model.fit_transform(docs)

df["topic"] = topics

df.to_csv(
    "data/topics.csv",
    index=False
)

print(df["topic"].value_counts())
print("Topics generated successfully")