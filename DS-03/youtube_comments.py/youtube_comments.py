from googleapiclient.discovery import build
import pandas as pd

API_KEY = "YOUR_API_KEY"

youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY
)

video_id = "dQw4w9WgXcQ"   # replace later

comments = []

request = youtube.commentThreads().list(
    part="snippet",
    videoId=video_id,
    maxResults=100,
    textFormat="plainText"
)

response = request.execute()

for item in response["items"]:
    comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
    comments.append(comment)

df = pd.DataFrame(
    comments,
    columns=["comment"]
)

df.to_csv(
    "data/comments.csv",
    index=False
)

print(df.head())