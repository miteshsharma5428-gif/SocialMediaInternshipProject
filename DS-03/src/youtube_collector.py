from googleapiclient.discovery import build
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

API_KEY = os.getenv("AIzaSyBOsgquwxYJyUnwq_BjXcQi7U8btL_A-58")

youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY
)

request = youtube.search().list(
    q="artificial intelligence",
    part="snippet",
    type="video",
    maxResults=50
)

response = request.execute()

videos = []

for item in response["items"]:
    videos.append({
        "video_id": item["id"]["videoId"],
        "title": item["snippet"]["title"],
        "channel": item["snippet"]["channelTitle"],
        "published": item["snippet"]["publishedAt"]
    })

df = pd.DataFrame(videos)

df.to_csv(
    "data/raw/youtube_data.csv",
    index=False
)

print(df.head())
print("Saved!")