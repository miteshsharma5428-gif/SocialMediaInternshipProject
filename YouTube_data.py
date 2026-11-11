from googleapiclient.discovery import build
import pandas as pd

API_KEY = "AIzaSyBOsgquwxYJyUnwq_BjXcQi7U8btL_A-58"

youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY,
    cache_discovery=False
)

request = youtube.search().list(
    q="artificial intelligence",
    part="snippet",
    maxResults=50
)

response = request.execute()

data = []

for item in response["items"]:
    title = item["snippet"]["title"]
    channel = item["snippet"]["channelTitle"]

    data.append([title, channel])

df = pd.DataFrame(
    data,
    columns=["title", "channel"]
)

df.to_csv(
    "data/youtube_data.csv",
    index=False
)

print("Data Saved")