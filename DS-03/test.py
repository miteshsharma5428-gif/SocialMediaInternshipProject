from googleapiclient.discovery import build

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
    maxResults=2
)

response = request.execute()

for item in response["items"]:
    print(item["snippet"]["title"])