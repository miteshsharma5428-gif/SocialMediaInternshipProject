from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "AI will destroy world",
    "Research paper released"
]

labels = [
    1,
    0
]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(
    texts
)

model = LogisticRegression()

model.fit(
    X,
    labels
)

print(
    "Fake News Model Created"
)