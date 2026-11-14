import os
import pandas as pd
from elasticsearch import Elasticsearch, helpers

# Connect to local Elasticsearch 8.x server
es = Elasticsearch("http://localhost:9200")

def create_index(index_name="social_media_intel"):
    # Field mappings for Elasticsearch v8
    mappings = {
        "properties": {
            "author": {"type": "keyword"},
            "text": {"type": "text"},
            "cleaned_text": {"type": "text"},
            "sentiment": {"type": "keyword"},
            "topic": {"type": "integer"},
            "timestamp": {"type": "date"},
            "likes": {"type": "integer"}
        }
    }
    
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, mappings=mappings)
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")

def index_csv_data(csv_path="data/clean_data.csv", index_name="social_media_intel"):
    if not os.path.exists(csv_path):
        print(f"File '{csv_path}' not found.")
        return

    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")

    actions = [
        {
            "_index": index_name,
            "_source": record
        }
        for record in records
    ]

    success, _ = helpers.bulk(es, actions)
    print(f"Successfully indexed {success} records into '{index_name}'.")

if __name__ == "__main__":
    create_index()
    index_csv_data()