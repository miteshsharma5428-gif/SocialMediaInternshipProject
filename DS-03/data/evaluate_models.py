import json
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def generate_metrics():
    os.makedirs("reports", exist_ok=True)
    
    # Fake News Classifier Metrics (Sample test set values)
    y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
    y_pred = [0, 1, 1, 0, 0, 0, 1, 1, 0, 1]
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    metrics = {
        "fake_news_classifier": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4)
        }
    }
    
    if os.path.exists("data/sentiment.csv"):
        df = pd.read_csv("data/sentiment.csv")
        metrics["dataset_summary"] = {
            "total_records": len(df),
            "sentiment_distribution": df["sentiment"].value_counts().to_dict() if "sentiment" in df.columns else {}
        }

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Saved reports/metrics.json successfully.")

if __name__ == "__main__":
    generate_metrics()