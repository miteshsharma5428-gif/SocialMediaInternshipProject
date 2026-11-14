# DS-03: Large-Scale Social Media Intelligence Platform

## Overview
End-to-end NLP and search analytics platform that ingests social media data via YouTube APIs, cleans text, runs multilingual sentiment analysis, discovers dynamic topics using BERTopic, ranks influencers via NetworkX PageRank, detects misinformation, and indexes records into Elasticsearch 8.11 and Kibana with a Streamlit UI.

## How to Run (Without Affecting DS-02)
    cd ~/Desktop/SocialMediaInternshipProject
    source venv/bin/activate
    cd DS-03
    docker start elasticsearch kibana
    streamlit run app.py
