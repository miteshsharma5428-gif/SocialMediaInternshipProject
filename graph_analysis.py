import pandas as pd
import networkx as nx

df = pd.read_csv(
    "data/topics.csv"
)

G = nx.Graph()

for _, row in df.iterrows():

    title = row["clean_title"]

    topic = str(row["topic"])

    G.add_edge(
        title,
        topic
    )

print(
    "Nodes:",
    G.number_of_nodes()
)

print(
    "Edges:",
    G.number_of_edges()
)