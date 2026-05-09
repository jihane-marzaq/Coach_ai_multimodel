import pandas as pd

# charger fichier CSV (adapte le nom)
df = pd.read_csv("ted_transcripts.csv")

print(df.head())