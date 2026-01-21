import pandas as pd
import re

# Caricamento dataset
df = pd.read_csv("data/data.csv")
print(df.head())

# Concatenazione title e body
df["full_text"] = df["title"] + " " + df["body"]

# Preprocessing del testo
df.full_text = df.full_text.str.lower()
print(df.full_text.head())

# Rimozione caratteri speciali e doppi spazi
pattern = r"[!?,.;:()\[\]{}@%*/\\|=+]"
df["full_text"] = df["full_text"].str.replace(pattern, " ", regex = True)
df["full_text"] = df["full_text"].str.replace(r"\s+", " ", regex=True).str.strip()

print(df["full_text"].head())

# Salvataggio dataset preprocessato
df.to_csv("data/dataset_clean.csv", index = False)
