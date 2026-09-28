import json
import urllib.request
from sentence_transformers import SentenceTransformer

print("Obtenint text pur (Moby Dick) per evitar els errors de l'API de HuggingFace...")

url = "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
response = urllib.request.urlopen(req)
raw_text = response.read().decode('utf-8')

sentences = [
    s.strip().replace('\r\n', ' ').replace('\n', ' ') + '.' 
    for s in raw_text.split('.') 
    if len(s.strip()) > 30
]

text_data = sentences[:10000]

with open("bookcorpus_10k.json", "w", encoding="utf-8") as f:
    json.dump(text_data, f)
print(f"S'han guardat {len(text_data)} frases a bookcorpus_10k.json")

print("Carregant el model all-MiniLM-L6-v2...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(["Prova de funcionament."])
print(f"Embeddings generats amb èxit. Dimensió del vector: {len(embeddings[0])}")