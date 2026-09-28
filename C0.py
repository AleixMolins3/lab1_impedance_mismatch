import json
import time
import chromadb
import statistics

client = chromadb.PersistentClient(path="./chroma_db")

try:
    client.delete_collection(name="lab_sentences")
except:
    pass

collection = client.create_collection(
    name="lab_sentences",
    metadata={"hnsw:space": "l2"}
)

with open("bookcorpus_10k.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)

insertion_times = []
batch_size = 100
print(f"Iniciant la inserció de {len(sentences)} frases a Chroma (En lots de {batch_size})...")

for i in range(0, len(sentences), batch_size):
    batch_sentences = sentences[i:i + batch_size]
    batch_ids = [str(j) for j in range(i, i + len(batch_sentences))]
    batch_embeddings = [[0.0] * 384 for _ in range(len(batch_sentences))]
    
    start_time = time.time()
    
    collection.add(
        documents=batch_sentences,
        embeddings=batch_embeddings,
        ids=batch_ids
    )
    
    end_time = time.time()
    
    time_per_doc = (end_time - start_time) / len(batch_sentences)
    insertion_times.extend([time_per_doc] * len(batch_sentences))

print("\n--- RESULTATS [C0]: Inserció de text a Chroma ---")
print(f"Temps mínim estimat: {min(insertion_times):.6f} segons")
print(f"Temps màxim estimat: {max(insertion_times):.6f} segons")
print(f"Temps mitjà: {statistics.mean(insertion_times):.6f} segons")
print(f"Desviació estàndard: {statistics.stdev(insertion_times):.6f} segons")