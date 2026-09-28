import time
import chromadb
import statistics

client = chromadb.PersistentClient(path="./chroma_db")

collection_l2 = client.get_collection(name="lab_sentences")

target_ids = [str(i) for i in range(10)]
targets = collection_l2.get(ids=target_ids, include=['documents', 'embeddings'])
target_embeddings = targets['embeddings']

print("\n--- Càlcul de similituds per a 10 frases a Chroma ---")

query_times_euclidean = []
print("Consultant utilitzant distància Euclidiana (índex l2)...")
for i in range(10):
    start_time = time.time()
    results = collection_l2.query(
        query_embeddings=[target_embeddings[i]],
        n_results=3
    )
    end_time = time.time()
    query_times_euclidean.append(end_time - start_time)

print("\nPreparant col·lecció temporal per a mètrica Cosinus...")
try:
    client.delete_collection(name="lab_sentences_cosine")
except:
    pass

collection_cosine = client.create_collection(
    name="lab_sentences_cosine",
    metadata={"hnsw:space": "cosine"}
)

all_data = collection_l2.get(include=['documents', 'embeddings'])
batch_size = 500
for i in range(0, len(all_data['ids']), batch_size):
    collection_cosine.add(
        ids=all_data['ids'][i:i+batch_size],
        documents=all_data['documents'][i:i+batch_size],
        embeddings=all_data['embeddings'][i:i+batch_size]
    )

query_times_cosine = []
print("Consultant utilitzant similitud Cosinus (índex cosine)...")
for i in range(10):
    start_time = time.time()
    results = collection_cosine.query(
        query_embeddings=[target_embeddings[i]],
        n_results=3
    )
    end_time = time.time()
    query_times_cosine.append(end_time - start_time)

print("\n--- RESULTATS [C2]: Temps de Consulta (Distància Euclidiana) ---")
print(f"Temps mínim: {min(query_times_euclidean):.6f} segons")
print(f"Temps màxim: {max(query_times_euclidean):.6f} segons")
print(f"Temps mitjà: {statistics.mean(query_times_euclidean):.6f} segons")
print(f"Desviació estàndard: {statistics.stdev(query_times_euclidean):.6f} segons")

print("\n--- RESULTATS [C2]: Temps de Consulta (Distància Cosinus) ---")
print(f"Temps mínim: {min(query_times_cosine):.6f} segons")
print(f"Temps màxim: {max(query_times_cosine):.6f} segons")
print(f"Temps mitjà: {statistics.mean(query_times_cosine):.6f} segons")
print(f"Desviació estàndard: {statistics.stdev(query_times_cosine):.6f} segons")