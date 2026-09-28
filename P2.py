import time
import psycopg2
import numpy as np
import statistics
from scipy.spatial import distance

conn = psycopg2.connect(
    dbname="lab1_db",
    user="usuari_lab",
    password="password",
    host="localhost"
)
cursor = conn.cursor()

print("Extraient dades de PostgreSQL...")
cursor.execute("""
    SELECT e.sentence_id, s.sentence, e.embedding 
    FROM embeddings e 
    JOIN sentences_text s ON e.sentence_id = s.id;
""")
all_rows = cursor.fetchall()

ids = [row[0] for row in all_rows]
sentences = [row[1] for row in all_rows]
embeddings = np.array([row[2] for row in all_rows])

target_indices = range(10)
query_times_euclidean = []
query_times_cosine = []

print("\n--- Càlcul de similituds per a 10 frases ---")

for idx in target_indices:
    target_vector = embeddings[idx]
    target_sentence = sentences[idx]
    
    start_time_euclidean = time.time()
    euclidean_distances = [distance.euclidean(target_vector, emb) for emb in embeddings]
    
    top3_euclidean_idx = np.argsort(euclidean_distances)[:3]
    top2_similar_euclidean = [i for i in top3_euclidean_idx if i != idx][:2]
    
    end_time_euclidean = time.time()
    query_times_euclidean.append(end_time_euclidean - start_time_euclidean)
    
    start_time_cosine = time.time()
    cosine_distances = [distance.cosine(target_vector, emb) for emb in embeddings]
    
    top3_cosine_idx = np.argsort(cosine_distances)[:3]
    top2_similar_cosine = [i for i in top3_cosine_idx if i != idx][:2]
    
    end_time_cosine = time.time()
    query_times_cosine.append(end_time_cosine - start_time_cosine)

print("\n--- RESULTATS [P2]: Temps de Consulta (Distància Euclidiana) ---")
print(f"Temps mínim: {min(query_times_euclidean):.6f} segons")
print(f"Temps màxim: {max(query_times_euclidean):.6f} segons")
print(f"Temps mitjà: {statistics.mean(query_times_euclidean):.6f} segons")
print(f"Desviació estàndard: {statistics.stdev(query_times_euclidean):.6f} segons")

print("\n--- RESULTATS [P2]: Temps de Consulta (Distància Cosinus) ---")
print(f"Temps mínim: {min(query_times_cosine):.6f} segons")
print(f"Temps màxim: {max(query_times_cosine):.6f} segons")
print(f"Temps mitjà: {statistics.mean(query_times_cosine):.6f} segons")
print(f"Desviació estàndard: {statistics.stdev(query_times_cosine):.6f} segons")

cursor.close()
conn.close()