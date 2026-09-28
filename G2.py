import time
import psycopg2
import statistics

conn = psycopg2.connect(
    dbname="lab1_db",
    user="usuari_lab",
    password="password",
    host="localhost"
)
cursor = conn.cursor()


cursor.execute("""
    SELECT sentence_id, embedding::text 
    FROM embeddings_pgvector 
    ORDER BY id 
    LIMIT 10;
""")
targets = cursor.fetchall()

print("\n--- Càlcul de similituds per a 10 frases a PostgreSQL (Amb Pgvector) ---")


query_times_euclidean = []
print("Consultant utilitzant distància Euclidiana...")
for target in targets:
    target_vector = target[1]
    
    start_time = time.time()

    cursor.execute("""
        SELECT sentence_id 
        FROM embeddings_pgvector 
        ORDER BY embedding <-> %s::vector 
        LIMIT 3;
    """, (target_vector,))
    results = cursor.fetchall()
    end_time = time.time()
    
    query_times_euclidean.append(end_time - start_time)


query_times_cosine = []
print("Consultant utilitzant similitud Cosinus...")
for target in targets:
    target_vector = target[1]
    
    start_time = time.time()

    cursor.execute("""
        SELECT sentence_id 
        FROM embeddings_pgvector 
        ORDER BY embedding <=> %s::vector 
        LIMIT 3;
    """, (target_vector,))
    results = cursor.fetchall()
    end_time = time.time()
    
    query_times_cosine.append(end_time - start_time)

print("\n--- RESULTATS [G2]: Temps de Consulta (Distància Euclidiana) ---")
print(f"Temps mínim: {min(query_times_euclidean):.6f} segons")
print(f"Temps màxim: {max(query_times_euclidean):.6f} segons")
print(f"Temps mitjà: {statistics.mean(query_times_euclidean):.6f} segons")
print(f"Desviació estàndard: {statistics.stdev(query_times_euclidean):.6f} segons")

print("\n--- RESULTATS [G2]: Temps de Consulta (Distància Cosinus) ---")
print(f"Temps mínim: {min(query_times_cosine):.6f} segons")
print(f"Temps màxim: {max(query_times_cosine):.6f} segons")
print(f"Temps mitjà: {statistics.mean(query_times_cosine):.6f} segons")
print(f"Desviació estàndard: {statistics.stdev(query_times_cosine):.6f} segons")

cursor.close()
conn.close()