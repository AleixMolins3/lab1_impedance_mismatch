import time
import psycopg2
import statistics
from sentence_transformers import SentenceTransformer

conn = psycopg2.connect(
    dbname="lab1_db",
    user="usuari_lab",
    password="password",
    host="localhost"
)
cursor = conn.cursor()

cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

cursor.execute("DROP TABLE IF EXISTS embeddings_pgvector;")
cursor.execute("""
    CREATE TABLE embeddings_pgvector (
        id SERIAL PRIMARY KEY,
        sentence_id INTEGER REFERENCES sentences_text(id),
        embedding VECTOR(384)
    );
""")
conn.commit()

cursor.execute("SELECT id, sentence FROM sentences_text;")
rows = cursor.fetchall()

print("Carregant el model all-MiniLM-L6-v2...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

insertion_times = []
print(f"Generant i emmagatzemant vectors natius (pgvector) per a {len(rows)} frases...")

for row in rows:
    sentence_id = row[0]
    text = row[1]
    
    vector = model.encode(text).tolist()
    
    start_time = time.time()
    
    cursor.execute(
        "INSERT INTO embeddings_pgvector (sentence_id, embedding) VALUES (%s, %s::vector)", 
        (sentence_id, vector)
    )
    conn.commit() 
    
    end_time = time.time()
    insertion_times.append(end_time - start_time)

print("\n--- RESULTATS [G1]: Inserció d'Embeddings (Amb Pgvector) ---")
print(f"Temps mínim: {min(insertion_times):.6f} segons")
print(f"Temps màxim: {max(insertion_times):.6f} segons")
print(f"Temps mitjà: {statistics.mean(insertion_times):.6f} segons")
print(f"Desviació estàndard: {statistics.stdev(insertion_times):.6f} segons")

cursor.close()
conn.close()