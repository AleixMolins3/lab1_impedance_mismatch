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

cursor.execute("DROP TABLE IF EXISTS embeddings;")
cursor.execute("""
    CREATE TABLE embeddings (
        id SERIAL PRIMARY KEY,
        sentence_id INTEGER REFERENCES sentences_text(id),
        embedding REAL[] NOT NULL
    );
""")
conn.commit()

cursor.execute("SELECT id, sentence FROM sentences_text;")
rows = cursor.fetchall()

print("Carregant el model all-MiniLM-L6-v2...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

insertion_times = []
print(f"Generant i emmagatzemant vectors per a {len(rows)} frases. Això pot trigar un parell de minuts...")

for row in rows:
    sentence_id = row[0]
    text = row[1]
    
    vector = model.encode(text).tolist()
    
    start_time = time.time()
    
    cursor.execute("INSERT INTO embeddings (sentence_id, embedding) VALUES (%s, %s)", (sentence_id, vector))
    conn.commit() 
    
    end_time = time.time()
    insertion_times.append(end_time - start_time)

min_time = min(insertion_times)
max_time = max(insertion_times)
avg_time = statistics.mean(insertion_times)
std_time = statistics.stdev(insertion_times)

print("\n--- RESULTATS [P1]: Inserció d'Embeddings (Sense Pgvector) ---")
print(f"Temps mínim: {min_time:.6f} segons")
print(f"Temps màxim: {max_time:.6f} segons")
print(f"Temps mitjà: {avg_time:.6f} segons")
print(f"Desviació estàndard: {std_time:.6f} segons")

cursor.close()
conn.close()