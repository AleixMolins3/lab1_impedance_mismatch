import json
import time
import psycopg2
import statistics

with open("bookcorpus_10k.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)

conn = psycopg2.connect(
    dbname="lab1_db",
    user="usuari_lab",
    password="password",
    host="localhost"
)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS sentences_text;")
cursor.execute("""
    CREATE TABLE sentences_text (
        id SERIAL PRIMARY KEY,
        sentence TEXT NOT NULL
    );
""")
conn.commit()

insertion_times = []
print(f"Iniciant la inserció de {len(sentences)} frases a PostgreSQL...")

for sentence in sentences:
    start_time = time.time()
    
    cursor.execute("INSERT INTO sentences_text (sentence) VALUES (%s)", (sentence,))
    conn.commit() 
    
    end_time = time.time()
    insertion_times.append(end_time - start_time)

min_time = min(insertion_times)
max_time = max(insertion_times)
avg_time = statistics.mean(insertion_times)
std_time = statistics.stdev(insertion_times)

print("\n--- RESULTATS [P0]: Inserció de text pur ---")
print(f"Temps mínim: {min_time:.6f} segons")
print(f"Temps màxim: {max_time:.6f} segons")
print(f"Temps mitjà: {avg_time:.6f} segons")
print(f"Desviació estàndard: {std_time:.6f} segons")

cursor.close()
conn.close()