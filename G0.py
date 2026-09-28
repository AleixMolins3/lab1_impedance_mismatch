import json
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

cursor.execute("DROP TABLE IF EXISTS sentences_text_g CASCADE;")
cursor.execute("""
    CREATE TABLE sentences_text_g (
        id SERIAL PRIMARY KEY,
        sentence TEXT NOT NULL
    );
""")
conn.commit()

with open("bookcorpus_10k.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)

insertion_times = []
print(f"Iniciant la inserció de {len(sentences)} frases a PostgreSQL (Fase G0 - Text pur)...")

for sentence in sentences:
    start_time = time.time()
    
    cursor.execute("INSERT INTO sentences_text_g (sentence) VALUES (%s)", (sentence,))
    conn.commit()
    
    end_time = time.time()
    insertion_times.append(end_time - start_time)

print("\n--- RESULTATS [G0]: Inserció de text pur ---")
print(f"Temps mínim: {min(insertion_times):.6f} segons")
print(f"Temps màxim: {max(insertion_times):.6f} segons")
print(f"Temps mitjà: {statistics.mean(insertion_times):.6f} segons")
print(f"Desviació estàndard: {statistics.stdev(insertion_times):.6f} segons")

cursor.close()
conn.close()