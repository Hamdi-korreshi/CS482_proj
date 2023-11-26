from pgvector.psycopg import register_vector
import psycopg
import os
import numpy as np


conn = psycopg.connect(dbname='postgres', host="localhost", port=5432)
conn.execute('CREATE EXTENSION IF NOT EXISTS vector;')
register_vector(conn)
conn.execute('DROP TABLE IF EXISTS text;')
conn.execute('CREATE TABLE text (id bigserial PRIMARY KEY, embedding vector(300));')

cur = conn.cursor()

for filename in os.listdir('text-embeddings'):
    if filename.endswith('.npy'):
        # Load the embedded vectors from the file
        vectors = np.load(os.path.join('text-embeddings', filename))

        # Insert the vectors into the database
        cur.execute("INSERT INTO text (embedding) VALUES (%s)", (vectors,))

result = conn.execute('SELECT id FROM text ORDER BY embedding <=> %s LIMIT 10;', (vectors,)).fetchall()
print(result)
print("done")