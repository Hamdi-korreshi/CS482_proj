import pandas as pd
import psycopg2

# Read CSV file (adjust the file path accordingly)
csv_file = 'ex.csv'
df = pd.read_csv(csv_file)

# Database connection details
db_params = {
    'dbname': 'images',
    'user': 'postgres',
    'password': 'hamdi',
    'host': '172.20.232.187',  # Change if your database is hosted elsewhere
    'port': 5432,  # Default PostgreSQL port
}

try:
    # Connect to the database
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    # Create the VECTORS table (if it doesn't exist)
    create_table_query = """
        CREATE TABLE IF NOT EXISTS VECTORS (
            vector_id SERIAL PRIMARY KEY,
            vector_data JSONB
        );
    """
    cursor.execute(create_table_query)
    conn.commit()

    # Insert data into the table
    for _, row in df.iterrows():
        insert_query = """
            INSERT INTO VECTORS (vector_data)
            VALUES (%s);
        """
        cursor.execute(insert_query, (row.to_json(),))

    conn.commit()
    print(f"Data loaded successfully into the VECTORS table.")

except Exception as e:
    print(f"Error: {e}")

finally:
    cursor.close()
    conn.close()