import pandas as pd
import psycopg2

# Load CSV
df = pd.read_csv("data/processed/reviews_processed.csv")

# Database connection
conn = psycopg2.connect(
    host="localhost",
    database="bank_reviews",
    user="postgres",
    password="ephi1888"  
)
cur = conn.cursor()

# Insert unique banks
banks = df[['bank_name', 'bank_code']].drop_duplicates()

for _, row in banks.iterrows():
    cur.execute("""
        INSERT INTO banks (bank_name, bank_code)
        VALUES (%s, %s)
        ON CONFLICT (bank_name) DO NOTHING;
    """, (row['bank_name'], row['bank_code']))

conn.commit()

# Fetch bank_id mapping
cur.execute("SELECT bank_id, bank_name FROM banks;")
bank_map = {name: id for (id, name) in cur.fetchall()}

# Insert reviews
for _, row in df.iterrows():
    cur.execute("""
        INSERT INTO reviews
        (bank_id, review_text, rating, review_date, review_year, review_month,
         user_name, thumbs_up, text_length, source)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        bank_map[row['bank_name']],
        row['review_text'],
        row['rating'],
        row['review_date'],
        row['review_year'],
        row['review_month'],
        row['user_name'],
        row['thumbs_up'],
        row['text_length'],
        row['source']
    ))

conn.commit()
cur.close()
conn.close()

print("Data inserted successfully!")
