import sqlite3

conn = sqlite3.connect('models.db')
conn.execute("""
Create table if not exists train_results (
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             model_name TEXT NOT NULL,
             model_type TEXT,
             rmse REAL,
             mae REAL,
             r2 REAL,
             model_path TEXT,
             filename TEXT,
             train_size REAL,
             train_samples INTEGER,
             test_samples INTEGER,
             n_features INTEGER,
             created_at TEXT DEFAULT current_timestamp 
             )
""")

conn.commit()
conn.close()