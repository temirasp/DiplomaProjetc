import sqlite3
import numpy as np



def compute_search_index_and_recommend(db_path='word_index.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # adding new columns if they don't exist
    cursor.execute('PRAGMA table_info(word_metrics)')
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'search_index' not in columns:
        cursor.execute('ALTER TABLE word_metrics ADD COLUMN search_index REAL')
    
    if 'is_recommended' not in columns:
        cursor.execute('ALTER TABLE word_metrics ADD COLUMN is_recommended INTEGER')
    
    conn.commit()

    # aquairing data
    cursor.execute('SELECT hash, filename, HAC, neural_index FROM word_metrics')
    rows = cursor.fetchall()

    for word_hash, filename, hac, nei in rows:
        if hac is None or nei is None:
            continue  # skip if HAC or neural_index is None

        try:
            si = np.log1p(hac + nei)
        except ValueError:
            continue  # skip if log is undefined

        ub = nei + nei * 0.20
        lb = nei - nei * 0.20

        is_rec = 1 if lb <= si <= ub else 0

        cursor.execute('''
            UPDATE word_metrics
            SET search_index = ?, is_recommended = ?
            WHERE hash = ? AND filename = ?
        ''', (si, is_rec, word_hash, filename))

    conn.commit()
    conn.close()
  
