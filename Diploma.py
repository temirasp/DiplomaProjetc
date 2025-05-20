import re
import hashlib
import sqlite3
import os
from hashlib import sha256
import json
from neural_index import generate_neural_index
from search_index import compute_search_index_and_recommend
from HAC import setup_database
from HAC import index_directory
from HAC import parallel_index_directory

from deduplication import process_all_docs
from deduplication import restore_file
import time
import tracemalloc

def get_key_from_password(password: str) -> bytes:

    return sha256(password.encode()).digest()








def get_recommended_filenames(word, db_path="word_index.db"):
   
    word_hash = hashlib.sha256(word.encode()).hexdigest()

   
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
       
        cursor.execute("""
            SELECT filename FROM word_metrics
            WHERE hash = ? AND is_recommended = 1
        """, (word_hash,))
        
        results = cursor.fetchall()

        if results:
            print("Recommended filenames:")
            for row in results:
                print("-", row[0])
        else:
            print("No recommended files found for this word.")

    except sqlite3.Error as e:
        print("Database error:", e)
    finally:
        conn.close()

def get_password_id_for_file(filename):
    conn = sqlite3.connect("word_index.db")
    c = conn.cursor()
    c.execute("SELECT pass FROM file WHERE filename = ?", (filename,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def load_keys(path="keys.json"):
    with open(path, "r") as f:
        keys_data = json.load(f)
    return {
        k: {
            "salt": bytes.fromhex(v["salt"])
        }
        for k, v in keys_data.items()
    }

def get_password():
    password = 'temir'.strip().encode()
    return password

def restore_file_by_name():
    metadata_dir = "metadata"
    filename = input('Enter filename ').strip()
    metadata_file = os.path.join(metadata_dir, filename + ".json")
    if not os.path.exists(metadata_file):
        print(f"Metadata for '{filename}' not found in '{metadata_dir}'.")
        return

    password_id = get_password_id_for_file(filename)
    if password_id is None:
        print(f"No entry found for '{filename}'.")
        return
    print(f"Password ID for '{filename}' is: {password_id}")
     
    pas=get_password()

    keys_data = load_keys()
    char_id = str(password_id)
    if char_id not in keys_data:
        print(f"Password ID '{char_id}' not found in keys.json.")
        return

    salt = keys_data[char_id]["salt"]  

    output_file = os.path.join("restored", filename)
    
    
    restore_file(metadata_file, output_file, pas, salt)

def get_total_size(folder):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size

def function(folder_path, conn):
    for filename in os.listdir(folder_path):
        name1=filename

            
            

# --- Example usage ---

if __name__ == '__main__':
    
    tracemalloc.start()

    
    
    start1 = time.time()
    conn = setup_database()
    docs_folder = 'docs'  
    parallel_index_directory(docs_folder, conn)
    conn.close()
    index_time1 = time.time() - start1
    print(f"HAC time: {index_time1:.2f} seconds")
    
    start2 = time.time()
    generate_neural_index()
    index_time2 = time.time() - start2
    print(f"generate_neural_index time: {index_time2:.2f} seconds")

    start = time.time()
    
    compute_search_index_and_recommend()
    index_time = time.time() - start
    print(f"compute_search_index_and_recommend time: {index_time:.2f} seconds")


 
    startDEDUP = time.time()
    
    process_all_docs()
    
    DEDUP_time = time.time() - startDEDUP
    print(f"DEDUPLICATION time: {DEDUP_time:.2f} seconds")
    
    

    KEYWORD = 'temir'
    startRecomendation = time.time()
    
    get_recommended_filenames(KEYWORD)
    
    query_time = time.time() - startRecomendation
    print(f"Time to give recomendations: {query_time:.2f} seconds")
    

    startSearch = time.time()
    
    
        
    restore_file_by_name()
    
    
    query_time = time.time() - startSearch
    print(f"Search time: {query_time:.2f} seconds")
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Peak memory: {peak / 10**6} MB")
    docs_dir = "docs"
    blocks_dir = "blocks"

    original_size = get_total_size(docs_dir)
    dedup_size = get_total_size(blocks_dir)

    print(f"Original size: {original_size / (1024 * 1024):.2f} MB")
    print(f"Deduplicated size: {dedup_size / (1024 * 1024):.2f} MB")

    if original_size > 0:
      dedup_ratio = (original_size - dedup_size) / original_size
      print(f"Deduplication ratio: {dedup_ratio:.2%}")
   

    