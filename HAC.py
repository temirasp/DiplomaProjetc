import re
import hashlib
import sqlite3
import os
import math
from collections import defaultdict
from docx import Document
import getpass
from hashlib import sha256
from PyPDF2 import PdfReader
from neural_index import generate_neural_index
from search_index import compute_search_index_and_recommend




# --- Utility functions ---

def get_word_hash(word):
    return hashlib.sha256(word.encode('utf-8')).hexdigest()

def read_text_from_file(file_path):
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    elif file_path.endswith('.pdf'):
        text = ""
        try:
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error reading PDF file {file_path}: {e}")
        return text
    else:
        raise ValueError("Unsupported file format. Use .txt, .docx or .pdf")

# --- TF-IDF-HAC related functions ---

def calculate_tf(word_counts, total_terms, word):
    return word_counts[word] / total_terms

def calculate_idf(word, all_word_counts):
    num_docs = len(all_word_counts)
    docs_with_word = sum(1 for doc_counts in all_word_counts if word in doc_counts and doc_counts[word] > 0)
   

    if docs_with_word == 0:
        return 0
    return math.log(num_docs / docs_with_word)

def calculate_M(tf, all_tfs):
    denominator = math.sqrt(sum(val**2 for val in all_tfs))
    return tf / denominator if denominator != 0 else 0

def calculate_Q(idf, all_tfs):
    denominator = math.sqrt(sum(val**2 for val in all_tfs))
    return idf / denominator if denominator != 0 else 0

# --- Database setup ---

def setup_database(db_path='word_index.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS words (
            hash TEXT PRIMARY KEY,
            word TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS word_metrics (
            hash TEXT,
            filename TEXT,
            Number INTEGER,
            tf REAL,
            idf REAL,
            M REAL,
            Q REAL,
            HAC REAL,
            PRIMARY KEY (hash, filename),
            FOREIGN KEY (hash) REFERENCES words(hash)
        )
    ''')

    
    conn.commit()
    return conn

# --- Indexing files ---

def index_file(file_path, conn):
    filename = os.path.basename(file_path)
    text = read_text_from_file(file_path)
    words = re.findall(r'\b\w+\b', text.lower())

    word_counts = defaultdict(int)
    for word in words:
        word_counts[word] += 1

    cursor = conn.cursor()

    # Save words into table
    for word in word_counts.keys():
        word_hash = get_word_hash(word)
        cursor.execute('INSERT OR IGNORE INTO words (hash, word) VALUES (?, ?)', (word_hash, word))

    conn.commit()

    # Fetch all files
    cursor.execute('SELECT DISTINCT filename FROM word_metrics')
    all_files = [row[0] for row in cursor.fetchall()]
    all_word_counts = []

    for file in all_files:
        file_word_counts = defaultdict(int)
        cursor.execute('SELECT w.word, wm.Number FROM word_metrics wm JOIN words w ON wm.hash = w.hash WHERE filename = ?', (file,))
        for word, Number in cursor.fetchall():
            file_word_counts[word] = Number
        all_word_counts.append(file_word_counts)

    # Add current document
    all_word_counts.append(word_counts)
    all_files.append(filename)
    current_doc_index = len(all_files) - 1

    total_terms = sum(word_counts.values())

    for word, count in word_counts.items():
        tf = calculate_tf(word_counts, total_terms, word)
        idf = calculate_idf(word, all_word_counts)

        all_tfs = []
        for doc_counts in all_word_counts:
            doc_total = sum(doc_counts.values())
            if doc_total > 0:
                all_tfs.append(doc_counts.get(word, 0) / doc_total)
            else:
                all_tfs.append(0)

        M = calculate_M(tf, all_tfs)
        Q = calculate_Q(idf, all_tfs)
        HAC = M / Q if Q != 0 else 0  

        word_hash = get_word_hash(word)

        cursor.execute('''
            INSERT INTO word_metrics (hash, filename, Number, tf, idf, M, Q, HAC)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(hash, filename) DO UPDATE SET
                Number=excluded.Number,
                tf=excluded.tf,
                idf=excluded.idf,
                M=excluded.M,
                Q=excluded.Q,
                HAC=excluded.HAC
        ''', (word_hash, filename, count, tf, idf, M, Q, HAC))

    conn.commit()

def index_directory(folder_path, conn):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.txt', '.docx', '.pdf')):
            index_file(os.path.join(folder_path, filename), conn)

def index_file_data(file_path):
    filename = os.path.basename(file_path)
    text = read_text_from_file(file_path)
    words = re.findall(r'\b\w+\b', text.lower())

    word_counts = defaultdict(int)
    for word in words:
        word_counts[word] += 1

    return (filename, word_counts)
from multiprocessing import Pool, cpu_count

def parallel_index_directory(folder_path, conn):
    file_paths = [
        os.path.join(folder_path, fname)
        for fname in os.listdir(folder_path)
        if fname.lower().endswith(('.txt', '.docx', '.pdf', '.xlsx'))
    ]

    with Pool(cpu_count()) as pool:
        results = pool.map(index_file_data, file_paths)

 
    all_word_counts = [r[1] for r in results]

    
    cursor = conn.cursor()
    for filename, word_counts in results:
        total_terms = sum(word_counts.values())

        for word, count in word_counts.items():
            tf = calculate_tf(word_counts, total_terms, word)
            idf = calculate_idf(word, all_word_counts)

            all_tfs = []
            for doc_counts in all_word_counts:
                doc_total = sum(doc_counts.values())
                all_tfs.append(doc_counts.get(word, 0) / doc_total if doc_total > 0 else 0)

            M = calculate_M(tf, all_tfs)
            Q = calculate_Q(idf, all_tfs)
            HAC_val = M / Q if Q != 0 else 0
            word_hash = get_word_hash(word)

            cursor.execute('INSERT OR IGNORE INTO words (hash, word) VALUES (?, ?)', (word_hash, word))

            cursor.execute('''
                INSERT INTO word_metrics (hash, filename, Number, tf, idf, M, Q, HAC)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(hash, filename) DO UPDATE SET
                    Number=excluded.Number,
                    tf=excluded.tf,
                    idf=excluded.idf,
                    M=excluded.M,
                    Q=excluded.Q,
                    HAC=excluded.HAC
            ''', (word_hash, filename, count, tf, idf, M, Q, HAC_val))

    conn.commit()

