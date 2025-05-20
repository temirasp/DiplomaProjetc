import os
import hashlib
import json
import sqlite3
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding, hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import zstandard as zstd
from docx import Document
import PyPDF2
import openpyxl

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from multiprocessing import Pool, cpu_count
import re
from datetime import datetime
def normalize_text(text: str) -> str:

    text = re.sub(r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b', '[DATE]', text)
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '[DATE]', text)
    
    text = re.sub(r'\b\d{1,2}:\d{2}(:\d{2})?\b', '[TIME]', text)
    
    text = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL]', text)
    
    return text
class RabinFingerprint:
    def __init__(self, window_size=48, polynomial=0x3DA3358B4DC173, mod=2**64):
        self.window_size = window_size
        self.polynomial = polynomial
        self.mod = mod
        self.table = [0] * 256
        self.init_tables()
        self.window = bytearray()
        self.fingerprint = 0

    def init_tables(self):
        for i in range(256):
            fp = i
            for j in range(8):
                if fp & 1:
                    fp = (fp >> 1) ^ self.polynomial
                else:
                    fp >>= 1
            self.table[i] = fp

    def slide(self, new_byte):
        if len(self.window) == self.window_size:
            out_byte = self.window.pop(0)
            self.fingerprint ^= self.table[out_byte]
        self.window.append(new_byte)
        self.fingerprint = ((self.fingerprint << 1) ^ self.table[new_byte]) % self.mod
        return self.fingerprint


def load_keys(path="keys.json"):
    with open(path, "r") as f:
        keys_data = json.load(f)
    return {
        k: {
            "salt": bytes.fromhex(v["salt"])
        } for k, v in keys_data.items()
    }


# Initialize database and ensure table/column exists
def init_db():
 
    conn = sqlite3.connect("word_index.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS file (
            filename TEXT PRIMARY KEY,
            pass INTEGER
        )
    """)
    conn.commit()
    conn.close()


    


# Check if file already processed
def is_file_processed(filename):
    conn = sqlite3.connect("word_index.db")
    c = conn.cursor()
    c.execute("SELECT 1 FROM file WHERE filename=?", (filename,))
    result = c.fetchone()
    conn.close()
    return result is not None


# Save processed file info
def save_file_to_db(filename, file_hash, pass_id):
    conn = sqlite3.connect("word_index.db")
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO file (filename, pass) VALUES (?, ?)",
              (filename, pass_id))
    conn.commit()
    conn.close()



def encrypt_data(data: bytes, raw_password: bytes, salt: bytes, hash_digest: str) -> bytes:
    cctx = zstd.ZstdCompressor(level=10)  
    compressed_data = b'ZSTD' + cctx.compress(data)


    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
        backend=default_backend()
    )
    key = kdf.derive(raw_password)

  
    iv = hashlib.md5(hash_digest.encode()).digest()[:16]
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()

  
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(compressed_data) + padder.finalize()
    return encryptor.update(padded_data) + encryptor.finalize()


def decrypt_data(encrypted_data: bytes, raw_password: bytes, salt: bytes, hash_digest: str) -> bytes:
  
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
        backend=default_backend()
    )
    key = kdf.derive(raw_password)

   
    iv = hashlib.md5(hash_digest.encode()).digest()[:16]
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()

   
    decrypted_padded = decryptor.update(encrypted_data) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    compressed_data = unpadder.update(decrypted_padded) + unpadder.finalize()

    if not compressed_data.startswith(b'ZSTD'):
        raise ValueError("Invalid compressed block format (missing ZSTD header)")
    
    compressed_data = compressed_data[4:]
    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(compressed_data)


import io  # Make sure this is imported at the top of the file

def deduplicate_and_store(file_path, key_data, raw_password: bytes, metadata_folder="metadata"):
   
    storage = {}
    file_structure = []

    if not os.path.exists("blocks"):
        os.mkdir("blocks")
    if not os.path.exists(metadata_folder):
        os.mkdir(metadata_folder)

    initial_blocks = set(os.listdir("blocks"))

    ext = os.path.splitext(file_path)[1].lower()
    is_textual = ext in ['.txt', '.docx', '.pdf', '.xlsx']

    # Extract content accordingly
    
    file_stream = open(file_path, "rb")

    try:
        chunks = content_defined_chunking(file_stream)

        for chunk in chunks:
            hash_digest = hashlib.sha1(chunk).hexdigest()
            block_path = f"blocks/{hash_digest}.bin"

            if hash_digest not in storage:
                if f"{hash_digest}.bin" not in initial_blocks:
                    encrypted_block = encrypt_data(chunk, raw_password, key_data["salt"], hash_digest)
                    with open(block_path, "wb") as block_file:
                        block_file.write(encrypted_block)
                storage[hash_digest] = hash_digest
                file_structure.append(hash_digest)
            else:
                file_structure.append(f"ref:{hash_digest}")
    finally:
        file_stream.close()

    base_name = os.path.basename(file_path)
    meta_path = os.path.join(metadata_folder, base_name + ".json")
    with open(meta_path, "w") as f:
        json.dump({"structure": file_structure}, f)



def restore_file(meta_file_path, output_path, raw_password: bytes, salt: bytes):

    print(raw_password)
    with open(meta_file_path, "r") as f:
        data = json.load(f)

    structure = data["structure"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        for item in structure:
            hash_digest = item.split(":")[1] if item.startswith("ref:") else item
            block_path = f"blocks/{hash_digest}.bin"
            if os.path.exists(block_path):
                with open(block_path, "rb") as block_file:
                    encrypted_block = block_file.read()
                decrypted = decrypt_data(encrypted_block, raw_password, salt, hash_digest)
                f.write(decrypted)
            else:
                print(f"[ERROR] Missing block {hash_digest}!")

    print(f"[+] File restored to {output_path}")


def process_all_docs(docs_dir="docs"):
    init_db()
    filenames = os.listdir(docs_dir)

    with Pool(cpu_count()) as pool:
        pool.map(process_single_file, filenames)
            
def content_defined_chunking(f, min_size=512, avg_size=4096, max_size=8192, window_size=48):
    rabin = RabinFingerprint(window_size)
    buffer = b""
    chunks = []

    mask = avg_size - 1 

    while True:
        byte = f.read(1)
        if not byte:
            if buffer:
                chunks.append(buffer)
            break

        buffer += byte
        fp = rabin.slide(byte[0])

        if len(buffer) >= min_size and (fp & mask) == 0:
            chunks.append(buffer)
            buffer = b""
            rabin = RabinFingerprint(window_size)  

        elif len(buffer) >= max_size:
            chunks.append(buffer)
            buffer = b""
            rabin = RabinFingerprint(window_size)

    return chunks

def process_single_file(filename):
    if is_file_processed(filename):
        
        return

    passwords = load_keys()
    choice = '1'
    raw_password = 'temir'.strip().encode()

    if choice not in passwords:
        print("[ERROR] Invalid password ID. Skipping file.")
        return

    input_path = os.path.join("docs", filename)
    deduplicate_and_store(input_path, passwords[choice], raw_password)
    file_hash = hashlib.sha1(open(input_path, "rb").read()).hexdigest()
    save_file_to_db(filename, file_hash, choice)