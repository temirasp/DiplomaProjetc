import sqlite3
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
# --- step 1 ---

def get_document_vectors(db_path='word_index.db'):
  
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('SELECT DISTINCT filename FROM word_metrics')
    files = [row[0] for row in cursor.fetchall()]

    doc_vectors = []

    for filename in files:
        cursor.execute('''
            SELECT tf, idf, M, Q FROM word_metrics
            WHERE filename = ?
        ''', (filename,))
        rows = cursor.fetchall()
   
        if rows:
            rows_array = np.array(rows)
            
            
            avg_vector = rows_array.mean(axis=0)  # the average value of TF, IDF, M, Q
            
            doc_vectors.append((filename, avg_vector))

    conn.close()
    return doc_vectors

# --- clustering k-means ---
def apply_kmeans(doc_vectors, n_clusters=3):
    print("apply_kmeans")
    vectors = np.array([vec for _, vec in doc_vectors])
    print("vectors shape:", vectors.shape)

    scaler = StandardScaler()
    vectors_scaled = scaler.fit_transform(vectors)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(vectors_scaled)
    return labels, kmeans

def apply_kmeans(doc_vectors, n_clusters=3):
    
    vectors = np.array([vec for _, vec in doc_vectors])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
  
    labels = kmeans.fit_predict(vectors)
   
    return labels, kmeans

# --- training ---

def train_neural_index_model(doc_vectors, labels):
   
    X = np.array([vec for _, vec in doc_vectors])
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MLPClassifier(hidden_layer_sizes=(6,), activation='relu', max_iter=1200, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
 
    return model

# --- adding neural_index column ---

def add_neural_index_column(db_path='word_index.db'):
   
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # checking if the column already exists
    cursor.execute('PRAGMA table_info(word_metrics)')
    columns = [col[1] for col in cursor.fetchall()]
    if 'neural_index' not in columns:
        cursor.execute('ALTER TABLE word_metrics ADD COLUMN neural_index REAL')
        conn.commit()
    conn.close()

# --- applying model for words and saving ---

def apply_model_to_words(model, db_path='word_index.db'):
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('SELECT DISTINCT filename FROM word_metrics')
    files = [row[0] for row in cursor.fetchall()]

    for filename in files:
        cursor.execute('''
            SELECT hash, tf, idf, M, Q FROM word_metrics
            WHERE filename = ?
        ''', (filename,))
        rows = cursor.fetchall()  # [(hash, tf, idf, M, Q), ...]

        for word_hash, tf, idf, M, Q in rows:
            feature_vector = np.array([[tf, idf, M, Q]])
            probabilities = model.predict_proba(feature_vector)[0]
            max_prob = np.max(probabilities)


            cursor.execute('''
                UPDATE word_metrics SET neural_index = ?
                WHERE hash = ? AND filename = ?
            ''', (max_prob, word_hash, filename))

    conn.commit()
    conn.close()


# --- main ---

def generate_neural_index():
    
   
    db_path = 'word_index.db'

    add_neural_index_column(db_path)

    doc_vectors = get_document_vectors(db_path)
    labels, kmeans_model = apply_kmeans(doc_vectors, n_clusters=3)
    unique_labels, counts = np.unique(labels, return_counts=True)
   

    model = train_neural_index_model(doc_vectors, labels)

    apply_model_to_words(model, db_path)

    
