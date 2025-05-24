
# जरूरी लाइब्रेरी इंस्टॉल करें
!apt-get update
!apt-get install -y libzbar0
!pip install numpy pandas scikit-learn tensorflow python-whois opencv-python-headless pyzbar requests beautifulsoup4

import numpy as np
import pandas as pd
import json
import sqlite3
import urllib.parse
import requests
import cv2
from pyzbar.pyzbar import decode
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pickle

# ------------------------- डेटा प्रीप्रोसेसिंग -------------------------
def load_and_clean_data(file_path):
    try:
        # JSON डेटा लोड करें और साफ करें
        with open(file_path) as f:
            raw_data = json.load(f)
        
        # डेटा क्लीनिंग
        clean_data = [
            entry for entry in raw_data
            if 'url' in entry 
            and isinstance(entry['url'], str) 
            and entry['url'].strip() != ''
        ]
        
        df = pd.DataFrame(clean_data)
        df['url'] = df['url'].astype(str).str.strip()
        df = df.dropna(subset=['url'])
        return df
    except Exception as e:
        print(f"डेटा लोड करने में त्रुटि: {str(e)}")
        return pd.DataFrame()

# डेटा लोड करें
data_path = '/content/drive/MyDrive/fishing_dataset.json'
df = load_and_clean_data(data_path)
print(f"सफलतापूर्वक लोड किए गए डेटा के रो: {len(df)}")

# ------------------------- मॉडल सेटअप -------------------------
def create_model(text_input_shape, feature_input_shape):
    # टेक्स्ट इनपुट पाइपलाइन
    text_input = Input(shape=(1, text_input_shape), name='text_input')
    lstm_layer = LSTM(64, return_sequences=True)(text_input)
    lstm_layer = Dropout(0.3)(lstm_layer)
    lstm_layer = LSTM(32)(lstm_layer)

    # फीचर इनपुट पाइपलाइन
    feature_input = Input(shape=(feature_input_shape,), name='feature_input')
    dense_layer = Dense(16, activation='relu')(feature_input)
    dense_layer = Dropout(0.3)(dense_layer)

    # संयुक्त मॉडल
    combined = Concatenate()([lstm_layer, dense_layer])
    combined = Dense(24, activation='relu')(combined)
    output = Dense(1, activation='sigmoid')(combined)

    model = Model(inputs=[text_input, feature_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ------------------------- फीचर एक्सट्रैक्शन -------------------------
def extract_features_from_url(url):
    try:
        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname or ''
        
        features = {
            'length': len(url),
            'num_dots': url.count('.'),
            'num_hyphens': url.count('-'),
            'num_slash': url.count('/'),
            'has_https': int(parsed.scheme == 'https'),
            'has_port': int(parsed.port is not None),
            'has_query': int(len(parsed.query) > 0),
            'path_length': len(parsed.path),
            'digit_count': sum(c.isdigit() for c in url),
            'special_chars': sum(c in '!@#$%^&*()' for c in url),
            'is_shortened': int(any(x in hostname for x in ['bit.ly', 'goo.gl'])),
            'url_entropy': calculate_entropy(url)
        }
        return list(features.values())
    except:
        return [0]*12  # फीचर्स की संख्या के अनुसार

def calculate_entropy(text):
    from collections import Counter
    import math
    counts = Counter(text)
    probs = [c/len(text) for c in counts.values()]
    return -sum(p * math.log(p) for p in probs)

# ------------------------- ट्रेनिंग पाइपलाइन -------------------------
# डेटा स्प्लिट
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# TF-IDF वेक्टराइजेशन
vectorizer = TfidfVectorizer(max_features=500)
X_train_text = vectorizer.fit_transform(train_df['url']).toarray().reshape(-1, 1, 500)
X_test_text = vectorizer.transform(test_df['url']).toarray().reshape(-1, 1, 500)

# फीचर एक्सट्रैक्शन
X_train_features = np.array([extract_features_from_url(url) for url in train_df['url']])
X_test_features = np.array([extract_features_from_url(url) for url in test_df['url']])

# मॉडल ट्रेनिंग
model = create_model(X_train_text.shape[2], X_train_features.shape[1])
history = model.fit(
    [X_train_text, X_train_features], train_df['label'],
    validation_data=([X_test_text, X_test_features], test_df['label']),
    epochs=15,
    batch_size=32,
    verbose=1
)

# ------------------------- मूल्यांकन -------------------------
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

def evaluate_model(model, X_test_text, X_test_features, y_true):
    y_pred = (model.predict([X_test_text, X_test_features]) > 0.5).astype(int)
    print(f"अंतिम टेस्ट सटीकता: {accuracy_score(y_true, y_pred):.2f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.2f}")

plot_training_history(history)
evaluate_model(model, X_test_text, X_test_features, test_df['label'])

# ------------------------- यूटिलिटी फंक्शन्स -------------------------
def setup_database():
    conn = sqlite3.connect('user_history.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS history 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     url TEXT UNIQUE, 
                     result TEXT,
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    return conn

def analyze_html_content(url):
    try:
        response = requests.get(url, timeout=5, verify=True)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # सुरक्षा विश्लेषण
        security_checks = {
            'https': url.startswith('https://'),
            'ssl_valid': response.ok,  # सरलीकृत चेक
            'hidden_elements': len(soup.find_all(style=lambda x: 'display:none' in x)),
            'external_scripts': len([s for s in soup.find_all('script') if 'http' in (s.get('src') or '')]),
            'suspicious_keywords': sum(keyword in response.text.lower() for keyword in ['password', 'login', 'banking'])
        }
        return security_checks
    except Exception as e:
        return {'error': str(e)}

def check_url(url):
    conn = setup_database()
    cursor = conn.cursor()
    
    try:
        # पूर्वानुमान
        text_input = vectorizer.transform([url]).toarray().reshape(1, 1, -1)
        features = np.array([extract_features_from_url(url)])
        prediction = model.predict([text_input, features])[0][0]
        
        # विश्लेषण
        html_analysis = analyze_html_content(url)
        risk_level = "उच्च जोखिम" if prediction > 0.7 else "मध्यम" if prediction > 0.4 else "कम"
        
        # डेटाबेस में सहेजें
        result = {
            'risk_score': float(prediction),
            'risk_level': risk_level,
            'html_analysis': html_analysis,
            'url': url
        }
        cursor.execute('''
            INSERT OR REPLACE INTO history (url, result) 
            VALUES (?, ?)
        ''', (url, json.dumps(result)))
        conn.commit()
        
        return result
    except Exception as e:
        return {'error': str(e)}
    finally:
        conn.close()

# ------------------------- यूजर इंटरफेस -------------------------
def main():
    # QR कोड स्कैनर
    from google.colab import files
    print("QR कोड इमेज अपलोड करें (या Enter दबाएं):")
    uploaded = files.upload()
    
    if uploaded:
        for filename in uploaded.keys():
            decoded_url = decode_qr(filename)
            if decoded_url:
                print(f"स्कैन किया गया URL: {decoded_url}")
                print(json.dumps(check_url(decoded_url), indent=2))
    else:
        while True:
            url = input("URL दर्ज करें (बाहर निकलने के लिए 'exit' टाइप करें): ").strip()
            if url.lower() == 'exit':
                break
            if url:
                print(json.dumps(check_url(url), indent=2))

def decode_qr(image_path):
    try:
        img = cv2.imread(image_path)
        decoded = decode(img)
        return decoded[0].data.decode('utf-8') if decoded else None
    except Exception as e:
        print(f"QR डिकोड त्रुटि: {str(e)}")
        return None

if __name__ == "__main__":
    main()
