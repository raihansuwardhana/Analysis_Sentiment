import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import streamlit as st
import re  # Import pustaka regular expression

# Download necessary NLTK resource
nltk.download('punkt')

# Unduh dataset twitter_samples dari NLTK
nltk.download('twitter_samples')

nltk.download('stopwords')

# Explicitly set NLTK data path for Streamlit Share
nltk.data.path.append('/app/nltk_data')

# Fungsi untuk membersihkan teks dari angka, tanda baca, simbol, URL, dan emotikon
def clean_text(text):
    # Hapus URL
    text = re.sub(r'http\S+', '', text)
    
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    
    # Hapus tanda baca dan simbol (kecuali @ untuk menyimpan username)
    text = re.sub(r'[^\w\s]', '', text.replace('_', ' '))
    
    # Hapus emotikon
    text = re.sub(r'(:\s?\)|:\s?D|;\s?\))', '', text)
    
    return text

# Fungsi untuk mengekstrak fitur dari teks komentar
def extract_features(words):
    return dict([(word, True) for word in words])

# Mengambil dataset twitter_samples
from nltk.corpus import twitter_samples
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# Menggabungkan dataset positif dan negatif
dataset = [(tweet, 'Sentiment Positive') for tweet in positive_tweets] + [(tweet, 'Sentiment Negative') for tweet in negative_tweets]

# Mengacak urutan dataset
import random
random.shuffle(dataset)

# Mengambil daftar berisi semua kata dari seluruh tweet
all_words = [word.lower() for tweet, _ in dataset for word in word_tokenize(tweet)]

# Mengambil 2000 kata unik yang paling umum
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:2000]

# Menghapus stop words dari kata-kata unik
stop_words = set(stopwords.words('english'))
word_features = [word for word in word_features if word not in stop_words]

# Menyiapkan data latih dan data uji
featuresets = [(extract_features(word_tokenize(clean_text(tweet.lower()))), sentiment) for (tweet, sentiment) in dataset]
train_set = featuresets[:int(len(featuresets) * 0.8)]
test_set = featuresets[int(len(featuresets) * 0.8):]

# Melatih klasifikasi Naive Bayes
classifier = NaiveBayesClassifier.train(train_set)

# Menghitung akurasi klasifikasi pada data uji
accuracy = nltk.classify.accuracy(classifier, test_set)

# Fungsi untuk melakukan stemming pada kata-kata
stemmer = PorterStemmer()
def stem_words(words):
    return [stemmer.stem(word) for word in words]

# Antarmuka Pengguna dengan Streamlit
st.set_page_config(page_title="Sentiment Analysis on Text", page_icon=":grinning:", layout="wide")
st.title("Sentiment Analysis on Text")

# Tampilan header
header, content = st.columns([2, 3])
with header:
    st.write("Welcome to Sentiment Analysis of Text from Social Media!")
    st.image("https://media.istockphoto.com/id/939688472/vector/happy-and-sad-face-icons-smileys.jpg?s=170667a&w=0&k=20&c=aPWGAnZORi77uDHyUdZqzPUao7CG05uDP-SVAQkL0hQ=", width=200)
with content:
    st.write("Source of Dataset: NLTK twitter_samples")
    st.write("Classification Accuracy on Test Data: {:.2%}".format(accuracy))

# Tampilan instruksi
st.header("How To Use")
st.write("1. Enter Text or Comments in the Input Field.")
st.write("2. Click 'Analysis' Button to View the Analysis results.")
st.write("3. The Analysis Results Will be Displayed Below")

# Input teks komentar dari pengguna
user_input = st.text_input("Input the comment text:")

# Melakukan analisis sentimen pada teks komentar media sosial
if st.button("Analysis"):
   # Preprocessing teks
    cleaned_text = clean_text(user_input)
    lowercase_text = cleaned_text.lower()
    
    # Case folding (Mengubah teks menjadi huruf kecil)
    tokenized_words = word_tokenize(lowercase_text)

    # Penghapusan Stopwords
    filtered_words = [word for word in tokenized_words if word not in stop_words]
   
    # Stemming (Pengubahan kata-kata menjadi bentuk dasar)
    stemmed_words = stem_words(filtered_words)
    
     # Tampilkan hasil preprocessing di sidebar
    st.sidebar.title("Preprocessing steps:")
    st.sidebar.write("1. Cleaning (Removing URLs, numbers, punctuation marks, symbols, and emoticons):", cleaned_text)
    st.sidebar.write("2. Case Folding (Change text to lowercase):", lowercase_text)
    st.sidebar.write("3. Tokenisasi (Convert text into tokens):", tokenized_words)
    st.sidebar.write("4. Penghapusan Stopwords:", filtered_words)
    st.sidebar.write("5. Stemming (Converting words into base forms):", stemmed_words)
   
    st.write(':point_left: You can see preprocessing step at left')
    
    # Menampilkan kata-kata yang menjadi alasan terdeteksi sentimen negatif atau positif
    word_features_set = set(word_features)
    words_in_input = [word for word in stemmed_words if word in word_features_set]
    st.write("Words that contribute to Sentiment Analysis:")
    if not words_in_input:
        st.write("No words contribute to the sentiment analysis result.")
    else:
        st.write(words_in_input)

    # Menampilkan kalimat dengan kata-kata penyebab yang di-highlight
    if not words_in_input:
        st.write("No words Cause.")
    else:
        highlighted_sentence = " ".join([f"<span style='background-color: #ffff00'>{word}</span>" if word in words_in_input else word for word in stemmed_words])
        st.write("Sentences with Cause Words:")
        st.markdown(highlighted_sentence, unsafe_allow_html=True)

    sentiment = classifier.classify(extract_features(stemmed_words))
    st.write("Sentiment Analysis Results:", sentiment if words_in_input else "Sorry, Can not Clasified")
