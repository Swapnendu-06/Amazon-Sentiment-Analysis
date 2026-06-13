import streamlit as st
import pandas as pd
import pickle
import re
import requests
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import nltk
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK data is available
for resource in ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True)


# --- Helper functions (previously in analysis.py) ---

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+|\[.*?\]|[^a-zA-Z\s]+|\w*\d\w*', ' ', text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    processed = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(processed)


def scrap_page(reviews_url):
    headers = {
        'authority': 'www.amazon.com',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'accept-language': 'en-US,en;q=0.9',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
    }
    response = requests.get(reviews_url, headers=headers)
    soup = BeautifulSoup(response.text, 'lxml')
    boxes = soup.select('div[data-hook="review"]')
    reviews = []
    for box in boxes:
        try:
            stars = box.select_one('[data-hook="review-star-rating"]').text.strip().split(' out')[0]
            title = box.select_one('[data-hook="review-title"]').text
            title = re.sub(r'\d.\d out of 5 stars', ' ', title).strip()
            description = box.select_one('[data-hook="review-body"]').text.strip()
            reviews.append({'Stars': float(stars), 'Review': title + ' ' + description})
        except Exception:
            continue
    return pd.DataFrame(reviews)


def fetch_live_reviews(url):
    pattern = r'(https:\/\/www\.amazon\.[a-z]+(\.[a-z]+)?\/[^\/]+)\/dp\/([^\/]+)\/?\??.*'
    match = re.match(pattern, url)
    if match:
        product_url, _, product_id = match.groups()
        reviews_url = f'{product_url}/product-reviews/{product_id}/'
        return scrap_page(reviews_url)
    else:
        raise ValueError('Invalid Amazon URL format')


# --- Streamlit App ---

# Load Artifacts
with open('models.p', 'rb') as mod:
    data = pickle.load(mod)
vect = data['vectorizer']

st.title('AI-Driven Sentiment Intelligence')
st.markdown('A robust NLP dashboard for real-time commercial product analysis.')

classifier = st.sidebar.radio('Select Classification Engine', ['Linear SVC (Recommended)', 'Logistic Regression'])
model = data['svm'] if 'SVC' in classifier else data['logreg']

st.subheader('1. Live Amazon Product Analysis')
url_review = st.text_input('Enter Amazon Product URL:')
if st.button('Scrape & Analyze Live Data'):
    with st.spinner('Extracting and processing live reviews...'):
        try:
            live_df = fetch_live_reviews(url_review)
            if live_df.empty:
                st.warning('No reviews found. Amazon may be blocking the scraper — try a different URL.')
            else:
                live_df['Cleaned'] = live_df['Review'].apply(preprocess_text)
                X_live = vect.transform(live_df['Cleaned'])
                live_df['Sentiment'] = model.predict(X_live)
                st.success(f'Successfully processed {len(live_df)} reviews.')
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(live_df[['Stars', 'Sentiment', 'Review']].head(10))
                with col2:
                    sent_counts = live_df['Sentiment'].value_counts()
                    fig = go.Figure(data=[go.Pie(labels=sent_counts.index, values=sent_counts.values)])
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f'Error scraping URL: {e}')
