# Trigger redeploy
import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader
import joblib

# Load data and model
ratings = pd.read_csv('Ratings.csv', encoding='latin-1')
ratings = ratings[['User-ID', 'ISBN', 'Book-Rating']]
ratings.columns = ['user_id', 'book_id', 'rating']
filtered = ratings.groupby('user_id').filter(lambda x: len(x) > 10)

reader = Reader(rating_scale=(0, 10))
data = Dataset.load_from_df(filtered[['user_id', 'book_id', 'rating']], reader)

model = joblib.load('svd_model.pkl')  # previously saved with joblib.dump(model, 'svd_model.pkl')

def recommend_books(user_id, df, model, n=5):
    all_books = df['book_id'].unique()
    read_books = df[df['user_id'] == user_id]['book_id'].tolist()
    unread_books = [book for book in all_books if book not in read_books]
    predictions = [model.predict(user_id, book) for book in unread_books]
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    return [pred.iid for pred in top_n]

# Streamlit UI
st.title("📚 Book Recommender")
user_input = st.text_input("Enter User ID:")
if user_input:
    try:
        user_id = int(user_input)
        if user_id in filtered['user_id'].values:
            recommendations = recommend_books(user_id, filtered, model)
            st.write("Top Recommendations:", recommendations)
        else:
            st.warning("User ID not found in dataset.")
    except ValueError:
        st.error("Please enter a valid numeric User ID.")
