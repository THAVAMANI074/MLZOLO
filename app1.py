import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Load the trained model
model_file = "random_forest_model.pkl"
model = joblib.load(model_file)

# Load the label encoder
label_encoder_file = "label_encoder.pkl"
label_encoder = joblib.load(label_encoder_file)

# Title and description for Streamlit app
st.title("🎬 Audience Rating Prediction App")
st.write(
    "Use this tool to predict the audience rating for a movie based on adjustable tomatometer ratings and other movie details.")

# Function to create movie data
def create_movie_data():
    """
    Prepare a DataFrame with user-input movie data.
    """
    # Get inputs from user inputs in the app
    movie_title = st.text_input("Movie Title", "Blockbuster Movie")
    movie_info = st.text_area("Movie Info", "An outstanding critically acclaimed movie")
    critics_consensus = st.text_input("Critics Consensus", "Overwhelmingly positive reviews")
    rating = st.selectbox("Rating", ["G", "PG", "PG-13", "R", "NC-17"], index=2)
    genre = st.text_input("Genre", "Drama")
    directors = st.text_input("Director(s)", "Famous Director")
    writers = st.text_input("Writer(s)", "Top Screenwriter")
    cast = st.text_input("Cast", "Famous Actor A, Famous Actor B")
    in_theaters_date = st.date_input("In Theaters Date")
    on_streaming_date = st.date_input("On Streaming Date")
    runtime = st.number_input("Runtime (in minutes)", min_value=1, max_value=500, value=150)
    studio_name = st.text_input("Studio Name", "Top Studio")
    tomatometer_status = st.selectbox("Tomatometer Status", ["Fresh", "Certified Fresh", "Rotten"], index=1)
    tomatometer_rating = st.number_input("Tomatometer Rating", min_value=0, max_value=100, value=85)
    tomatometer_count = st.number_input("Tomatometer Count", min_value=1, value=550)

    data = {
        "movie_title": [movie_title],
        "movie_info": [movie_info],
        "critics_consensus": [critics_consensus],
        "rating": [rating],
        "genre": [genre],
        "directors": [directors],
        "writers": [writers],
        "cast": [cast],
        "in_theaters_date": [in_theaters_date],
        "on_streaming_date": [on_streaming_date],
        "runtime_in_minutes": [runtime],
        "studio_name": [studio_name],
        "tomatometer_status": [tomatometer_status],
        "tomatometer_rating": [tomatometer_rating],
        "tomatometer_count": [tomatometer_count],
    }
    return pd.DataFrame(data)

# Generate predictions for the selected rating
if st.button("Predict Audience Rating"):
    # Create movie data
    movie_data = create_movie_data()

    # Encode the categorical column
    movie_data["tomatometer_status_encoded"] = label_encoder.transform(movie_data["tomatometer_status"])

    # Define the features for prediction
    X = movie_data[["tomatometer_status_encoded", "tomatometer_rating", "tomatometer_count"]]

    # Predict audience rating
    predicted_rating = model.predict(X)[0]

    # Display the result
    st.subheader("🎯 Prediction Results")
    st.write(f"**Predicted Audience Rating:** {predicted_rating:.2f}")

    # Display final movie features
    st.write("**Movie Features Used for Prediction:**")
    st.dataframe(movie_data)