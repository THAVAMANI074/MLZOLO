# Import necessary libraries
import joblib
import pandas as pd
import streamlit as st

# Load the trained Random Forest model
model_file = "random_forest_model.pkl"
model = joblib.load(model_file)

# Title and description for Streamlit app
st.title("🎬 Random Forest Audience Rating Prediction")
st.markdown(
    """
    **Welcome!**  
    This app predicts the audience rating for a movie using a trained Random Forest model.  
    Adjust inputs on the left, and see predictions on the right!
    """
)

# Create a two-column layout
col1, col2 = st.columns(2)


# Function to prepare input features for the Random Forest model
def prepare_features():
    """
    Prepare the feature set (X) for Random Forest prediction using user inputs.
    """
    # Collect inputs from session state or defaults
    movie_title = st.session_state.get("movie_title", "Blockbuster Movie")
    movie_info = st.session_state.get("movie_info", "An outstanding critically acclaimed movie")
    critics_consensus = st.session_state.get("critics_consensus", "Overwhelmingly positive reviews")
    rating = st.session_state.get("rating", "PG-13")
    genre = st.session_state.get("genre", "Drama")
    directors = st.session_state.get("directors", "Famous Director")
    writers = st.session_state.get("writers", "Top Screenwriter")
    cast = st.session_state.get("cast", "Famous Actor A, Famous Actor B")
    in_theaters_date = st.session_state.get("in_theaters_date", "2024-12-01")
    on_streaming_date = st.session_state.get("on_streaming_date", "2025-01-01")
    runtime = st.session_state.get("runtime", 150)
    studio_name = st.session_state.get("studio_name", "Top Studio")
    tomatometer_status = st.session_state.get("tomatometer_status", "Certified Fresh")
    tomatometer_rating = st.session_state.get("tomatometer_rating", 85)
    tomatometer_count = st.session_state.get("tomatometer_count", 550)

    # Encode the tomatometer status into numeric form
    status_mapping = {"Fresh": 0, "Certified Fresh": 1, "Rotten": 2}
    tomatometer_status_encoded = status_mapping[tomatometer_status]

    # Combine all features into a DataFrame
    data = pd.DataFrame(
        {
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
            "tomatometer_status_encoded": [tomatometer_status_encoded],
            "tomatometer_rating": [tomatometer_rating],
            "tomatometer_count": [tomatometer_count],
        }
    )
    return data


# Inputs on the left column
with col1:
    st.header("📋 Input Movie Details")
    st.text_input("Movie Title", "Blockbuster Movie", key="movie_title")
    st.text_area("Movie Info", "An outstanding critically acclaimed movie", key="movie_info")
    st.text_input("Critics Consensus", "Overwhelmingly positive reviews", key="critics_consensus")
    st.selectbox("Rating", ["G", "PG", "PG-13", "R", "NC-17"], index=2, key="rating")
    st.text_input("Genre", "Drama", key="genre")
    st.text_input("Director(s)", "Famous Director", key="directors")
    st.text_input("Writer(s)", "Top Screenwriter", key="writers")
    st.text_input("Cast", "Famous Actor A, Famous Actor B", key="cast")
    st.date_input("In Theaters Date", key="in_theaters_date")
    st.date_input("On Streaming Date", key="on_streaming_date")
    st.number_input("Runtime (in minutes)", min_value=1, max_value=500, value=150, key="runtime")
    st.text_input("Studio Name", "Top Studio", key="studio_name")
    st.selectbox(
        "Tomatometer Status",
        ["Fresh", "Certified Fresh", "Rotten"],
        index=1,
        key="tomatometer_status"
    )
    st.slider(
        "Tomatometer Rating",
        min_value=10,
        max_value=100,
        value=85,
        key="tomatometer_rating"
    )
    st.number_input("Tomatometer Count", min_value=1, value=550, key="tomatometer_count")

# Predictions on the right column
with col2:
    st.header("🎯 Prediction Results")
    if st.button("Predict Audience Rating"):
        # Prepare the features (X)
        X = prepare_features()

        # Extract relevant features for the model
        model_features = X[["tomatometer_status_encoded", "tomatometer_rating", "tomatometer_count"]]

        # Predict using the Random Forest model
        y_pred = model.predict(model_features)[0]

        # Display prediction results
        st.success(f"**Predicted Audience Rating:** {y_pred:.2f}")

        # Display the feature set used for prediction (all columns)
        st.write("### Features Used for Prediction:")
        st.dataframe(X, use_container_width=True)
    else:
        st.info("👈 Adjust the details on the left and click **Predict Audience Rating 🎯**.")
