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
    Adjust inputs in the sidebar and hit **Predict**!
    """
)

# Function to prepare input features for the Random Forest model
def prepare_features(tomatometer_rating):
    """
    Prepare the feature set (X) for Random Forest prediction using user inputs.
    """
    tomatometer_status = st.session_state.get("tomatometer_status", "Certified Fresh")
    tomatometer_count = st.session_state.get("tomatometer_count", 550)

    # Encode the tomatometer status into numeric form
    status_mapping = {"Fresh": 0, "Certified Fresh": 1, "Rotten": 2}
    tomatometer_status_encoded = status_mapping[tomatometer_status]

    # Define the feature set (X)
    X = pd.DataFrame(
        {
            "tomatometer_status_encoded": [tomatometer_status_encoded],
            "tomatometer_rating": [tomatometer_rating],
            "tomatometer_count": [tomatometer_count],
        }
    )
    return X

# Sidebar inputs
st.sidebar.header("📋 Movie Details")
st.sidebar.selectbox("Tomatometer Status", ["Fresh", "Certified Fresh", "Rotten"], index=1, key="tomatometer_status")
st.sidebar.number_input("Tomatometer Count", min_value=1, value=550, key="tomatometer_count")

# Main UI for tomatometer rating selection
st.subheader("📊 Select Tomatometer Rating")
selected_tomatometer_rating = st.slider(
    "Choose a Tomatometer Rating:", min_value=10, max_value=100, value=85
)

# Predict button
if st.button("Predict Audience Rating 🎯"):
    # Prepare the features (X)
    X = prepare_features(selected_tomatometer_rating)

    # Predict using the Random Forest model
    y_pred = model.predict(X)[0]

    # Display prediction results
    st.success(f"**Predicted Audience Rating:** {y_pred:.2f}")
    st.write(f"**Selected Tomatometer Rating:** {selected_tomatometer_rating}")

    # Display the feature set used for prediction
    st.write("### Features Used for Prediction:")
    st.dataframe(X, use_container_width=True)
else:
    st.info("👈 Adjust the details in the sidebar, select a tomatometer rating, and click **Predict Audience Rating 🎯**.")
