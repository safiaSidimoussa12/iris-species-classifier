import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- CSS FIX: Hides the broken mobile elements ---
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, 
    unsafe_html=True
)

st.title("Iris Species Classifier")
st.write("An interactive machine learning application that utilizes a Random Forest model to classify flower species based on physical measurements.")

# --- 2. THE DATASET & TRAINING ---
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)

st.sidebar.header("Model Metrics")
st.sidebar.metric(label="Model Accuracy", value=f"{score * 100:.2f}%")

# --- 3. USER INTERFACE FOR PREDICTION ---
st.subheader("Model Inference")
st.write("Adjust the input features below to evaluate the classification model:")

sepal_length = st.slider("Sepal Length (cm)", 4.3, 7.9, 5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.4, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 6.9, 4.3)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

user_input = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=iris.feature_names)

prediction_numeric = model.predict(user_input)[0]
prediction_flower_name = iris.target_names[prediction_numeric]

st.success(f"Predicted Class: **Iris-{prediction_flower_name.capitalize()}**")
