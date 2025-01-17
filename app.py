import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Title
st.title("Dashboard Prediksi Harga Rumah")

# Load Excel data
@st.cache
def load_data(file_path):
    try:
        data = pd.read_excel(file_path)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load Machine Learning Model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Prediction Function
def make_prediction(model, input_data):
    try:
        prediction = model.predict([input_data])
        return prediction[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# File paths
excel_path = "data_rumahh.xlsx"  # Path ke file Excel
model_path = "model.pkl"  # Path ke model machine learning

# Load data dan model
data = load_data(excel_path)
model = load_model(model_path)

# Tab layout for UI
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Rumah", "🔮 Prediksi", "📈 Visualisasi", "📂 Model Info"])

# Tab 1: Data Rumah
with tab1:
    st.header("Data Rumah")
    if data is not None:
        st.dataframe(data)
    else:
        st.error("Data tidak tersedia.")

# Tab 2: Prediksi
with tab2:
    st.header("Prediksi Harga Rumah")

    if model is not None and data is not None:
        # Input fields
        luas_tanah = st.number_input("Luas Tanah (m2)", min_value=0.0, value=100.0)
        jumlah_kamar = st.number_input("Jumlah Kamar", min_value=1, value=2)
        lokasi = st.selectbox("Lokasi Rumah", data["lokasi"].unique())

        # Encode lokasi
        lokasi_encoded = data["lokasi"].astype("category").cat.categories.tolist().index(lokasi)

        # Button for prediction
        if st.button("Prediksi Harga"):
            input_data = [luas_tanah, jumlah_kamar, lokasi_encoded]
            result = make_prediction(model, input_data)
            if result is not None:
                st.success(f"Harga Rumah yang Diprediksi: Rp {result:,}")
    else:
        st.error("Model atau data tidak tersedia untuk prediksi.")

# Tab 3: Visualisasi
with tab3:
    st.header("Visualisasi Data")
    if data is not None:
        # Scatter Plot
        st.subheader("Hubungan Luas Tanah dan Harga Rumah Berdasarkan Lokasi")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=data["luas_tanah"], y=data["harga_rumah"], hue=data["lokasi"], ax=ax1)
        ax1.set_title("Hubungan Luas Tanah dan Harga Rumah")
        ax1.set_xlabel("Luas Tanah (m2)")
        ax1.set_ylabel("Harga Rumah (Rp)")
        st.pyplot(fig1)

        # Bar Plot
        st.subheader("Distribusi Jumlah Rumah Berdasarkan Lokasi")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(x=data["lokasi"].value_counts().index, y=data["lokasi"].value_counts().values, ax=ax2)
        ax2.set_title("Distribusi Rumah Berdasarkan Lokasi")
        ax2.set_xlabel("Lokasi")
        ax2.set_ylabel("Jumlah Rumah")
        st.pyplot(fig2)

        # Distribution of Harga Rumah
        st.subheader("Distribusi Harga Rumah")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.histplot(data["harga_rumah"], bins=20, kde=True, ax=ax3)
        ax3.set_title("Distribusi Harga Rumah")
        ax3.set_xlabel("Harga Rumah (Rp)")
        ax3.set_ylabel("Frekuensi")
        st.pyplot(fig3)
    else:
        st.error("Data tidak tersedia untuk visualisasi.")

# Tab 4: Model Info
with tab4:
 
    st.header("Informasi Model")
    if model is not None:
        # Hyperparameter Model
        st.subheader("Hyperparameter Model")
        st.json(model.get_params())  # Menampilkan parameter model dalam format JSON


        st.subheader("Evaluasi Model")
        if data is not None:
            # Encode lokasi untuk data
            data["lokasi_encoded"] = data["lokasi"].astype("category").cat.codes

            # Pilih fitur dan target
            X = data[["luas_tanah", "jumlah_kamar", "lokasi_encoded"]]
            y = data["harga_rumah"]

            # Bagi data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Prediksi dan evaluasi
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            st.write(f"Mean Absolute Error (MAE): {mae}")
            st.write(f"R2 Score: {r2}")

            # Scatter plot actual vs predicted
            st.subheader("Perbandingan Harga Rumah Aktual vs Prediksi")
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=y_test, y=predictions, ax=ax4)
            ax4.set_title("Harga Rumah Aktual vs Prediksi")
            ax4.set_xlabel("Harga Rumah Aktual")
            ax4.set_ylabel("Harga Rumah Prediksi")
            st.pyplot(fig4)
    else:
        st.error("Model tidak tersedia.")
