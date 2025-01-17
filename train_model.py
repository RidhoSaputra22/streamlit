import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
file_path = "data_rumahh.xlsx"
data = pd.read_excel(file_path)

# Pastikan kolom yang diperlukan ada dalam dataset
if not all(col in data.columns for col in ["luas_tanah", "jumlah_kamar", "lokasi", "harga_rumah"]):
    raise ValueError("Dataset harus memiliki kolom: 'luas_tanah', 'jumlah_kamar', 'lokasi', dan 'harga_rumah'.")

# Encode lokasi menjadi numerik
data["lokasi_encoded"] = data["lokasi"].astype("category").cat.codes

# Pilih fitur dan target
X = data[["luas_tanah", "jumlah_kamar", "lokasi_encoded"]]
y = data["harga_rumah"]

# Bagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat model Random Forest
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Evaluasi model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error: {mae}")
print(f"R2 Score: {r2}")

# Simpan model ke file
model_path = "model.pkl"
joblib.dump(model, model_path)
print(f"Model disimpan ke {model_path}")

# Visualisasi data
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data["luas_tanah"], y=data["harga_rumah"], hue=data["lokasi"])
plt.title("Hubungan Luas Tanah dan Harga Rumah Berdasarkan Lokasi")
plt.xlabel("Luas Tanah (m2)")
plt.ylabel("Harga Rumah (Rp)")
plt.legend(title="Lokasi")
plt.savefig("scatterplot_harga_rumah.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x=data["lokasi"].value_counts().index, y=data["lokasi"].value_counts().values)
plt.title("Distribusi Jumlah Rumah Berdasarkan Lokasi")
plt.xlabel("Lokasi")
plt.ylabel("Jumlah Rumah")
plt.savefig("barplot_lokasi.png")
plt.show()
