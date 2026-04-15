import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# 1. Charger le dataset
df = pd.read_csv("weatherAUS.csv")

# 2. Prétraitement
df = df[['MinTemp','MaxTemp','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','RainToday','RainTomorrow']].dropna()

le = LabelEncoder()
df['RainToday'] = le.fit_transform(df['RainToday'])
df['RainTomorrow'] = le.fit_transform(df['RainTomorrow'])

X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entraînement
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Évaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. Sauvegarde
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Modèle sauvegardé dans model/model.pkl")