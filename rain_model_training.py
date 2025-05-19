import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


df = pd.read_csv("weather.csv")


df.drop(columns=['Date', 'Events'], inplace=True)


df['PrecipitationSumInches'] = df['PrecipitationSumInches'].replace(['T', 'M', '--'], 0.005)


for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')


df.dropna(subset=['PrecipitationSumInches'], inplace=True)


df.fillna(df.mean(), inplace=True)


if df.empty:
    raise ValueError("❌ All rows were dropped. Please check your dataset for invalid entries.")


df['RainToday'] = df['PrecipitationSumInches'].apply(lambda x: 1 if x > 0.05 else 0)


df.drop(columns=['PrecipitationSumInches'], inplace=True)


X = df.drop(columns=['RainToday'])
y = df['RainToday']


if X.empty or y.empty:
    raise ValueError("❌ Not enough valid data to train the model.")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


print(f"✅ Model trained successfully. Accuracy: {accuracy:.2f}")


joblib.dump(model, "rain_model.pkl")
print("📦 Model saved as 'rain_model.pkl'")
