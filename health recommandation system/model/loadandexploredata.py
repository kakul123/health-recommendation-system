# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score
)
import joblib
import logging
import warnings
warnings.filterwarnings("ignore")

# Step 2: Logging Setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# Step 3: Health Recommendation Generator
def generate_recommendation(row, disorder_label):
    recs = []

    # General disorder-specific suggestions
    if disorder_label == "Insomnia":
        recs += [
            "📅 Set a regular sleep schedule.",
            "📵 Avoid screens 1 hour before bed.",
            "☕ Reduce caffeine intake after 2 PM.",
            "🧘 Try mindfulness or breathing exercises before sleep."
        ]
    elif disorder_label == "Sleep Apnea":
        recs += [
            "🏥 Consult a sleep specialist for further evaluation.",
            "⚖️ Work towards maintaining a healthy weight.",
            "🚭 Avoid smoking and alcohol before sleep."
        ]
    elif disorder_label == "None":
        recs += [
            "✅ Your sleep patterns look healthy! Maintain your routine.",
            "💡 Keep monitoring your stress, activity, and sleep duration."
        ]

    # Personalized based on features
    if row['Sleep Duration'] < 6:
        recs.append("💤 Try to get at least 7-8 hours of sleep daily.")
    if row['Stress Level'] > 6:
        recs.append("😌 Consider stress management strategies like yoga or meditation.")
    if row['Physical Activity Level'] < 4:
        recs.append("🏃‍♂️ Aim for moderate exercise at least 3-5 times a week.")
    if row['BMI Category'] >= 2:
        recs.append("🥗 Adopt a balanced diet to improve your BMI.")
    if row['Heart Rate'] > 100:
        recs.append("❤️ Elevated heart rate detected — consider medical consultation.")

    return recs


# Step 4: Main pipeline
def main():
    try:
        # Load Dataset
        df = pd.read_csv(r"C:\Users\hp\Downloads\ss.csv")
        logging.info("📥 Dataset Loaded Successfully")
        print(df.head())

        # Split Blood Pressure if exists
        if 'Blood Pressure' in df.columns:
            df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True)
            df['Systolic_BP'] = pd.to_numeric(df['Systolic_BP'], errors='coerce')
            df['Diastolic_BP'] = pd.to_numeric(df['Diastolic_BP'], errors='coerce')
            df.drop(columns=['Blood Pressure'], inplace=True)

        # Drop irrelevant columns and handle nulls
        df.drop(columns=['Person ID'], errors='ignore', inplace=True)
        df.dropna(inplace=True)

        # Encode categorical features
        label_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
        label_encoders = {}
        for col in label_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        # Feature and target split
        X = df.drop('Sleep Disorder', axis=1)
        y = df['Sleep Disorder']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logging.info("🧠 Training model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logging.info("✅ Model training completed!")

        # Model Evaluation
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n📊 Accuracy: {acc:.2%}")
        print("📄 Classification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        # Feature Importance Plot
        importances = model.feature_importances_
        feature_names = X.columns
        sorted_indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[sorted_indices], y=feature_names[sorted_indices], palette='viridis')
        plt.title("Top Feature Importances")
        plt.tight_layout()
        plt.show()

        # Visualizations
        plt.hist(df['Age'], bins=10, color='skyblue', edgecolor='black')
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Count')
        plt.grid(True)
        plt.show()

        plt.scatter(df['Sleep Duration'], df['Stress Level'], c='purple', alpha=0.6)
        plt.title('Sleep Duration vs Stress Level')
        plt.xlabel('Sleep Duration (hrs)')
        plt.ylabel('Stress Level')
        plt.grid(True)
        plt.show()

        bmi_counts = df['BMI Category'].value_counts()
        plt.bar(bmi_counts.index, bmi_counts.values, color='orange')
        plt.title('BMI Category Distribution')
        plt.xlabel('BMI Category')
        plt.ylabel('Users')
        plt.show()

        sleep_disorders = df['Sleep Disorder'].value_counts()
        sleep_disorder_labels = label_encoders['Sleep Disorder'].inverse_transform(sleep_disorders.index)
        plt.pie(sleep_disorders.values, labels=sleep_disorder_labels, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999', '#99ff99'])
        plt.title('Sleep Disorder Distribution')
        plt.show()

        # Save the model
        joblib.dump(model, 'sleep_disorder_model.pkl')
        joblib.dump(label_encoders, 'label_encoders.pkl')
        with open('feature_names.txt', 'w') as f:
            f.write('\n'.join(feature_names))
        logging.info("💾 Model, encoders, and features saved!")

        # Try 1 prediction with recommendations
        sample = X_test.iloc[0]
        sample_index = sample.name
        pred_encoded = model.predict([sample])[0]
        pred_label = label_encoders['Sleep Disorder'].inverse_transform([pred_encoded])[0]

        print(f"\n🔍 Prediction Result: {pred_label}")
        print("🩺 Personalized Recommendations:")
        recs = generate_recommendation(df.iloc[sample_index], pred_label)
        for rec in recs:
            print(" -", rec)

    except Exception as e:
        logging.error(f"❌ Error occurred: {str(e)}")


# Entry point
if __name__ == "__main__":
    main()
