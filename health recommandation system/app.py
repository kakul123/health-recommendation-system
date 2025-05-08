from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

app = Flask("befit", static_folder='static')
app.secret_key = 'kakul_this_side'

# Load model and encoders
model = joblib.load('model/sleep_disorder_model.pkl')
label_encoders = joblib.load('model/label_encoders.pkl')
with open('model/feature_names.txt', 'r') as f:
    feature_names = f.read().splitlines()

# -----------------------------
# Login Page
# -----------------------------
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username.strip() and password.strip():
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Please enter both username and password.')

    return render_template('login.html')


# -----------------------------
# Dashboard - User Input Form
# -----------------------------
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        form_data = request.form
        input_data = []

        for feature in feature_names:
            value = form_data.get(feature)
            if feature in label_encoders:
                value = label_encoders[feature].transform([value])[0]
            else:
                value = float(value)
            input_data.append(value)

        prediction = model.predict([input_data])[0]
        disorder_label = label_encoders['Sleep Disorder'].inverse_transform([prediction])[0]

        row_dict = dict(zip(feature_names, input_data))
        row_df = pd.DataFrame([row_dict])

        recs = generate_recommendation(row_df.iloc[0], disorder_label)

        # Generate charts
        pie_chart_html = generate_pie_chart(row_df.iloc[0])
        bar_chart_html = generate_bar_chart(row_df.iloc[0])

        return render_template(
            'result.html',
            prediction=disorder_label,
            recommendations=recs,
            pie_chart=pie_chart_html,
            bar_chart=bar_chart_html
        )

    return render_template('dashboard.html', feature_names=feature_names, label_encoders=label_encoders)


# -----------------------------
# Logout
# -----------------------------
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


# -----------------------------
# Recommendation Logic
# -----------------------------
def generate_recommendation(row, disorder_label):
    recs = []

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


# -----------------------------
# Chart Generators
# -----------------------------
def generate_pie_chart(row):
    labels = ['Sleep Duration', 'Stress Level', 'Physical Activity Level']
    values = [row['Sleep Duration'], row['Stress Level'], row['Physical Activity Level']]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(title_text="Your Lifestyle Distribution")
    return fig.to_html(full_html=False)

def generate_bar_chart(row):
    categories = ['Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP']
    values = [row.get('Heart Rate', 0), row.get('Daily Steps', 0),
              row.get('Systolic_BP', 0), row.get('Diastolic_BP', 0)]
    fig = go.Figure([go.Bar(x=categories, y=values)])
    fig.update_layout(title_text="Health Metrics", yaxis_title="Values")
    return fig.to_html(full_html=False)


# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
