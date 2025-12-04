
from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import os
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = "change_this_secret"

# Try to load your model and top_features; if not present, create a dummy model for testing
MODEL_PATH = "model.pkl"
FEATURES_PATH = "top_features.pkl"

model = None
top_features = None

if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        top_features = joblib.load(FEATURES_PATH)
    except Exception as e:
        print("Failed to load model or features:", e)

# Fallback dummy model (for development/testing only)
class DummyModel:
    def predict(self, X):
        # predict 0 for all rows (no churn)
        return np.zeros((len(X),), dtype=int)
    def predict_proba(self, X):
        # low churn probability
        return np.vstack([1 - 0.1 * np.ones((len(X),)), 0.1 * np.ones((len(X),))]).T

if model is None or top_features is None:
    print("Using dummy model/features. Place model.pkl and top_features.pkl in project root to use real model.")
    model = DummyModel()
    top_features = [
        # minimal example features — replace with your real feature names once available
        'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'PaperlessBilling'
    ]



@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {}
        for f in top_features:
            raw = request.form.get(f)
            if raw is None:
                # missing field
                flash(f"Missing value for {f}")
                return redirect(url_for('index'))
            # try to convert to float; if fails keep original
            try:
                val = float(raw)
            except ValueError:
                val = raw
            data[f] = val

        df = pd.DataFrame([data], columns=top_features)

        # If there are categorical columns that need encoding, you should encode them here
        # This app expects the provided model to accept the same columns in top_features order.

        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1] if hasattr(model, 'predict_proba') else None

        label = "Customer May Churn" if int(pred) == 1 else "Customer Likely To Stay"
        prob_pct = round(float(proba) * 100, 2) if proba is not None else None

        return render_template('result.html', label=label, probability=prob_pct, data=data)
    except Exception as e:
        flash(f"Error during prediction: {e}")
        return redirect(url_for('index'))

@app.route('/batch', methods=['GET', 'POST'])
def batch():
    # upload CSV, expect columns matching top_features
    if request.method == 'POST':
        uploaded = request.files.get('csvfile')
        if not uploaded:
            flash('No file uploaded')
            return redirect(url_for('batch'))
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            flash(f'Failed to read CSV: {e}')
            return redirect(url_for('batch'))

        missing = [c for c in top_features if c not in df.columns]
        if missing:
            flash(f'Missing columns in CSV: {missing}')
            return redirect(url_for('batch'))

        X = df[top_features]
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else [None] * len(preds)

        out = df.copy()
        out['churn_pred'] = preds
        out['churn_prob'] = probs

        # Save predictions to a file and provide link
        out_path = 'batch_predictions.csv'
        out.to_csv(out_path, index=False)
        return redirect(url_for('download_batch'))

    return render_template('batch.html')

@app.route('/download')
def download_batch():
    from flask import send_file
    path = 'batch_predictions.csv'
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    flash('No batch file available')
    return redirect(url_for('batch'))

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    if request.method == 'POST':
        file = request.files.get("csvfile")
        if not file:
            flash("Please upload a CSV file.")
            return redirect(url_for('analysis'))

        import pandas as pd
        df = pd.read_csv(file)
        df.to_csv("uploaded_analysis.csv", index=False)

        return redirect(url_for("show_analysis"))

    return render_template("analysis.html")

@app.route('/show_analysis')
def show_analysis():
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import joblib
    import os
    import numpy as np

    if not os.path.exists("uploaded_analysis.csv"):
        flash("Please upload a CSV first.")
        return redirect(url_for("analysis"))

    df = pd.read_csv("uploaded_analysis.csv")

    # 1️⃣ Churn Count Plot
    plt.figure(figsize=(6,4))
    sns.countplot(x='Churn', data=df)
    plt.title("Churn Distribution")
    plt.savefig("static/churn_dist.png")
    plt.close()

    # 2️⃣ Correlation Heatmap
    plt.figure(figsize=(8,6))
    numeric_df = df.select_dtypes(include=['number'])
    sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("static/corr_heatmap.png")
    plt.close()

    # 3️⃣ Churn by Gender
    if 'gender' in df.columns:
        plt.figure(figsize=(6,4))
        sns.countplot(x='gender', hue='Churn', data=df)
        plt.title("Churn by Gender")
        plt.savefig("static/churn_gender.png")
        plt.close()

    # 4️⃣ Churn by Contract Type
    if 'Contract' in df.columns:
        plt.figure(figsize=(7,4))
        sns.countplot(x='Contract', hue='Churn', data=df)
        plt.title("Churn by Contract Type")
        plt.xticks(rotation=20)
        plt.savefig("static/churn_contract.png")
        plt.close()

    # 5️⃣ Distribution of MonthlyCharges (Churn vs Not)
    if 'MonthlyCharges' in df.columns:
        plt.figure(figsize=(7,4))
        sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True)
        plt.title("Monthly Charges Distribution")
        plt.savefig("static/monthly_dist.png")
        plt.close()

    # 6️⃣ Tenure Distribution
    if 'tenure' in df.columns:
        plt.figure(figsize=(7,4))
        sns.histplot(df['tenure'], kde=True, bins=30)
        plt.title("Tenure Distribution")
        plt.savefig("static/tenure_dist.png")
        plt.close()

    # 7️⃣ Churn vs Tenure Line Plot
    if 'tenure' in df.columns:
        tenure_churn = df.groupby('tenure')['Churn'].apply(lambda x: (x=='Yes').mean())
        plt.figure(figsize=(8,4))
        plt.plot(tenure_churn.index, tenure_churn.values)
        plt.title("Churn Probability by Tenure")
        plt.xlabel("Tenure (months)")
        plt.ylabel("Churn Probability")
        plt.savefig("static/tenure_churn.png")
        plt.close()

    # 8️⃣ Feature Importance graph from your saved model
    if os.path.exists("model.pkl"):
        model = joblib.load("model.pkl")
        try:
            # Extract feature importances if model supports it
            vclf = model.named_steps['vclf']
            gb = vclf.estimators_[0]  
            importances = gb.feature_importances_
            features = model.named_steps['scaler'].feature_names_in_

            imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
            imp_df = imp_df.sort_values(by='Importance', ascending=False)

            plt.figure(figsize=(8,5))
            sns.barplot(x='Importance', y='Feature', data=imp_df.head(15))
            plt.title("Top Features Contributing to Churn")
            plt.savefig("static/feature_importance.png")
            plt.close()

        except:
            pass

    return render_template("show_analysis.html")

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict_form')
def predict_form():
    return render_template("predict_form.html", features=top_features)



if __name__ == '__main__':
    app.run(debug=True)

