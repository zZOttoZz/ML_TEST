import streamlit as st
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🔍 Enkel ML + SHAP-insikter från Excel-data")

# 1. Ladda upp fil
uploaded_file = st.file_uploader("📤 Ladda upp en Excel-fil", type=["xlsx", "xls"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.subheader("🔎 Förhandsgranskning av data")
    st.dataframe(df.head())

    # 2. Välj målvariabel
    target_col = st.selectbox("🎯 Välj målvariabel (den kolumn du vill förutsäga)", df.columns)

    if target_col:
        # 3. Förbered data
        df_model = df.copy()
        df_model = df_model.dropna()  # Enkel hantering av null

        X = df_model.drop(columns=[target_col])
        y = df_model[target_col]

        # Label Encoding om target är text
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        # Label Encoding för kategoriska features
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col])

        # 4. Träna modell
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        # 5. Utvärdera
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_prob)
        st.success(f"✅ Modell tränad! AUC = {auc_score:.2f}")

        # 6. SHAP-analys
        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        st.subheader("📊 Viktigaste drivare (SHAP Feature Importance)")
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values, max_display=10, show=False)
        st.pyplot(fig)

        with st.expander("📈 Visa detaljerad SHAP-summary plot"):
            fig2, ax2 = plt.subplots()
            shap.summary_plot(shap_values, X, plot_type="dot", show=False)
            st.pyplot(fig2)
