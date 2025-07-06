import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from urllib.error import URLError

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, mean_squared_error, r2_score,
    silhouette_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ------------------------ Load CSV ------------------------ #
GITHUB_CSV_URL = (
    "https://raw.githubusercontent.com/YOUR_USERNAME/"
    "YOUR_REPO_NAME/main/data/synthetic_airline_survey_data.csv"
)
LOCAL_CSV_PATH = "data/synthetic_airline_survey_data.csv"

try:
    df = pd.read_csv(GITHUB_CSV_URL)
    data_source = "GitHub"
except URLError:
    df = pd.read_csv(LOCAL_CSV_PATH)
    data_source = "local file"
except Exception as e:
    st.error(f"Could not load data:\n{e}")
    st.stop()

st.sidebar.success(f"Loaded data from {data_source}")

# ------------------------ Helpers ------------------------ #
def preprocess_classification(df_in, target_col):
    dfc = df_in.dropna().copy()
    for c in dfc.select_dtypes(include="object").columns:
        dfc[c] = LabelEncoder().fit_transform(dfc[c].astype(str))
    X = dfc.drop(columns=[target_col])
    y = dfc[target_col]
    return X, y

def preprocess_numerical(df_in):
    return df_in.select_dtypes(include=["int64", "float64"]).dropna()

def prepare_transactions(df_in, cols):
    records = []
    for _, row in df_in[cols].dropna().iterrows():
        items = []
        for c in cols:
            parts = str(row[c]).split(",")
            items += [p.strip() for p in parts if p.strip()]
        records.append(items)
    te = TransactionEncoder()
    te_ary = te.fit(records).transform(records)
    return pd.DataFrame(te_ary, columns=te.columns_)

# ------------------------ Streamlit Layout ------------------------ #
st.set_page_config(layout="wide", page_title="Airline CX Dashboard")
st.title("✈️ Airline Customer Experience Analytics Dashboard")

tabs = st.tabs([
    "📊 Data Visualization",
    "🔎 Classification",
    "📍 Clustering",
    "🔗 Association Rules",
    "📈 Regression"
])

# ------------------------ Tab 1: Visualization ------------------------ #
with tabs[0]:
    st.header("📊 Descriptive Analytics: 15 Insights")
    st.subheader("Preview & Stats")
    st.dataframe(df.head())
    st.dataframe(df.describe(include="all"))

    visuals = [
        ("Age","Age Distribution"),
        ("MonthlyIncome","Income Distribution"),
        ("FlightsPerYear","Flights Per Year"),
        ("TravelPurpose","Travel Purpose Counts"),
        ("Class","Booking Class Counts"),
        ("AirlinePrefReason","Airline Preference Reasons"),
        ("AvgSpendPerFlight","Avg Spend Distribution"),
        ("PersonalizationImportance","Personalization Importance"),
        ("BookingDevice","Booking Device Counts"),
        ("TechComfortLevel","Tech Comfort Level"),
        ("WillingToUsePersonalizationApp","Willingness to Use App"),
        ("WillingToPayExtra","Willingness to Pay Extra"),
        ("TrustsAirlinesWithData","Trust in Data Handling"),
        ("WillRecommendToOthers","Referral Likelihood"),
        ("PercentWillingToSpendMore","% Willing to Spend More")
    ]

    for col, title in visuals:
        st.markdown(f"#### {title}")
        if df[col].dtype == "object":
            fig, ax = plt.subplots()
            sns.countplot(y=df[col], ax=ax)
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

# ------------------------ Tab 2: Classification ------------------------ #
with tabs[1]:
    st.header("🔎 Classification Models")

    binary_cols = [c for c in df.columns if df[c].nunique() == 2]
    if not binary_cols:
        st.error("No binary columns available for classification.")
    else:
        target = st.selectbox("Select Binary Target", binary_cols)
        X, y = preprocess_classification(df.copy(), target)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        models = {
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "GBRT": GradientBoostingClassifier()
        }

        results, rocs, importances = {}, {}, {}
        for name, mdl in models.items():
            mdl.fit(X_train, y_train)
            y_pred = mdl.predict(X_test)
            results[name] = classification_report(
                y_test, y_pred, output_dict=True
            )
            if hasattr(mdl, "predict_proba"):
                proba = mdl.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, proba)
                rocs[name] = (fpr, tpr)
            if hasattr(mdl, "feature_importances_"):
                importances[name] = mdl.feature_importances_
            else:
                importances[name] = np.zeros(X.shape[1])

        st.subheader("Model Performance")
        for name, rpt in results.items():
            st.markdown(f"**{name}**")
            st.dataframe(pd.DataFrame(rpt).T)

        sel = st.selectbox("Choose Model for Confusion & ROC", list(models.keys()))
        # Confusion
        cm = pd.crosstab(
            y_test, models[sel].predict(X_test),
            rownames=["Actual"], colnames=["Predicted"]
        )
        st.subheader("Confusion Matrix")
        st.dataframe(cm)

        # ROC Curve
        st.subheader("ROC Curve")
        fig, ax = plt.subplots()
        for name, (fpr, tpr) in rocs.items():
            ax.plot(fpr, tpr, label=name)
        ax.legend(); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        st.pyplot(fig)

        # Feature Importance
        st.subheader("Feature Importances")
        imp = importances[sel]
        fig, ax = plt.subplots()
        sns.barplot(x=imp, y=X.columns, ax=ax)
        ax.set_title(f"{sel} Importances")
        st.pyplot(fig)

# ------------------------ Tab 3: Clustering ------------------------ #
with tabs[2]:
    st.header("📍 KMeans Clustering")

    num_df = preprocess_numerical(df)
    ks = list(range(2, 11))
    inertias = []
    silhouettes = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42).fit(num_df)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(num_df, km.labels_))

    st.subheader("Elbow Plot")
    fig, ax = plt.subplots()
    ax.plot(ks, inertias, marker="o")
    ax.set_xlabel("k"); ax.set_ylabel("Inertia")
    st.pyplot(fig)

    st.subheader("Silhouette Scores")
    fig, ax = plt.subplots()
    ax.plot(ks, silhouettes, marker="o")
    ax.set_xlabel("k"); ax.set_ylabel("Silhouette Score")
    st.pyplot(fig)

    k_sel = st.slider("Select k", 2, 10, 3)
    km_final = KMeans(n_clusters=k_sel, random_state=42).fit(num_df)
    dfc = df.copy()
    dfc["Cluster"] = km_final.labels_
    st.dataframe(dfc.head())
    st.download_button(
        "Download Clustered Data",
        dfc.to_csv(index=False),
        file_name="clustered_data.csv"
    )

# ------------------------ Tab 4: Association Rules ------------------------ #
with tabs[3]:
    st.header("🔗 Association Rule Mining")

    basket_cols = [
        c for c in df.columns 
        if df[c].dtype == "object" and df[c].str.contains(",").any()
    ]
    if len(basket_cols) < 2:
        st.warning("Need at least 2 comma-list columns.")
    else:
        c1 = st.selectbox("Column 1", basket_cols, key="a1")
        c2 = st.selectbox("Column 2", [c for c in basket_cols if c!=c1], key="a2")
        sup = st.slider("Min Support", 0.01, 1.0, 0.1)
        conf = st.slider("Min Confidence", 0.1, 1.0, 0.5)

        tdf = prepare_transactions(df, [c1, c2])
        freq = apriori(tdf, min_support=sup, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=conf)
        st.dataframe(rules.sort_values("confidence", ascending=False).head(10))

# ------------------------ Tab 5: Regression ------------------------ #
with tabs[4]:
    st.header("📈 Regression Models")

    num_df = preprocess_numerical(df)
    target_reg = st.selectbox("Select Numeric Target", num_df.columns)
    Xr = num_df.drop(columns=[target_reg])
    yr = num_df[target_reg]
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)

    regs = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor()
    }

    metrics = {}
    coefs = {}
    for name, model in regs.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        metrics[name] = {
            "MSE": mean_squared_error(yte, pred),
            "R2": r2_score(yte, pred)
        }
        if hasattr(model, "coef_"):
            coefs[name] = np.abs(model.coef_)
        elif hasattr(model, "feature_importances_"):
            coefs[name] = model.feature_importances_

    st.subheader("Regression Metrics")
    st.dataframe(pd.DataFrame(metrics).T)

    st.subheader("Feature Importances / Coefficients")
    sel_reg = st.selectbox("Model for Importances", list(regs.keys()))
    vals = coefs[sel_reg]
    fig, ax = plt.subplots()
    sns.barplot(x=vals, y=Xr.columns, ax=ax)
    ax.set_title(f"{sel_reg} Feature Importance")
    st.pyplot(fig)
