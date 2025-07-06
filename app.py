import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

# ------------------------ Load CSV from GitHub ------------------------ #
GITHUB_CSV_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO_NAME/main/synthetic_airline_survey_data.csv"
df = pd.read_csv(GITHUB_CSV_URL)

st.set_page_config(layout="wide")
st.title("✈️ Airline Customer Experience Analytics Dashboard")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Visualization",
    "🔎 Classification",
    "📍 Clustering",
    "🔗 Association Rules",
    "📈 Regression"
])

# ------------------------ Helpers ------------------------ #
def preprocess_classification(df, target):
    df_clean = df.dropna().copy()
    for col in df_clean.select_dtypes(include="object").columns:
        df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))
    X = df_clean.drop(columns=[target])
    y = df_clean[target]
    return X, y

def preprocess_numerical(df):
    return df.select_dtypes(include=["int64", "float64"]).dropna()

def prepare_transactions(df, cols):
    records = []
    for _, row in df[cols].dropna().iterrows():
        items = []
        for c in cols:
            parts = str(row[c]).split(",")
            items += [p.strip() for p in parts if p.strip()]
        records.append(items)
    te = TransactionEncoder()
    te_ary = te.fit(records).transform(records)
    return pd.DataFrame(te_ary, columns=te.columns_)

# ------------------------ Tab 1: Visualization ------------------------ #
with tab1:
    st.header("📊 Descriptive Analytics: 15+ Insights")
    st.dataframe(df.head())
    st.dataframe(df.describe(include="all"))
    visuals = [
        ("Age","Age Distribution"),
        ("MonthlyIncome","Monthly Income Distribution"),
        ("FlightsPerYear","Flights Per Year Count"),
        ("TravelPurpose","Travel Purpose Count"),
        ("Class","Booking Class Count"),
        ("AirlinePrefReason","Airline Preference Reasons"),
        ("AvgSpendPerFlight","Average Spend Distribution"),
        ("PersonalizationImportance","Personalization Importance"),
        ("BookingDevice","Booking Device Count"),
        ("TechComfortLevel","Tech Comfort Level Distribution"),
        ("WillingToUsePersonalizationApp","Willingness to Use App"),
        ("WillingToPayExtra","Willingness to Pay Extra"),
        ("TrustsAirlinesWithData","Trust with Personal Data"),
        ("WillRecommendToOthers","Referral Willingness"),
        ("PercentWillingToSpendMore","Percent Willing to Spend More")
    ]
    for col, title in visuals:
        st.markdown(f"#### {title}")
        fig, ax = plt.subplots()
        if df[col].dtype == "object":
            sns.countplot(y=df[col], ax=ax, order=df[col].value_counts().index)
        else:
            sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

# ------------------------ Tab 2: Classification ------------------------ #
with tab2:
    st.header("🔎 Classification Models")
    binary_cols = [c for c in df.columns if df[c].nunique()==2]
    if not binary_cols:
        st.error("No binary columns available for classification.")
        st.stop()
    target = st.selectbox("Select Binary Target", binary_cols)
    X, y = preprocess_classification(df.copy(), target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "GBRT": GradientBoostingClassifier()
    }

    results = {}
    rocs = {}
    importances = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        results[name] = classification_report(y_test, y_pred, output_dict=True)
        # ROC
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            rocs[name] = (fpr, tpr)
        # Feature importance
        if hasattr(m, "feature_importances_"):
            importances[name] = m.feature_importances_
        else:
            importances[name] = [0]*X.shape[1]

    st.subheader("Model Performance")
    for nm, rpt in results.items():
        st.markdown(f"**{nm}**")
        st.dataframe(pd.DataFrame(rpt).T)

    st.subheader("Confusion Matrix")
    sel = st.selectbox("Choose Model", list(models.keys()))
    cm = pd.crosstab(y_test, models[sel].predict(X_test),
                     rownames=["Actual"], colnames=["Predicted"])
    st.dataframe(cm)

    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    for nm, (fpr, tpr) in rocs.items():
        ax.plot(fpr, tpr, label=nm)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    st.pyplot(fig)

    st.subheader("Feature Importances")
    imp = importances[sel]
    fig, ax = plt.subplots()
    sns.barplot(x=imp, y=X.columns, ax=ax)
    ax.set_title(f"{sel} Importances")
    st.pyplot(fig)

# ------------------------ Tab 3: Clustering ------------------------ #
with tab3:
    st.header("📍 KMeans Clustering")

    num_df = preprocess_numerical(df)
    inertias = []
    silhouettes = []
    ks = list(range(2,11))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42).fit(num_df)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(num_df, km.labels_))

    st.subheader("Elbow Method (Inertia vs k)")
    fig, ax = plt.subplots()
    ax.plot(ks, inertias, marker='o')
    ax.set_xlabel("k"); ax.set_ylabel("Inertia")
    ax.set_title("Elbow Plot")
    st.pyplot(fig)

    st.subheader("Silhouette Score vs k")
    fig, ax = plt.subplots()
    ax.plot(ks, silhouettes, marker='o')
    ax.set_xlabel("k"); ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Analysis")
    st.pyplot(fig)

    k = st.slider("Select Number of Clusters", 2, 10, 3)
    km = KMeans(n_clusters=k, random_state=42).fit(num_df)
    df_c = df.copy()
    df_c["Cluster"] = km.labels_
    st.dataframe(df_c.head())
    st.download_button("Download Clustered Data",
                       df_c.to_csv(index=False),
                       file_name="clustered_data.csv")

# ------------------------ Tab 4: Association Rules ------------------------ #
with tab4:
    st.header("🔗 Association Rule Mining")
    # pick only comma-list columns
    basket_cols = [
        c for c in df.columns 
        if df[c].dtype=="object" and df[c].str.contains(',').any()
    ]
    if len(basket_cols) < 2:
        st.warning("Need at least 2 comma-list columns.")
    else:
        c1 = st.selectbox("Column 1", basket_cols, key="c1")
        c2 = st.selectbox("Column 2", [c for c in basket_cols if c!=c1], key="c2")
        sup = st.slider("Min Support", 0.01, 1.0, 0.1)
        conf = st.slider("Min Confidence", 0.1, 1.0, 0.5)
        tdf = prepare_transactions(df, [c1,c2])
        freq = apriori(tdf, min_support=sup, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=conf)
        st.dataframe(rules.sort_values("confidence", ascending=False).head(10))

# ------------------------ Tab 5: Regression ------------------------ #
with tab5:
    st.header("📈 Regression Models")

    num_df = preprocess_numerical(df)
    target = st.selectbox("Select Numeric Target", num_df.columns)
    X = num_df.drop(columns=[target])
    y = num_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    regs = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor()
    }

    reg_results = {}
    coefs = {}
    for name, mdl in regs.items():
        mdl.fit(X_train, y_train)
        pred = mdl.predict(X_test)
        reg_results[name] = {
            "MSE": mean_squared_error(y_test, pred),
            "R2": r2_score(y_test, pred)
        }
        # capture feature importance / coefficients
        if hasattr(mdl, "coef_"):
            coefs[name] = np.abs(mdl.coef_)
        elif hasattr(mdl, "feature_importances_"):
            coefs[name] = mdl.feature_importances_

    st.subheader("Regression Metrics")
    st.dataframe(pd.DataFrame(reg_results).T)

    st.subheader("Feature Importances / Coefficients")
    sel_reg = st.selectbox("Select Model for Importances", list(regs.keys()))
    imp_vals = coefs[sel_reg]
    fig, ax = plt.subplots()
    sns.barplot(x=imp_vals, y=X.columns, ax=ax)
    ax.set_title(f"{sel_reg} Feature Importance")
    st.pyplot(fig)
