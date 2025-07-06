import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import label_binarize

# ------------------------ Load CSV from GitHub ------------------------ #
GITHUB_CSV_URL = "https://raw.githubusercontent.com/RANVEERLAL/Aviation/refs/heads/main/synthetic_airline_survey_data.csv"
df = pd.read_csv(GITHUB_CSV_URL)

# ------------------------ Streamlit Layout ------------------------ #
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
    label_encoders = {}
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            label_encoders[col] = le
    X = df_clean.drop(columns=[target])
    y = df_clean[target]
    return X, y, label_encoders

def preprocess_numerical(df):
    return df.select_dtypes(include=["int64", "float64"]).dropna()

def prepare_transactions(df, cols):
    tx = df[cols].dropna().apply(lambda x: [i.strip() for i in ','.join(x).split(',')], axis=1).tolist()
    te = TransactionEncoder()
    te_ary = te.fit(tx).transform(tx)
    return pd.DataFrame(te_ary, columns=te.columns_)

# ------------------------ Tab 1: Visualization ------------------------ #
with tab1:
    st.header("📊 Descriptive Analytics: 15+ Insights")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Statistical Overview")
    st.dataframe(df.describe(include='all'))

    visuals = [
        ("Age", "Age Distribution"),
        ("MonthlyIncome", "Monthly Income Distribution"),
        ("FlightsPerYear", "Flights Per Year Count"),
        ("TravelPurpose", "Travel Purpose Count"),
        ("Class", "Booking Class Count"),
        ("AirlinePrefReason", "Airline Preference Reasons"),
        ("AvgSpendPerFlight", "Average Spend Distribution"),
        ("PersonalizationImportance", "Personalization Importance"),
        ("BookingDevice", "Booking Device Count"),
        ("TechComfortLevel", "Tech Comfort Level Distribution"),
        ("WillingToUsePersonalizationApp", "Willingness to Use App"),
        ("WillingToPayExtra", "Willingness to Pay Extra"),
        ("TrustsAirlinesWithData", "Trust with Personal Data"),
        ("WillRecommendToOthers", "Referral Willingness"),
        ("PercentWillingToSpendMore", "Percent Willing to Spend More")
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

    target_col = st.selectbox("Select Target Variable", df.columns)
    X, y, encoders = preprocess_classification(df.copy(), target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "GBRT": GradientBoostingClassifier()
    }
    # … (after training each model)
    results = {}
    rocs    = {}
    importances = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = classification_report(y_test, y_pred, output_dict=True)

        # Only compute ROC if it’s a binary problem and model supports predict_proba
        if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
            try:
                proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, proba)
                rocs[name] = (fpr, tpr)
            except Exception as e:
                st.warning(f"Could not compute ROC for {name}: {e}")
        else:
            st.info(f"Skipping ROC for {name}: target not binary or no predict_proba")

        # Feature importances (where available)
        if hasattr(model, "feature_importances_"):
            importances[name] = model.feature_importances_
        elif name == "KNN":
            importances[name] = [0] * X.shape[1]

# ------------------------ Tab 3: Clustering ------------------------ #
with tab3:
    st.header("📍 KMeans Clustering")

    num_df = preprocess_numerical(df)
    scores = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(num_df)
        score = silhouette_score(num_df, km.labels_)
        scores.append(score)

    fig, ax = plt.subplots()
    ax.plot(range(2, 11), scores, marker='o')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Analysis")
    st.pyplot(fig)

    k_val = st.slider("Select number of clusters", 2, 10, 3)
    km_final = KMeans(n_clusters=k_val)
    labels = km_final.fit_predict(num_df)
    df_clustered = df.copy()
    df_clustered["Cluster"] = labels
    st.dataframe(df_clustered.head())
    st.download_button("Download Clustered Data", df_clustered.to_csv(index=False), file_name="clustered_data.csv")

    # ---------------------- Helper for Association ---------------------- #
def prepare_transactions(df, cols):
    # Coerce values to strings, split on commas, strip whitespace
    records = []
    for _, row in df[cols].dropna().iterrows():
        items = []
        for col in cols:
            # split on comma, then strip each piece
            parts = str(row[col]).split(',')
            items.extend([p.strip() for p in parts if p.strip()])
        records.append(items)
    te = TransactionEncoder()
    te_ary = te.fit(records).transform(records)
    return pd.DataFrame(te_ary, columns=te.columns_)

# ---------------------- Tab 4: Association Rules ---------------------- #
with tab4:
    st.header("🔗 Association Rule Mining (Apriori)")

    # Only allow columns that are truly comma-separated lists
    list_cols = [
        c for c in df.columns 
        if df[c].dtype == "object" and df[c].str.contains(',').any()
    ]
    if len(list_cols) < 2:
        st.warning("No appropriate columns found for association mining (need at least two comma-list columns).")
    else:
        col1 = st.selectbox("Column 1 (basket)", list_cols, key="assoc1")
        col2 = st.selectbox("Column 2 (basket)", [c for c in list_cols if c != col1], key="assoc2")

        min_support = st.slider("Min Support", 0.01, 1.0, 0.1)
        min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.5)

        trans_df = prepare_transactions(df, [col1, col2])
        freq_items = apriori(trans_df, min_support=min_support, use_colnames=True)
        rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
        rules = rules.sort_values("confidence", ascending=False).head(10)

        if rules.empty:
            st.info("No rules found with these parameters.")
        else:
            st.dataframe(rules)

# ------------------------ Tab 5: Regression ------------------------ #
with tab5:
    st.header("📈 Regression Models")
    num_df = preprocess_numerical(df)
    target = st.selectbox("Select Numeric Target", num_df.columns)
    X = num_df.drop(columns=[target])
    y = num_df[target]

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor()
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    reg_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        reg_results[name] = {
            "MSE": mean_squared_error(y_test, pred),
            "R2 Score": r2_score(y_test, pred)
        }

    st.dataframe(pd.DataFrame(reg_results).T)
