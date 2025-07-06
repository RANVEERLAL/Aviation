import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Setup
st.set_page_config(layout="wide")
st.title("✈️ Airline Customer Experience Analytics Dashboard")

# Data Upload
uploaded = st.file_uploader("Upload your survey CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("Dataset successfully loaded!")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Visualization", 
        "🔎 Classification", 
        "📍 Clustering", 
        "🔗 Association Rule Mining", 
        "📈 Regression"
    ])

    # -----------------------------
    with tab1:
        st.header("📊 Data Visualization & Descriptive Insights")
        st.subheader("Basic Info")
        st.write(df.describe(include='all'))
        st.write("Dataset Shape:", df.shape)

        st.subheader("15+ Visual Insights")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Age Distribution")
            fig = plt.figure()
            sns.histplot(df["Age"], kde=True)
            st.pyplot(fig)

            st.write("Flights Per Year Count")
            st.bar_chart(df["FlightsPerYear"].value_counts())

            st.write("Employment Status")
            st.bar_chart(df["EmploymentStatus"].value_counts())

            st.write("Class Booked")
            st.bar_chart(df["Class"].value_counts())

            st.write("Valued Services")
            services = pd.Series(','.join(df["ValuedServices"].dropna()).split(','))
            st.bar_chart(services.value_counts().head(10))

            st.write("Customizable Services Preferences")
            custom = pd.Series(','.join(df["CustomizableServices"].dropna()).split(','))
            st.bar_chart(custom.value_counts().head(10))

            st.write("App Usage Frequency")
            st.bar_chart(df["AppUsage"].value_counts())

        with col2:
            st.write("Monthly Income Distribution")
            fig = plt.figure()
            sns.boxplot(x=df["MonthlyIncome"])
            st.pyplot(fig)

            st.write("Average Spend per Flight")
            fig = plt.figure()
            sns.histplot(df["AvgSpendPerFlight"], kde=True)
            st.pyplot(fig)

            st.write("Travel Purpose")
            st.bar_chart(df["TravelPurpose"].value_counts())

            st.write("Willingness to Use Personalization App")
            st.bar_chart(df["WillingToUsePersonalizationApp"].value_counts())

            st.write("Trust Airlines with Data")
            st.bar_chart(df["TrustsAirlinesWithData"].value_counts())

            st.write("Recommendation Intent")
            st.bar_chart(df["WillRecommendToOthers"].value_counts())

            st.write("Device Used for Booking")
            st.bar_chart(df["BookingDevice"].value_counts())

    # -----------------------------
    with tab2:
        st.header("🔎 Classification")
        st.subheader("Choose Target Variable")
        class_df = df.dropna()
        le = LabelEncoder()
        encoded_df = class_df.apply(lambda x: le.fit_transform(x.astype(str)) if x.dtypes == 'object' else x)

        target = st.selectbox("Target Column", encoded_df.columns)
        X = encoded_df.drop(columns=[target])
        y = encoded_df[target]
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

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = classification_report(y_test, y_pred, output_dict=True)
            if hasattr(model, 'predict_proba'):
                fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
                rocs[name] = (fpr, tpr)
            importances[name] = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.zeros(X.shape[1])

        st.subheader("Performance Metrics")
        for name, report in results.items():
            st.write(f"**{name}**")
            st.dataframe(pd.DataFrame(report).T)

        selected_model = st.selectbox("Select model to view Confusion Matrix", list(models.keys()))
        model = models[selected_model]
        y_pred = model.predict(X_test)
        cm = pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"])
        st.write("Confusion Matrix")
        st.dataframe(cm)

        st.subheader("ROC Curve")
        plt.figure(figsize=(8,5))
        for name, (fpr, tpr) in rocs.items():
            plt.plot(fpr, tpr, label=name)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.legend()
        plt.title("ROC Curves")
        st.pyplot(plt.gcf())

        st.subheader("Feature Importances")
        import_df = pd.DataFrame(importances, index=X.columns)
        st.bar_chart(import_df)

        st.subheader("Upload new data to predict")
        new_data = st.file_uploader("Upload new data (CSV without target)", type="csv", key="pred_upload")
        if new_data:
            pred_df = pd.read_csv(new_data)
            pred_encoded = pred_df.apply(lambda x: le.fit_transform(x.astype(str)) if x.dtypes == 'object' else x)
            preds = model.predict(pred_encoded)
            pred_df["Prediction"] = preds
            st.write(pred_df)
            st.download_button("Download Predictions", pred_df.to_csv(index=False), "predictions.csv")

    # -----------------------------
    with tab3:
        st.header("📍 Clustering (KMeans)")
        num_clusters = st.slider("Select number of clusters", 2, 10, 3)
        cluster_data = df.select_dtypes(include=["int64", "float64"]).dropna()
        km = KMeans(n_clusters=num_clusters, random_state=42)
        km.fit(cluster_data)
        cluster_data["Cluster"] = km.labels_

        st.subheader("Silhouette Scores")
        silhouette_scores = []
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(cluster_data.drop(columns=["Cluster"]))
            score = silhouette_score(cluster_data.drop(columns=["Cluster"]), labels)
            silhouette_scores.append(score)
        plt.figure()
        plt.plot(range(2, 11), silhouette_scores, marker='o')
        plt.xlabel("k")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score vs K")
        st.pyplot(plt.gcf())

        st.subheader("Clustered Data Sample")
        st.dataframe(cluster_data.head())
        st.download_button("Download Clustered Data", cluster_data.to_csv(index=False), "clustered_data.csv")

    # -----------------------------
    with tab4:
        st.header("🔗 Association Rule Mining")
        col1 = st.selectbox("Column 1 (comma-separated)", df.columns)
        col2 = st.selectbox("Column 2 (comma-separated)", df.columns)
        minsup = st.slider("Minimum Support", 0.01, 1.0, 0.1)
        minconf = st.slider("Minimum Confidence", 0.1, 1.0, 0.5)

        transactions = df[[col1, col2]].dropna().apply(lambda row: list(set(','.join(row).split(','))), axis=1).tolist()
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        frequent = apriori(df_encoded, min_support=minsup, use_colnames=True)
        rules = association_rules(frequent, metric="confidence", min_threshold=minconf)
        st.dataframe(rules.sort_values("confidence", ascending=False).head(10))

    # -----------------------------
    with tab5:
        st.header("📈 Regression Insights")
        reg_target = st.selectbox("Select numeric target variable", df.select_dtypes(include=["int64", "float64"]).columns)
        X_reg = df.select_dtypes(include=["int64", "float64"]).drop(columns=[reg_target])
        y_reg = df[reg_target]

        X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2)

        regressors = {
            "Linear": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "Decision Tree": DecisionTreeRegressor()
        }

        st.subheader("Model Performance")
        perf = {}
        for name, model in regressors.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            perf[name] = {
                "MSE": mean_squared_error(y_test, pred),
                "R²": r2_score(y_test, pred)
            }
        st.dataframe(pd.DataFrame(perf).T)

        st.subheader("Actual vs Predicted")
        model = regressors["Linear"]
        pred = model.predict(X_test)
        fig = plt.figure()
        plt.scatter(y_test, pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Linear Regression: Actual vs Predicted")
        st.pyplot(fig)
