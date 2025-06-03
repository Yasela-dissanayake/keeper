import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb # Ensure xgboost is imported
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Page Configuration
st.set_page_config(
    page_title="Rain Prediction App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title
st.title("Rain Prediction and EDA App")

# Sidebar
st.sidebar.header("Navigation and Options")

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return None
    else:
        st.info("Please upload a CSV file to begin EDA and model training.")
        return None

    # Display raw data (optional, for verification)
    if st.sidebar.checkbox("Show raw data sample", False):
        st.subheader("Raw Data (Sample)")
        st.write(df.head())

    # Preprocessing
    try:
        # Handle missing values (numerical columns with mean, categorical with most frequent)
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].nunique() < 10: # Consider low-cardinality numerical as categorical for imputation
                imputer = SimpleImputer(strategy='most_frequent')
                df[col] = imputer.fit_transform(df[[col]]).ravel()
            else:
                imputer = SimpleImputer(strategy='mean')
                df[col] = imputer.fit_transform(df[[col]]).ravel()

        # Encode categorical variables
        label_encoders = {}
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        # Ensure target variable 'RainTomorrow' is present
        if 'RainTomorrow' not in df.columns:
            st.error("Target column 'RainTomorrow' not found in the uploaded file.")
            return None

        # Convert all columns to numeric, coercing errors for robustness
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Re-check for NaNs introduced by pd.to_numeric coercion and impute them (e.g., with median)
        if df.isnull().sum().sum() > 0:
            st.warning("NaNs were introduced after numeric conversion. Imputing with median.")
            for col in df.columns[df.isnull().any()]:
                 df[col] = df[col].fillna(df[col].median())

        st.success("Data loaded and preprocessed successfully!")
        return df, label_encoders
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None

# Data Loading in Sidebar
st.sidebar.subheader("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

data_load_state = st.text("Loading data...")
processed_data_tuple = load_and_preprocess_data(uploaded_file)
data_load_state.text("Data loading complete!") # This will display immediately after calling the function

if processed_data_tuple:
    df_processed, le_encoders = processed_data_tuple
    st.session_state.df_processed = df_processed
    st.session_state.le_encoders = le_encoders
else:
    df_processed = None # Ensure df_processed is defined even if loading fails
    # Optionally, clear session state if data loading fails or no file is uploaded
    if 'df_processed' in st.session_state:
        del st.session_state.df_processed
    if 'le_encoders' in st.session_state:
        del st.session_state.le_encoders

# App Sections Radio Button
st.sidebar.subheader("App Sections")
app_section = st.sidebar.radio(
    "Go to",
    ["Exploratory Data Analysis (EDA)", "Model Training & Evaluation", "Rain Prediction"]
)

# --- EDA Section Functions ---
def display_descriptive_stats(df):
    st.subheader("Descriptive Statistics")

    # DataFrame head
    if st.checkbox("Show DataFrame Head"):
        st.write(df.head())

    # DataFrame shape
    if st.checkbox("Show DataFrame Shape"):
        st.write(f"Shape: {df.shape}")

    # Column data types
    if st.checkbox("Show Column Data Types"):
        st.write(df.dtypes.astype(str)) # Convert to string for display

    # Missing values count (after imputation, should be 0)
    if st.checkbox("Show Missing Values Count"):
        missing_vals = df.isnull().sum()
        st.write(missing_vals)
        if missing_vals.sum() == 0:
            st.success("No missing values after imputation.")
        else:
            st.warning("There are still missing values. Please check the preprocessing steps.")

    # Descriptive statistics for numerical columns
    if st.checkbox("Show Numerical Descriptive Statistics"):
        st.write(df.describe())

    # Value counts for the target variable 'RainTomorrow'
    if 'RainTomorrow' in df.columns:
        if st.checkbox("Show Value Counts for RainTomorrow"):
            st.write(df['RainTomorrow'].value_counts())
    else:
        st.warning("'RainTomorrow' column not found for value counts display.")

    # Descriptive statistics for categorical columns (original, before encoding)
    # This requires the original dataframe or knowledge of original categorical columns.
    # For now, we'll show describe(include='all') to give some info,
    # or we can try to infer categorical based on nunique from the processed df.
    if st.checkbox("Show General Descriptive Statistics (include all)"):
            st.write(df.describe(include='all'))

def display_eda_visualizations(df):
    st.subheader("Exploratory Data Visualizations")

    # Ensure 'RainTomorrow' is numeric for plotting if it exists
    if 'RainTomorrow' in df.columns:
        df['RainTomorrow'] = pd.to_numeric(df['RainTomorrow'], errors='coerce')
        # df.dropna(subset=['RainTomorrow'], inplace=True) # Optional: drop rows where target couldn't be coerced

    numerical_cols_for_viz = df.select_dtypes(include=np.number).columns.tolist()
    potential_categoricals_encoded = [col for col in numerical_cols_for_viz if df[col].nunique() < 10 and col != 'RainTomorrow']
    numerical_cols_for_viz = [col for col in numerical_cols_for_viz if col not in potential_categoricals_encoded or col == 'RainTomorrow']

    if not numerical_cols_for_viz:
        st.warning("No clear numerical columns found for some visualizations after preprocessing.")
        numerical_cols_for_viz = df.select_dtypes(include=np.number).columns.tolist()

    # Histograms for Numerical Features
    if st.checkbox("Show Histograms for Numerical Features"):
        if numerical_cols_for_viz:
            st.write("Histograms:")
            n_cols = 3
            n_rows = (len(numerical_cols_for_viz) - 1) // n_cols + 1
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
            axes = axes.flatten()

            for i, col in enumerate(numerical_cols_for_viz):
                if i < len(axes):
                    sns.histplot(df[col], kde=True, ax=axes[i])
                    axes[i].set_title(f'Histogram of {col}')

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No numerical columns selected/available for histograms.")

    # Count plot for Target Variable
    if 'RainTomorrow' in df.columns:
        if st.checkbox("Show Target Variable Distribution (RainTomorrow)"):
            st.write("Target Variable Distribution:")
            fig, ax = plt.subplots(figsize=(6, 4))
            rain_tomorrow_series = df['RainTomorrow'].copy()
            sns.countplot(x=rain_tomorrow_series, ax=ax)
            ax.set_title('Distribution of RainTomorrow')
            st.pyplot(fig)
    else:
        st.warning("Target column 'RainTomorrow' not found for count plot.")

    # Correlation Heatmap
    if st.checkbox("Show Correlation Heatmap (Numerical Features)"):
        if numerical_cols_for_viz:
            st.write("Correlation Heatmap:")
            numeric_df_for_corr = df[numerical_cols_for_viz].apply(pd.to_numeric, errors='coerce')
            numeric_df_for_corr.dropna(axis=1, how='all', inplace=True)

            if not numeric_df_for_corr.empty and numeric_df_for_corr.shape[1] > 1 : # Need at least 2 columns for correlation
                corr = numeric_df_for_corr.corr()
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
                ax.set_title('Correlation Heatmap')
                st.pyplot(fig)
            else:
                st.warning("Not enough numeric data (or <2 columns) to generate a correlation heatmap.")
        else:
            st.info("No numerical columns selected/available for correlation heatmap.")

    # Boxplots for Outliers
    if st.checkbox("Show Boxplots for Numerical Features (Outlier Detection)"):
        if numerical_cols_for_viz:
            st.write("Boxplots:")
            n_cols = 2
            n_rows = (len(numerical_cols_for_viz) - 1) // n_cols + 1
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
            axes = axes.flatten()

            for i, col in enumerate(numerical_cols_for_viz):
                    if i < len(axes):
                        sns.boxplot(y=df[col], ax=axes[i])
                        axes[i].set_title(f'Boxplot of {col}')

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No numerical columns selected/available for boxplots.")

    if 'le_encoders' in st.session_state and st.session_state.le_encoders:
        if st.checkbox("Show Distributions for Originally Categorical Features (after encoding)"):
            st.write("Distributions for Originally Categorical Features (now encoded):")
            encoded_cat_cols = list(st.session_state.le_encoders.keys())

            # Filter out columns that might not exist in df anymore or are not suitable for countplot
            valid_encoded_cat_cols = [col for col in encoded_cat_cols if col in df.columns and df[col].nunique() < 50] # Heuristic limit for countplot

            if valid_encoded_cat_cols:
                n_cols = 2
                n_rows = (len(valid_encoded_cat_cols) - 1) // n_cols + 1
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4))
                axes = axes.flatten()
                plot_idx = 0
                for col_name in valid_encoded_cat_cols:
                    if plot_idx < len(axes):
                        # Ensure data is int for countplot if it's from LabelEncoder
                        data_to_plot = df[col_name]
                        if pd.api.types.is_float_dtype(data_to_plot) and data_to_plot.apply(float.is_integer).all():
                             data_to_plot = data_to_plot.astype(int)

                        sns.countplot(y=data_to_plot, ax=axes[plot_idx], order = data_to_plot.value_counts().index)
                        axes[plot_idx].set_title(f'Distribution of {col_name} (encoded)')
                        plot_idx +=1

                for j in range(plot_idx, len(axes)):
                    if j < len(axes): fig.delaxes(axes[j])

                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No suitable originally categorical features found for count plots or they are not in the processed data.")

# --- Main App Logic Based on Section ---
if app_section == "Exploratory Data Analysis (EDA)":
    st.header("Exploratory Data Analysis (EDA)")
    if 'df_processed' in st.session_state and st.session_state.df_processed is not None:
        df_to_display = st.session_state.df_processed.copy() # Use a copy for plotting to avoid modifying session state df

        st.subheader("Data Overview")
        display_descriptive_stats(df_to_display)

        st.markdown("---") # Separator
        display_eda_visualizations(df_to_display)
    else:
        st.warning("Please upload and preprocess data first in the sidebar.")

elif app_section == "Model Training & Evaluation":
    st.header("Model Training & Evaluation")
    if 'df_processed' in st.session_state and st.session_state.df_processed is not None:
        df_for_training = st.session_state.df_processed.copy() # Use a copy

        model_options = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost']
        selected_models_to_train = st.sidebar.multiselect(
            "Select model(s) to train:",
            model_options,
            default=[model_options[0]] # Default to Logistic Regression in a list
        )

        if not selected_models_to_train:
            st.info("Please select at least one model from the sidebar to train.")
        else:
            if st.sidebar.button("Start Training & Evaluation", key="start_training"):
                with st.spinner("Training models... This may take a few minutes."):
                    train_and_evaluate_models(df_for_training, selected_models_to_train)
                st.success("Model training and evaluation complete!")
    else:
        st.warning("Please upload and preprocess data first in the sidebar.")


elif app_section == "Rain Prediction":
    st.header("Rain Prediction")
    if 'df_processed' not in st.session_state or st.session_state.df_processed is None:
        st.warning("Please upload data in the EDA section first to define features for prediction.")
        return # Early exit if no df_processed, as its columns are needed for input fields

    if 'trained_models' not in st.session_state or not st.session_state.trained_models:
        st.info("No models trained in this session. Please train a model first in the 'Model Training & Evaluation' section.")
        # Placeholder for loading a pre-trained model could go here
        # e.g., if st.button("Load a pre-saved model"): ...
        return # Early exit if no trained models

    df_columns_for_input = st.session_state.df_processed.columns.tolist()

    available_trained_models = list(st.session_state.trained_models.keys())
    if not available_trained_models: # Should be caught by the check above, but good for safety
        st.warning("No models available from the current session.")
        return

    model_to_use_for_prediction = st.selectbox(
        "Select a trained model for prediction:",
        options=available_trained_models,
        key="pred_model_select"
    )

    if model_to_use_for_prediction:
        make_prediction(df_columns_for_input,
                        st.session_state.get('le_encoders', {}),
                        model_to_use_for_prediction)
    else:
        st.info("Please select a model to use for prediction.")


st.sidebar.markdown("---")
st.sidebar.info(
    "This app performs EDA and predicts whether it will rain tomorrow "
    "based on historical weather data."
)
