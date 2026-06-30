"""
ML Model Comparison Dashboard
A fully interactive Streamlit app for uploading datasets, training/comparing
supervised & unsupervised ML models, making predictions, and downloading results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import warnings

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Supervised – Classification & Regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB

# Supervised – Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)

# Unsupervised
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ML Model Comparison Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Main header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF, #3F8EFC, #00D2FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.05rem;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1A1F2E, #252B3B);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(108, 99, 255, 0.18);
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e18, #111827);
    }
    section[data-testid="stSidebar"] .stRadio > label { display: none; }
    section[data-testid="stSidebar"] .stRadio > div {
        gap: 2px;
    }
    section[data-testid="stSidebar"] .stRadio > div > label {
        padding: 10px 14px;
        border-radius: 10px;
        transition: all 0.2s ease;
        font-size: 0.92rem;
    }
    section[data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(108, 99, 255, 0.1);
    }
    section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.2), rgba(63, 142, 252, 0.15));
        border-left: 3px solid #6C63FF;
    }
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #3F8EFC);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(108, 99, 255, 0.4);
    }
    /* Custom boxes */
    .success-box {
        background: linear-gradient(135deg, #0d2818, #132a1c);
        border-left: 4px solid #00D26A;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 0.95rem;
    }
    .info-box {
        background: linear-gradient(135deg, #0d1b2a, #1a2332);
        border-left: 4px solid #3F8EFC;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 0.95rem;
    }
    .warning-box {
        background: linear-gradient(135deg, #2a1f0d, #332a1a);
        border-left: 4px solid #FFB020;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 0.95rem;
    }
    .tip-box {
        background: linear-gradient(135deg, #1a1a2e, #252540);
        border-left: 4px solid #A78BFA;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #ccc;
    }
    /* Divider */
    .section-divider {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #6C63FF, transparent);
        margin: 2rem 0;
    }
    /* Card container */
    .glass-card {
        background: rgba(26, 31, 46, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    /* Progress steps */
    .step-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 6px;
        vertical-align: middle;
    }
    .step-done { background: #00D26A22; color: #00D26A; }
    .step-pending { background: #ffffff10; color: #666; }
    /* Table styling */
    .dataframe { border-radius: 10px; overflow: hidden; }
    /* Expander */
    .streamlit-expanderHeader { font-weight: 600; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Session-state helpers
# ──────────────────────────────────────────────
_DEFAULTS = dict(
    df=None, df_processed=None, target_col=None, task_type=None,
    X_train=None, X_test=None, y_train=None, y_test=None,
    trained_models={}, model_results=None, best_model_name=None,
    best_model=None, cluster_results=None, prediction_history=[],
    feature_columns=None, scaler=None, label_encoders={},
    preprocessing_done=False, supervised_done=False, unsupervised_done=False,
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ──────────────────────────────────────────────
# Sidebar navigation with progress tracker
# ──────────────────────────────────────────────
st.sidebar.markdown(
    '<div style="text-align:center;padding:1rem 0 0.5rem;">'
    '<span style="font-size:2rem;">🤖</span><br>'
    '<span style="font-size:1.15rem;font-weight:700;'
    'background:linear-gradient(135deg,#6C63FF,#00D2FF);'
    '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
    'ML Dashboard</span></div>', unsafe_allow_html=True,
)

# Workflow progress
steps_status = {
    "data": "✅" if st.session_state.df is not None else "1️⃣",
    "preprocess": "✅" if st.session_state.preprocessing_done else ("2️⃣" if st.session_state.df is not None else "🔒"),
    "supervised": "✅" if st.session_state.supervised_done else ("3️⃣" if st.session_state.preprocessing_done else "🔒"),
    "unsupervised": "✅" if st.session_state.unsupervised_done else ("4️⃣" if st.session_state.preprocessing_done else "🔒"),
    "best": "✅" if st.session_state.best_model is not None else "🔒",
}

completed = sum(1 for v in steps_status.values() if v == "✅")
st.sidebar.markdown(
    f'<div style="text-align:center;padding:0.3rem;margin:0.5rem 0.8rem;'
    f'background:linear-gradient(135deg,#1A1F2E,#252B3B);border-radius:10px;'
    f'border:1px solid #333;">'
    f'<span style="font-size:0.8rem;color:#888;">Progress</span><br>'
    f'<span style="font-size:1.1rem;font-weight:700;color:#6C63FF;">{completed}/5</span>'
    f'<span style="font-size:0.8rem;color:#666;"> steps done</span></div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

SECTIONS = [
    f"{steps_status['data']} Dataset Upload",
    "🔍 Data Exploration",
    "📊 Visualization",
    f"{steps_status['preprocess']} Preprocessing",
    f"{steps_status['supervised']} Supervised Models",
    f"{steps_status['unsupervised']} Unsupervised Models",
    f"{steps_status['best']} Best Model",
    "🎯 Prediction",
    "📥 Download Results",
]
section = st.sidebar.radio("Navigate", SECTIONS, label_visibility="collapsed")

# Sidebar tips
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="padding:0.8rem;margin:0 0.3rem;background:#1A1F2E;'
    'border-radius:10px;border:1px solid #252B3B;">'
    '<span style="font-size:0.82rem;color:#A78BFA;font-weight:600;">💡 Quick Tip</span><br>'
    '<span style="font-size:0.78rem;color:#888;line-height:1.5;">'
    'Follow the steps in order: Upload → Preprocess → Train for best results.</span></div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<p style="text-align:center;color:#444;font-size:0.75rem;margin-top:1rem;">'
    'Built with ❤️ by Vivek Singh</p>',
    unsafe_allow_html=True,
)

# Normalize section name for matching (strip dynamic emojis)
def _sec(s):
    for token in ["✅","1️⃣","2️⃣","3️⃣","4️⃣","🔒","🔍","📊","🎯","📥"]:
        s = s.replace(token, "")
    return s.strip()
section_key = _sec(section)


# ══════════════════════════════════════════════
# 1. DATASET UPLOAD
# ══════════════════════════════════════════════
if section_key == "Dataset Upload":
    st.markdown('<h1 class="main-header">ML Model Comparison Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload → Explore → Preprocess → Train → Compare → Predict → Download</p>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown(
        '<div class="tip-box">💡 <strong>Getting started:</strong> Upload any CSV file to begin. '
        'The app will auto-detect column types, show missing values, and guide you through '
        'the entire ML workflow step by step.</div>', unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload your CSV dataset", type=["csv"], help="Maximum 200 MB")

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.success(f"✅ Dataset loaded — **{df.shape[0]:,}** rows × **{df.shape[1]}** columns")

            st.markdown("### 📋 Dataset Preview")
            st.dataframe(df.head(20), use_container_width=True, height=350)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", f"{df.shape[0]:,}")
            col2.metric("Columns", f"{df.shape[1]:,}")
            col3.metric("Numeric Cols", f"{df.select_dtypes(include=np.number).shape[1]}")
            col4.metric("Categorical Cols", f"{df.select_dtypes(include='object').shape[1]}")

            st.markdown("### 📑 Column Information")
            info_df = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Non-Null": df.notnull().sum().values,
                "Null": df.isnull().sum().values,
                "Null %": (df.isnull().sum().values / len(df) * 100).round(2),
                "Unique": df.nunique().values,
            })
            st.dataframe(info_df, use_container_width=True, hide_index=True)

            if df.isnull().sum().sum() > 0:
                st.warning(f"⚠️ Dataset contains **{df.isnull().sum().sum()}** missing values across **{(df.isnull().sum() > 0).sum()}** columns.")
            else:
                st.markdown('<div class="success-box">✅ No missing values detected!</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    else:
        st.markdown(
            '<div class="info-box">👆 <strong>Drag & drop</strong> a CSV file above, '
            'or click <strong>Browse files</strong> to select one from your computer.</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════
# 2. DATA EXPLORATION
# ══════════════════════════════════════════════
elif section_key == "Data Exploration":
    st.markdown('<h1 class="main-header">Data Exploration</h1>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        df = st.session_state.df

        # Statistical summary
        st.markdown("### 📈 Statistical Summary")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Categorical column analysis
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        if cat_cols:
            st.markdown("### 🏷️ Categorical Column Analysis")
            sel_cat = st.selectbox("Select a categorical column", cat_cols)
            vc = df[sel_cat].value_counts().head(20)
            fig = px.bar(
                x=vc.index.astype(str), y=vc.values,
                labels={"x": sel_cat, "y": "Count"},
                title=f"Value Counts — {sel_cat}",
                color=vc.values,
                color_continuous_scale="Viridis",
            )
            fig.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Outlier detection (IQR)
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            st.markdown("### 🔎 Outlier Detection (IQR Method)")
            outlier_info = []
            for c in num_cols:
                q1 = df[c].quantile(0.25)
                q3 = df[c].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                n_outliers = ((df[c] < lower) | (df[c] > upper)).sum()
                outlier_info.append({"Column": c, "Q1": round(q1, 2), "Q3": round(q3, 2),
                                     "IQR": round(iqr, 2), "Lower": round(lower, 2),
                                     "Upper": round(upper, 2), "Outliers": n_outliers})
            st.dataframe(pd.DataFrame(outlier_info), use_container_width=True, hide_index=True)

        # Class balance check
        st.markdown("### ⚖️ Class Balance Check")
        potential_target = st.selectbox("Select potential target column", df.columns.tolist(), key="class_bal")
        if df[potential_target].nunique() <= 30:
            vc = df[potential_target].value_counts()
            fig = px.pie(values=vc.values, names=vc.index.astype(str),
                         title=f"Class Distribution — {potential_target}",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.histogram(df, x=potential_target, title=f"Distribution — {potential_target}",
                               color_discrete_sequence=["#6C63FF"])
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# 3. VISUALIZATION
# ══════════════════════════════════════════════
elif section_key == "Visualization":
    st.markdown('<h1 class="main-header">Data Visualization</h1>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        df = st.session_state.df
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        all_cols = df.columns.tolist()

        viz_type = st.selectbox(
            "Select Visualization",
            ["Correlation Heatmap", "Feature Distributions", "Pair Plot",
             "Histogram", "Scatter Plot", "Box Plot"],
        )

        if viz_type == "Correlation Heatmap":
            if len(num_cols) < 2:
                st.warning("Need at least 2 numeric columns.")
            else:
                corr = df[num_cols].corr()
                fig = px.imshow(
                    corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Correlation Heatmap",
                )
                fig.update_layout(template="plotly_dark", height=600)
                st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Feature Distributions":
            sel_cols = st.multiselect("Select numeric columns", num_cols, default=num_cols[:4])
            if sel_cols:
                fig = make_subplots(rows=1, cols=len(sel_cols),
                                    subplot_titles=sel_cols)
                colors = px.colors.qualitative.Set2
                for i, c in enumerate(sel_cols):
                    fig.add_trace(
                        go.Histogram(x=df[c], name=c,
                                     marker_color=colors[i % len(colors)]),
                        row=1, col=i + 1,
                    )
                fig.update_layout(template="plotly_dark", height=400,
                                  showlegend=False, title="Feature Distributions")
                st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Pair Plot":
            sel_cols = st.multiselect("Select columns (max 5 recommended)", num_cols,
                                      default=num_cols[:min(4, len(num_cols))])
            color_col = st.selectbox("Color by (optional)", [None] + all_cols)
            if sel_cols and len(sel_cols) >= 2:
                fig = px.scatter_matrix(
                    df, dimensions=sel_cols,
                    color=color_col if color_col else None,
                    title="Pair Plot",
                    color_continuous_scale="Viridis",
                )
                fig.update_layout(template="plotly_dark", height=700)
                st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Histogram":
            col = st.selectbox("Select column", all_cols)
            nbins = st.slider("Number of bins", 10, 100, 30)
            fig = px.histogram(df, x=col, nbins=nbins, title=f"Histogram — {col}",
                               color_discrete_sequence=["#6C63FF"],
                               marginal="box")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Scatter Plot":
            c1, c2 = st.columns(2)
            x_col = c1.selectbox("X axis", all_cols, index=0)
            y_col = c2.selectbox("Y axis", all_cols, index=min(1, len(all_cols) - 1))
            color_col = st.selectbox("Color by (optional)", [None] + all_cols, key="scatter_color")
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col if color_col else None,
                             title=f"{y_col} vs {x_col}",
                             color_continuous_scale="Viridis")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Box Plot":
            col = st.selectbox("Select numeric column", num_cols)
            group_col = st.selectbox("Group by (optional)", [None] + all_cols, key="box_group")
            fig = px.box(df, y=col, x=group_col if group_col else None,
                         title=f"Box Plot — {col}",
                         color=group_col if group_col else None,
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# 4. PREPROCESSING
# ══════════════════════════════════════════════
elif section_key == "Preprocessing":
    st.markdown('<h1 class="main-header">Data Preprocessing</h1>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        df = st.session_state.df.copy()
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        # Target column
        st.markdown("### 🎯 Target Column")
        target_col = st.selectbox("Select target column", df.columns.tolist())
        st.session_state.target_col = target_col

        # Auto-detect task type
        if df[target_col].nunique() <= 20 or df[target_col].dtype == "object":
            task_type = "Classification"
        else:
            task_type = "Regression"
        task_override = st.radio("Task type", ["Classification", "Regression"],
                                  index=0 if task_type == "Classification" else 1)
        st.session_state.task_type = task_override

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Missing values
        st.markdown("### 🩹 Handle Missing Values")
        missing_total = df.isnull().sum().sum()
        if missing_total > 0:
            miss_strategy = st.selectbox(
                "Strategy",
                ["Drop rows with missing values", "Fill with mean (numeric)",
                 "Fill with median (numeric)", "Fill with mode (all)"],
            )
        else:
            st.markdown('<div class="success-box">✅ No missing values — nothing to do.</div>', unsafe_allow_html=True)
            miss_strategy = None

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Encode categoricals
        st.markdown("### 🔤 Encode Categorical Variables")
        if cat_cols:
            encode_method = st.radio("Encoding method", ["Label Encoding", "One-Hot Encoding"])
        else:
            st.markdown('<div class="success-box">✅ No categorical columns detected.</div>', unsafe_allow_html=True)
            encode_method = None

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Feature scaling
        st.markdown("### 📏 Feature Scaling")
        scaling_method = st.radio("Scaler", ["None", "StandardScaler", "MinMaxScaler"])

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Train-test split
        st.markdown("### ✂️ Train-Test Split")
        test_size = st.slider("Test size (%)", 10, 50, 20) / 100
        random_state = st.number_input("Random state", value=42, step=1)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # APPLY button
        if st.button("🚀 Apply Preprocessing", use_container_width=True):
            with st.spinner("Processing..."):
                # 1. Missing values
                if miss_strategy == "Drop rows with missing values":
                    df.dropna(inplace=True)
                elif miss_strategy == "Fill with mean (numeric)":
                    for c in num_cols:
                        df[c].fillna(df[c].mean(), inplace=True)
                    for c in cat_cols:
                        df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else "Unknown", inplace=True)
                elif miss_strategy == "Fill with median (numeric)":
                    for c in num_cols:
                        df[c].fillna(df[c].median(), inplace=True)
                    for c in cat_cols:
                        df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else "Unknown", inplace=True)
                elif miss_strategy == "Fill with mode (all)":
                    for c in df.columns:
                        if not df[c].mode().empty:
                            df[c].fillna(df[c].mode()[0], inplace=True)

                # 2. Encode categoricals
                label_encoders = {}
                cat_cols_current = df.select_dtypes(include="object").columns.tolist()
                if encode_method == "Label Encoding" and cat_cols_current:
                    for c in cat_cols_current:
                        le = LabelEncoder()
                        df[c] = le.fit_transform(df[c].astype(str))
                        label_encoders[c] = le
                elif encode_method == "One-Hot Encoding" and cat_cols_current:
                    df = pd.get_dummies(df, columns=[c for c in cat_cols_current if c != target_col],
                                        drop_first=True)
                    if df[target_col].dtype == "object":
                        le = LabelEncoder()
                        df[target_col] = le.fit_transform(df[target_col].astype(str))
                        label_encoders[target_col] = le

                st.session_state.label_encoders = label_encoders

                # 3. Split
                X = df.drop(columns=[target_col])
                y = df[target_col]

                # Ensure all numeric
                X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
                y = pd.to_numeric(y, errors="coerce").fillna(0) if task_override == "Regression" else y

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=int(random_state),
                )

                # 4. Scaling
                if scaling_method == "StandardScaler":
                    scaler = StandardScaler()
                    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
                    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
                    st.session_state.scaler = scaler
                elif scaling_method == "MinMaxScaler":
                    scaler = MinMaxScaler()
                    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
                    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
                    st.session_state.scaler = scaler
                else:
                    st.session_state.scaler = None

                st.session_state.df_processed = df
                st.session_state.feature_columns = X.columns.tolist()
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.preprocessing_done = True

            st.success("✅ Preprocessing complete!")

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Samples", f"{len(df):,}")
            c2.metric("Train Samples", f"{len(X_train):,}")
            c3.metric("Test Samples", f"{len(X_test):,}")

            st.markdown("### Processed Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)


# ══════════════════════════════════════════════
# 5. SUPERVISED MODELS
# ══════════════════════════════════════════════
elif section_key == "Supervised Models":
    st.markdown('<h1 class="main-header">Supervised Learning</h1>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if not st.session_state.preprocessing_done:
        st.warning("⚠️ Please complete preprocessing first.")
    else:
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        task = st.session_state.task_type

        st.markdown(f'<div class="info-box">📌 Task type: <strong>{task}</strong></div>', unsafe_allow_html=True)

        if task == "Classification":
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "KNN": KNeighborsClassifier(),
                "SVM": SVC(probability=True, random_state=42),
                "Naive Bayes": GaussianNB(),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            }
        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "KNN": KNeighborsRegressor(),
                "SVM": SVR(),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            }

        if st.button("🚀 Train All Models", use_container_width=True):
            results = []
            trained = {}
            progress = st.progress(0, text="Training models…")
            total = len(models)

            # Determine safe number of CV folds based on smallest class
            if task == "Classification":
                min_class_size = int(y_train.value_counts().min())
                n_cv = max(2, min(5, min_class_size))
            else:
                n_cv = 5

            for i, (name, model) in enumerate(models.items()):
                progress.progress((i) / total, text=f"Training {name}…")
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    trained[name] = model

                    if task == "Classification":
                        acc = accuracy_score(y_test, y_pred)
                        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                        # Cross-validation
                        try:
                            cv_scores = cross_val_score(model, X_train, y_train, cv=n_cv, scoring="accuracy")
                            cv_mean, cv_std = round(cv_scores.mean(), 4), round(cv_scores.std(), 4)
                        except Exception:
                            cv_mean, cv_std = None, None
                        results.append({
                            "Model": name, "Accuracy": round(acc, 4),
                            "Precision": round(prec, 4), "Recall": round(rec, 4),
                            "F1 Score": round(f1, 4),
                            "CV Mean": cv_mean,
                            "CV Std": cv_std,
                        })
                    else:
                        mae = mean_absolute_error(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        try:
                            cv_scores = cross_val_score(model, X_train, y_train, cv=n_cv, scoring="r2")
                            cv_mean, cv_std = round(cv_scores.mean(), 4), round(cv_scores.std(), 4)
                        except Exception:
                            cv_mean, cv_std = None, None
                        results.append({
                            "Model": name, "MAE": round(mae, 4),
                            "MSE": round(mse, 4), "R² Score": round(r2, 4),
                            "CV Mean R²": cv_mean,
                            "CV Std": cv_std,
                        })
                except Exception as e:
                    st.warning(f"⚠️ {name} failed: {e}")

            progress.progress(1.0, text="✅ Done!")
            st.session_state.trained_models = trained
            results_df = pd.DataFrame(results)
            st.session_state.model_results = results_df
            st.session_state.supervised_done = True

            if results_df.empty:
                st.error("❌ All models failed to train. Try adjusting preprocessing or using a different dataset.")
            else:
                # Determine best model
                if task == "Classification":
                    best_idx = results_df["Accuracy"].idxmax()
                    primary_metric = "Accuracy"
                else:
                    best_idx = results_df["R² Score"].idxmax()
                    primary_metric = "R² Score"

                best_name = results_df.loc[best_idx, "Model"]
                st.session_state.best_model_name = best_name
                st.session_state.best_model = trained[best_name]

                st.markdown("### 📊 Model Comparison Table")
                st.dataframe(
                    results_df.style.highlight_max(
                        subset=[primary_metric], color="#2d6a4f"
                    ).format(precision=4),
                    use_container_width=True, hide_index=True,
                )

                # Bar chart comparison
                st.markdown("### 📈 Performance Comparison")
                fig = px.bar(
                    results_df, x="Model", y=primary_metric,
                    color=primary_metric,
                    color_continuous_scale="Viridis",
                    title=f"Model Comparison — {primary_metric}",
                    text=primary_metric,
                )
                fig.update_layout(template="plotly_dark", height=450)
                fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

                # Confusion matrices (classification)
                if task == "Classification":
                    st.markdown("### 🔢 Confusion Matrices")
                    n_display = min(3, len(trained))
                    if n_display > 0:
                        cols = st.columns(n_display)
                        for idx, (name, model) in enumerate(trained.items()):
                            with cols[idx % n_display]:
                                y_pred = model.predict(X_test)
                                cm = confusion_matrix(y_test, y_pred)
                                fig_cm = px.imshow(cm, text_auto=True,
                                                   color_continuous_scale="Blues",
                                                   title=name, aspect="auto")
                                fig_cm.update_layout(template="plotly_dark", height=300,
                                                     margin=dict(t=40, b=10))
                                st.plotly_chart(fig_cm, use_container_width=True)

                # Feature importance (tree-based)
                st.markdown("### 🌲 Feature Importance (Tree-Based Models)")
                tree_models = {n: m for n, m in trained.items()
                              if hasattr(m, "feature_importances_")}
                if tree_models:
                    sel_tree = st.selectbox("Select model", list(tree_models.keys()))
                    importances = tree_models[sel_tree].feature_importances_
                    feat_imp_df = pd.DataFrame({
                        "Feature": st.session_state.feature_columns,
                        "Importance": importances,
                    }).sort_values("Importance", ascending=True).tail(15)

                    fig = px.bar(feat_imp_df, x="Importance", y="Feature",
                                 orientation="h", title=f"Feature Importance — {sel_tree}",
                                 color="Importance", color_continuous_scale="Viridis")
                    fig.update_layout(template="plotly_dark", height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No tree-based models available for feature importance.")


# ══════════════════════════════════════════════
# 6. UNSUPERVISED MODELS
# ══════════════════════════════════════════════
elif section_key == "Unsupervised Models":
    st.markdown('<h1 class="main-header">Unsupervised Learning</h1>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if not st.session_state.preprocessing_done:
        st.warning("⚠️ Please complete preprocessing first.")
    else:
        X_full = pd.concat([st.session_state.X_train, st.session_state.X_test])

        st.markdown("### ⚙️ Clustering Parameters")
        c1, c2 = st.columns(2)
        n_clusters = c1.slider("Number of clusters (K-Means, Hierarchical, GMM)", 2, 10, 3)
        eps_val = c2.slider("DBSCAN eps", 0.1, 5.0, 0.5, 0.1)
        min_samples = c2.slider("DBSCAN min_samples", 2, 20, 5)

        if st.button("🚀 Train Clustering Models", use_container_width=True):
            cluster_results = []
            cluster_labels_all = {}

            with st.spinner("Training clustering models…"):
                # K-Means
                try:
                    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    km_labels = km.fit_predict(X_full)
                    cluster_labels_all["K-Means"] = km_labels
                    sil = silhouette_score(X_full, km_labels)
                    db = davies_bouldin_score(X_full, km_labels)
                    cluster_results.append({"Model": "K-Means", "Silhouette": round(sil, 4),
                                            "Davies-Bouldin": round(db, 4), "Clusters": n_clusters})
                except Exception as e:
                    st.warning(f"K-Means failed: {e}")

                # Hierarchical
                try:
                    hc = AgglomerativeClustering(n_clusters=n_clusters)
                    hc_labels = hc.fit_predict(X_full)
                    cluster_labels_all["Hierarchical"] = hc_labels
                    sil = silhouette_score(X_full, hc_labels)
                    db = davies_bouldin_score(X_full, hc_labels)
                    cluster_results.append({"Model": "Hierarchical", "Silhouette": round(sil, 4),
                                            "Davies-Bouldin": round(db, 4), "Clusters": n_clusters})
                except Exception as e:
                    st.warning(f"Hierarchical failed: {e}")

                # DBSCAN
                try:
                    dbs = DBSCAN(eps=eps_val, min_samples=min_samples)
                    dbs_labels = dbs.fit_predict(X_full)
                    n_found = len(set(dbs_labels) - {-1})
                    cluster_labels_all["DBSCAN"] = dbs_labels
                    if n_found >= 2:
                        mask = dbs_labels != -1
                        sil = silhouette_score(X_full[mask], dbs_labels[mask]) if mask.sum() > 1 else -1
                        db = davies_bouldin_score(X_full[mask], dbs_labels[mask]) if mask.sum() > 1 else -1
                    else:
                        sil, db = -1, -1
                    cluster_results.append({"Model": "DBSCAN", "Silhouette": round(sil, 4),
                                            "Davies-Bouldin": round(db, 4), "Clusters": n_found})
                except Exception as e:
                    st.warning(f"DBSCAN failed: {e}")

                # Gaussian Mixture
                try:
                    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
                    gmm_labels = gmm.fit_predict(X_full)
                    cluster_labels_all["Gaussian Mixture"] = gmm_labels
                    sil = silhouette_score(X_full, gmm_labels)
                    db = davies_bouldin_score(X_full, gmm_labels)
                    cluster_results.append({"Model": "Gaussian Mixture", "Silhouette": round(sil, 4),
                                            "Davies-Bouldin": round(db, 4), "Clusters": n_clusters})
                except Exception as e:
                    st.warning(f"GMM failed: {e}")

            cr_df = pd.DataFrame(cluster_results)
            st.session_state.cluster_results = cr_df
            st.session_state.unsupervised_done = True

            st.markdown("### 📊 Clustering Comparison")
            st.dataframe(
                cr_df.style.highlight_max(subset=["Silhouette"], color="#2d6a4f")
                     .highlight_min(subset=["Davies-Bouldin"], color="#2d6a4f")
                     .format(precision=4),
                use_container_width=True, hide_index=True,
            )

            # Bar chart
            fig = px.bar(cr_df, x="Model", y="Silhouette",
                         color="Silhouette", color_continuous_scale="Viridis",
                         title="Silhouette Score Comparison", text="Silhouette")
            fig.update_layout(template="plotly_dark")
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            # Cluster scatter plots (PCA 2D)
            st.markdown("### 🗺️ Cluster Visualization (PCA 2D)")
            if X_full.shape[1] >= 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_full)
                cols = st.columns(2)
                for idx, (name, labels) in enumerate(cluster_labels_all.items()):
                    with cols[idx % 2]:
                        scatter_df = pd.DataFrame({
                            "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
                            "Cluster": labels.astype(str),
                        })
                        fig = px.scatter(scatter_df, x="PC1", y="PC2", color="Cluster",
                                         title=name,
                                         color_discrete_sequence=px.colors.qualitative.Set2)
                        fig.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# 7. BEST MODEL
# ══════════════════════════════════════════════
elif section_key == "Best Model":
    st.markdown('<h1 class="main-header">Best Model Selection</h1>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if not st.session_state.supervised_done:
        st.warning("⚠️ Please train supervised models first.")
    else:
        best_name = st.session_state.best_model_name
        results_df = st.session_state.model_results
        task = st.session_state.task_type

        st.markdown(f"""
        <div style="text-align:center; padding: 2rem; background: linear-gradient(135deg, #1A1F2E, #252B3B);
                    border-radius: 16px; border: 2px solid #6C63FF; margin: 1rem 0;">
            <h2 style="color:#6C63FF; margin-bottom: 0.5rem;">🏆 Best Model</h2>
            <h1 style="color:#00D2FF; font-size: 2.5rem;">{best_name}</h1>
        </div>
        """, unsafe_allow_html=True)

        # Show best model's metrics
        best_row = results_df[results_df["Model"] == best_name].iloc[0]
        if task == "Classification":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{best_row['Accuracy']:.4f}")
            c2.metric("Precision", f"{best_row['Precision']:.4f}")
            c3.metric("Recall", f"{best_row['Recall']:.4f}")
            c4.metric("F1 Score", f"{best_row['F1 Score']:.4f}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{best_row['MAE']:.4f}")
            c2.metric("MSE", f"{best_row['MSE']:.4f}")
            c3.metric("R² Score", f"{best_row['R² Score']:.4f}")

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Learning curve
        st.markdown("### 📈 Learning Curve")
        with st.spinner("Generating learning curve…"):
            best_model = st.session_state.best_model
            X_full = pd.concat([st.session_state.X_train, st.session_state.X_test])
            y_full = pd.concat([st.session_state.y_train, st.session_state.y_test])

            scoring = "accuracy" if task == "Classification" else "r2"
            # Dynamic CV folds for small datasets
            if task == "Classification":
                min_class_size = int(y_full.value_counts().min())
                lc_cv = max(2, min(5, min_class_size))
            else:
                lc_cv = 5
            try:
                train_sizes, train_scores, val_scores = learning_curve(
                    best_model, X_full, y_full, cv=lc_cv,
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    scoring=scoring, n_jobs=-1,
                )
                lc_df = pd.DataFrame({
                    "Training Size": np.tile(train_sizes, 2),
                    "Score": np.concatenate([train_scores.mean(axis=1), val_scores.mean(axis=1)]),
                    "Type": ["Train"] * len(train_sizes) + ["Validation"] * len(train_sizes),
                })
                fig = px.line(lc_df, x="Training Size", y="Score", color="Type",
                              title=f"Learning Curve — {best_name}",
                              color_discrete_map={"Train": "#6C63FF", "Validation": "#00D2FF"})
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate learning curve: {e}")

        # All results table
        st.markdown("### 📋 All Model Results")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Cluster results if available
        if st.session_state.cluster_results is not None:
            st.markdown("### 🔮 Clustering Results")
            st.dataframe(st.session_state.cluster_results, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# 8. PREDICTION
# ══════════════════════════════════════════════
elif section_key == "Prediction":
    st.markdown('<h1 class="main-header">Make Predictions</h1>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if st.session_state.best_model is None:
        st.warning("⚠️ Please train models to enable predictions.")
    else:
        best = st.session_state.best_model
        features = st.session_state.feature_columns
        task = st.session_state.task_type

        st.markdown(f'<div class="info-box">Using <strong>{st.session_state.best_model_name}</strong> '
                    f'for {task.lower()} predictions.</div>', unsafe_allow_html=True)

        st.markdown("### ✏️ Enter Feature Values")

        # Build input form dynamically
        input_values = {}
        cols = st.columns(3)
        for i, feat in enumerate(features):
            with cols[i % 3]:
                # Use training data stats for sensible defaults
                col_data = st.session_state.X_train[feat]
                default = float(col_data.median())
                input_values[feat] = st.number_input(
                    feat, value=default,
                    format="%.4f", key=f"pred_{feat}",
                )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        if st.button("🎯 Predict", use_container_width=True):
            input_df = pd.DataFrame([input_values])

            # Apply scaling if used
            if st.session_state.scaler is not None:
                input_df = pd.DataFrame(
                    st.session_state.scaler.transform(input_df),
                    columns=features,
                )

            prediction = best.predict(input_df)[0]

            # Decode if label-encoded target
            target_col = st.session_state.target_col
            if target_col in st.session_state.label_encoders:
                le = st.session_state.label_encoders[target_col]
                try:
                    prediction = le.inverse_transform([int(prediction)])[0]
                except Exception:
                    pass

            st.markdown(f"""
            <div style="text-align:center; padding: 2rem; background: linear-gradient(135deg, #1a3a2a, #1A1F2E);
                        border-radius: 16px; border: 2px solid #00D26A; margin: 1rem 0;">
                <h3 style="color:#888;">Prediction Result</h3>
                <h1 style="color:#00D26A; font-size: 3rem;">{prediction}</h1>
            </div>
            """, unsafe_allow_html=True)

            # Save to history
            record = dict(input_values)
            record["Prediction"] = prediction
            st.session_state.prediction_history.append(record)

        # Show prediction history
        if st.session_state.prediction_history:
            st.markdown("### 📜 Prediction History")
            hist_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(hist_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# 9. DOWNLOAD RESULTS
# ══════════════════════════════════════════════
elif section_key == "Download Results":
    st.markdown('<h1 class="main-header">Download Results</h1>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("### 📦 Available Downloads")

    c1, c2 = st.columns(2)

    # Trained model (.pkl)
    with c1:
        st.markdown("#### 🤖 Best Trained Model (.pkl)")
        if st.session_state.best_model is not None:
            model_bytes = pickle.dumps(st.session_state.best_model)
            st.download_button(
                "⬇️ Download Model",
                data=model_bytes,
                file_name=f"{st.session_state.best_model_name.replace(' ', '_')}_model.pkl",
                mime="application/octet-stream",
                use_container_width=True,
            )
        else:
            st.info("Train models first.")

    # Processed dataset
    with c2:
        st.markdown("#### 📄 Processed Dataset (.csv)")
        if st.session_state.df_processed is not None:
            csv_data = st.session_state.df_processed.to_csv(index=False)
            st.download_button(
                "⬇️ Download Processed Data",
                data=csv_data,
                file_name="processed_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Process data first.")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    c3, c4 = st.columns(2)

    # Prediction results
    with c3:
        st.markdown("#### 🎯 Prediction Results (.csv)")
        if st.session_state.prediction_history:
            pred_df = pd.DataFrame(st.session_state.prediction_history)
            csv_pred = pred_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Predictions",
                data=csv_pred,
                file_name="prediction_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Make some predictions first.")

    # Model comparison report
    with c4:
        st.markdown("#### 📊 Model Comparison Report (.csv)")
        if st.session_state.model_results is not None:
            report_csv = st.session_state.model_results.to_csv(index=False)
            st.download_button(
                "⬇️ Download Report",
                data=report_csv,
                file_name="model_comparison_report.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Train models first.")

    # Cluster report
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("#### 🔮 Clustering Report (.csv)")
    if st.session_state.cluster_results is not None:
        clust_csv = st.session_state.cluster_results.to_csv(index=False)
        st.download_button(
            "⬇️ Download Clustering Report",
            data=clust_csv,
            file_name="clustering_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("Train clustering models first.")
