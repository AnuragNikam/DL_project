# streamlit.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

st.set_page_config(page_title="Road Accident Severity Model", layout="wide")

st.title("🚗 Road Accident Severity Prediction (Deep Learning)")
st.write(
    "Upload your **Road Accident Data.csv**, train a deep learning model, "
    "and view accuracy, confusion matrix, and predictions."
)

# ---------------------------------------------
# 1. File Upload
# ---------------------------------------------
uploaded_file = st.file_uploader("Upload Road Accident Data CSV", type=["csv"])

if uploaded_file is None:
    st.info("Please upload `Road Accident Data.csv` to continue.")
    st.stop()

# ---------------------------------------------
# 2. Load Dataset
# ---------------------------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

df = load_data(uploaded_file)

st.subheader("📊 Raw Data Preview")
st.write(df.head())
st.write("Shape:", df.shape)

# ---------------------------------------------
# 3. Preprocessing Function
# ---------------------------------------------
TARGET_COL = "Accident_Severity"

def preprocess_data(df: pd.DataFrame):
    df = df.copy()

    # --- Date & time features ---
    if 'Accident Date' in df.columns:
        df['Accident Date'] = pd.to_datetime(df['Accident Date'], errors='coerce')
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        df['Hour'] = df['Time'].dt.hour
    else:
        df['Hour'] = np.nan

    if 'Accident Date' in df.columns:
        df['DayOfWeek'] = df['Accident Date'].dt.dayofweek
    else:
        df['DayOfWeek'] = np.nan

    # Drop ID / raw datetime / lat-long
    drop_cols = []
    for col in ['Accident_Index', 'Accident Date', 'Time']:
        if col in df.columns:
            drop_cols.append(col)
    for col in ['Latitude', 'Longitude']:
        if col in df.columns:
            drop_cols.append(col)

    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COL])

    # Separate target and features
    y_raw = df[TARGET_COL]
    X_raw = df.drop(columns=[TARGET_COL])

    # Identify numeric & categorical
    numeric_cols = X_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_raw.select_dtypes(include=['object']).columns.tolist()

    # Fill numeric missing with mean
    for col in numeric_cols:
        X_raw[col] = X_raw[col].fillna(X_raw[col].mean())

    # Fill categorical missing with placeholder
    for col in categorical_cols:
        X_raw[col] = X_raw[col].fillna("Unknown")

    # One-hot encode categoricals
    X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)

    # Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(y_raw)
    class_names = le_target.classes_
    num_classes = len(class_names)

    return X, y, le_target, class_names, num_classes

# ---------------------------------------------
# 4. Apply Preprocessing
# ---------------------------------------------
with st.spinner("Preprocessing data..."):
    try:
        X, y, le_target, class_names, num_classes = preprocess_data(df)
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        st.stop()

st.subheader("🧹 After Preprocessing")
st.write("Number of samples:", X.shape[0])
st.write("Number of features:", X.shape[1])
st.write("Target classes:", class_names)

st.write("Class distribution:")
st.bar_chart(pd.Series(y).value_counts().sort_index())

# ---------------------------------------------
# 5. Train/Test Split & Scaling
# ---------------------------------------------
test_size = st.sidebar.slider("Test Size (fraction)", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", min_value=0, value=42, step=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.write("Train shape:", X_train.shape, " | Test shape:", X_test.shape)

# ---------------------------------------------
# 6. Build Model Function
# ---------------------------------------------
def build_model(input_dim, num_classes):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ---------------------------------------------
# 7. Train Model (on button click)
# ---------------------------------------------
st.subheader("🧠 Train Deep Learning Model")

epochs = st.sidebar.slider("Epochs", 20, 200, 100, 10)
batch_size = st.sidebar.selectbox("Batch Size", [32, 64, 128], index=1)

if st.button("🚀 Train Model"):
    with st.spinner("Training the model..."):
        # Class weights
        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = {i: w for i, w in enumerate(class_weights_array)}

        model = build_model(X_train_scaled.shape[1], num_classes)

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            class_weight=class_weights,
            verbose=0
        )

    st.success("Training completed!")

    # -----------------------------------------
    # 8. Plot Training History
    # -----------------------------------------
    st.subheader("📈 Training History")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history['loss'], label='Train Loss')
    ax[0].plot(history.history['val_loss'], label='Val Loss')
    ax[0].set_title("Loss")
    ax[0].legend()

    ax[1].plot(history.history['accuracy'], label='Train Acc')
    ax[1].plot(history.history['val_accuracy'], label='Val Acc')
    ax[1].set_title("Accuracy")
    ax[1].legend()

    st.pyplot(fig)

    # -----------------------------------------
    # 9. Evaluation
    # -----------------------------------------
    st.subheader("✅ Model Evaluation on Test Set")

    y_proba = model.predict(X_test_scaled)
    y_pred = np.argmax(y_proba, axis=1)

    test_acc = accuracy_score(y_test, y_pred)
    st.write(f"**Test Accuracy:** `{test_acc * 100:.2f}%`")

    st.text("Classification Report:")
    report_str = classification_report(
        y_test, y_pred, target_names=class_names.astype(str)
    )
    st.text(report_str)

    cm = confusion_matrix(y_test, y_pred)

    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax_cm
    )
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

    # -----------------------------------------
    # 10. Prediction on Sample Test Row
    # -----------------------------------------
    st.subheader("🔍 Try a Sample Prediction")

    idx = st.number_input(
        "Choose a test sample index",
        min_value=0,
        max_value=len(X_test_scaled) - 1,
        value=0,
        step=1
    )

    sample_features = X_test_scaled[idx].reshape(1, -1)
    sample_true = y_test[idx]
    sample_pred = np.argmax(model.predict(sample_features), axis=1)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.write("**True Severity:**", class_names[sample_true])
    with col2:
        st.write("**Predicted Severity:**", class_names[sample_pred])

    st.write("Feature values (original scale) for this sample:")
    st.write(pd.DataFrame([X_test.iloc[idx]], columns=X_test.columns))

    # -----------------------------------------
    # 11. Save Model
    # -----------------------------------------
    if st.button("💾 Save Model as road_accident_severity_model_improved.h5"):
        model.save("road_accident_severity_model_improved.h5")
        st.success("Model saved as `road_accident_severity_model_improved.h5` in working directory.")
else:
    st.info("Set parameters in the sidebar and click **Train Model** to start.")
