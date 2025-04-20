
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ฝึกโมเดลฝังไว้ (3 ตาหลังสุด)
@st.cache_resource
def train_model():
    import pandas as pd
    np.random.seed(42)
    results = np.random.choice(['P', 'B', 'T'], size=5000, p=[0.45, 0.45, 0.1])
    df = pd.DataFrame({'result': results})
    result_map = {'P': 0, 'B': 1, 'T': 2}
    df['result_code'] = df['result'].map(result_map)

    for i in range(1, 4):
        df[f'prev_{i}'] = df['result_code'].shift(i)

    df.dropna(inplace=True)

    X = df[['prev_1', 'prev_2', 'prev_3']]
    y = df['result_code']

    model = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# Mapping
label_map = {'P': 0, 'B': 1, 'T': 2}
reverse_map = {0: "Player (P)", 1: "Banker (B)", 2: "Tie (T)"}

# UI
st.title("Baccarat Predictor 3-Turns Pro")
st.write("พิมพ์ผล 3 ตาหลังสุด เช่น: `B P B` หรือ `P P T`")

input_text = st.text_input("ผล 3 ตาหลังสุด (คั่นด้วยเว้นวรรค)").strip().upper()

if input_text:
    tokens = input_text.split()
    if len(tokens) != 3 or any(t not in label_map for t in tokens):
        st.error("กรุณาพิมพ์ให้ถูกต้อง เช่น: `B P B` หรือ `P T P`")
    else:
        code_seq = [label_map[t] for t in tokens]
        pred = model.predict([code_seq])[0]
        st.success(f"ระบบคาดการณ์ว่า: **{reverse_map[pred]}**")
