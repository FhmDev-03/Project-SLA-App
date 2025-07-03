import shap
import streamlit.components.v1 as components
import streamlit as st
import joblib, json, pandas as pd

@st.cache_resource
def load_artifacts():
    model = joblib.load("rf_sla_model.joblib")
    feature_order = json.load(open("rf_feature_order.json"))
    return model, feature_order

model, feature_order = load_artifacts()

st.set_page_config(page_title="SLA Breach Predictor", page_icon="⏰")
st.title("🕒 SLA Breach Risk Predictor")
st.write("Enter ticket details to estimate the probability of breaching its SLA.")

col1, col2 = st.columns(2)

with col1:
    reassignment_count = st.slider("Reassignment Count", 0, 10, 0)
    reopen_count       = st.slider("Reopen Count",       0,  5, 0)
    sys_mod_count      = st.slider("Sys‑mod Count",      0, 20, 0)
    priority_num       = st.selectbox("Priority (1 = Critical … 4 = Low)", [1, 2, 3, 4], index=2)

with col2:
    resolution_time_hrs = st.number_input("Resolution Time (hrs)", 0.0, 1000.0, 5.0, 0.5)
    sla_threshold       = st.number_input("SLA Threshold (hrs)",   1.0, 100.0, 24.0, 0.5)
    created_hour        = st.slider("Created Hour (0‑23)", 0, 23, 9)
    created_dayofweek   = st.slider("Day of Week (0=Mon … 6=Sun)", 0, 6, 2)
    is_weekend          = st.radio("Is Weekend?", [0, 1], horizontal=True)

if st.button("Predict SLA Breach"):
    new_ticket = pd.DataFrame([{
        "reassignment_count": reassignment_count,
        "reopen_count": reopen_count,
        "sys_mod_count": sys_mod_count,
        "priority_num": priority_num,
        "resolution_time_hrs": resolution_time_hrs,
        "sla_threshold": sla_threshold,
        "created_hour": created_hour,
        "created_dayofweek": created_dayofweek,
        "is_weekend": is_weekend
    }])[feature_order]

    prob = model.predict_proba(new_ticket)[0, 1]
    pred = model.predict(new_ticket)[0]

    st.metric("Breach Probability", f"{prob:.1%}")
    if pred == 1:
        st.error("⚠️ High risk ‑ SLA likely to be breached")
    else:
        st.success("✅ Low risk ‑ SLA likely to be met")
        explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(new_ticket)

    force_plot = shap.force_plot(
        explainer.expected_value[1], 
        shap_values[1][0], 
        new_ticket,
        matplotlib=False
    )

    shap_html_path = "shap_force_plot.html"
    shap.save_html(shap_html_path, force_plot)

    st.markdown("---")
    st.subheader("🔍 SHAP Explanation")
    components.html(open(shap_html_path, "r", encoding="utf-8").read(), height=400, scrolling=True)

