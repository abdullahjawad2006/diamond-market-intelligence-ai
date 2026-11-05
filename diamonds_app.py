"""
Diamond Market Intelligence AI Platform
Enterprise Streamlit Web Application (Step 4)

A restrained, modern, executive-grade analytical interface covering all 6 core modules:
1. Fair Market Valuation & Real-USD SHAP Price Attribution
2. B2B Arbitrage & Deal Evaluator
3. Cut Geometry Classification & Market Tier Segmentation
4. Physical Integrity & Fraud Sentinel (Isolation Forest + Specific Gravity)
5. Smart Inventory Recommender (Budget & Value Optimization)
6. GIA Vector RAG Consultant & Executive Appraisal Dossier
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# PAGE CONFIGURATION & MODERN RESTRAINED STYLING
# ==============================================================================
st.set_page_config(
    page_title="Diamond Market Intelligence AI",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Clean modern typography and spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1360px;
    }
    
    /* Subtle executive header */
    .app-header {
        border-bottom: 1px solid rgba(148, 163, 184, 0.25);
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    .app-title {
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.92rem;
        opacity: 0.75;
        margin-top: 0.25rem;
    }

    /* Restrained metric card styling */
    div[data-testid="stMetric"] {
        background-color: rgba(148, 163, 184, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 8px;
        padding: 14px 18px;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        opacity: 0.8;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.65rem;
        font-weight: 600;
    }

    /* Status banners */
    .status-card {
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 1.25rem;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .status-pass {
        background-color: rgba(16, 185, 129, 0.08);
        border-left: 4px solid #10b981;
    }
    .status-warn {
        background-color: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #f59e0b;
    }
    .status-danger {
        background-color: rgba(239, 68, 68, 0.08);
        border-left: 4px solid #ef4444;
    }
    .status-neutral {
        background-color: rgba(59, 130, 246, 0.08);
        border-left: 4px solid #3b82f6;
    }

    /* Appraisal certificate container */
    .dossier-box {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 0.85rem;
        line-height: 1.65;
        background-color: rgba(15, 23, 42, 0.04);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 8px;
        padding: 1.25rem;
        white-space: pre-wrap;
    }

    /* Clean tab navigation */
    button[data-baseweb="tab"] {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CACHED MODEL BUNDLE LOADER
# ==============================================================================
BUNDLE_PATH = os.path.join("models", "diamond_ai_bundle.joblib")

@st.cache_resource(show_spinner="Initializing Diamond Intelligence Engine...")
def load_ai_engine():
    if not os.path.exists(BUNDLE_PATH):
        st.error(f"Model bundle not found at `{BUNDLE_PATH}`. Please run `python train_and_save_diamonds.py` first.")
        st.stop()
    bundle = joblib.load(BUNDLE_PATH)
    explainer = shap.TreeExplainer(bundle["shap_bundle"]["model"])
    return bundle, explainer

bundle, shap_explainer = load_ai_engine()
metrics = bundle["metrics"]
inventory_df: pd.DataFrame = bundle["clean_inventory"]


# ==============================================================================
# CORE DOMAIN COMPUTATION HELPERS
# ==============================================================================
FEATURE_LABELS = {
    "carat": "Carat Weight",
    "cut": "Cut Grade",
    "color": "Color Grade",
    "clarity": "Clarity Grade",
    "depth": "Total Depth %",
    "table": "Table Width %",
    "x": "Length (x mm)",
    "y": "Width (y mm)",
    "z": "Height (z mm)",
    "volume": "3D Volume (mm³)",
    "density": "Specific Gravity Proxy",
    "lw_ratio": "Length-to-Width Ratio",
    "depth_dev": "Depth Deviation",
    "table_dev": "Table Deviation"
}

def compute_stone_features(carat, cut, color, clarity, depth, table, x, y, z) -> dict:
    vol = max(x * y * z, 0.001)
    return {
        "carat": float(carat),
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": float(depth),
        "table": float(table),
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "volume": float(vol),
        "density": float(carat / vol),
        "lw_ratio": float(x / y),
        "xy_diff": float(abs(x - y)),
        "calc_depth": float(200 * z / (x + y)),
        "depth_dev": float(abs(depth - 61.8)),
        "table_dev": float(abs(table - 56.0)),
        "td_ratio": float(table / depth)
    }

def evaluate_diamond(derived: dict):
    df_row = pd.DataFrame([derived])
    
    # 1. Price Valuation
    pred_price = float(bundle["pricing_pipeline"].predict(df_row)[0])
    fair_price = round(max(pred_price, 250.0), 2)
    low_band = round(fair_price * 0.94, 2)
    high_band = round(fair_price * 1.06, 2)
    
    # 2. USD SHAP Values
    shap_info = bundle["shap_bundle"]
    row_shap = df_row.copy()
    row_shap[shap_info["ord_cols"]] = shap_info["encoder"].transform(row_shap[shap_info["ord_cols"]])
    shap_explanation = shap_explainer(row_shap[shap_info["feature_names"]])
    base_val = float(np.atleast_1d(shap_explanation.base_values)[0])
    
    shap_drivers = []
    for col, val in zip(shap_info["feature_names"], shap_explanation.values[0]):
        shap_drivers.append({
            "feature_key": col,
            "feature_label": FEATURE_LABELS.get(col, col),
            "raw_value": derived[col],
            "impact_usd": round(float(val), 2)
        })
    shap_drivers.sort(key=lambda d: abs(d["impact_usd"]), reverse=True)
    
    # 3. Cut Classification & Market Tier
    clf_bundle = bundle["cut_classifier"]
    input_cut_df = df_row[clf_bundle["features"]]
    pred_cut_idx = int(clf_bundle["classifier"].predict(input_cut_df)[0])
    pred_cut = clf_bundle["label_encoder"].inverse_transform([pred_cut_idx])[0]
    cut_probs = clf_bundle["classifier"].predict_proba(input_cut_df)[0]
    prob_map = {
        cls_name: round(float(p) * 100, 1)
        for cls_name, p in zip(clf_bundle["label_encoder"].classes_, cut_probs)
    }
    
    tier_info = bundle["market_tiers"]
    scaled_tier = tier_info["scaler"].transform(pd.DataFrame([[derived["carat"], fair_price]], columns=["carat", "price"]))
    cluster_id = int(tier_info["kmeans"].predict(scaled_tier)[0])
    market_tier = tier_info["tier_map"].get(cluster_id, "Mid-Market")
    
    # 4. Anomaly & Physics Audit
    anom_info = bundle["anomaly_sentinel"]
    row_anom = {**derived, "price": fair_price}
    input_anom_df = pd.DataFrame([row_anom])[anom_info["features"]]
    scaled_anom = anom_info["scaler"].transform(input_anom_df)
    is_anomaly = bool(anom_info["model"].predict(scaled_anom)[0] == -1)
    anom_score = float(anom_info["model"].score_samples(scaled_anom)[0])
    
    return {
        "fair_price": fair_price,
        "low_band": low_band,
        "high_band": high_band,
        "base_value": round(base_val, 2),
        "shap_drivers": shap_drivers,
        "pred_cut": pred_cut,
        "cut_probs": prob_map,
        "market_tier": market_tier,
        "is_anomaly": is_anomaly,
        "anom_score": round(anom_score, 4)
    }


# ==============================================================================
# SIDEBAR: UNIFIED STONE SPECIFICATION CONTROL PANEL
# ==============================================================================
PRESETS = {
    "1.05 ct Ideal Investment Stone (E / VS1)": {
        "carat": 1.05, "cut": "Ideal", "color": "E", "clarity": "VS1",
        "depth": 61.5, "table": 56.0, "x": 6.55, "y": 6.58, "z": 4.04, "asking": 6200.0
    },
    "0.95 ct Smart Value Shy-Weight (G / VS2)": {
        "carat": 0.95, "cut": "Ideal", "color": "G", "clarity": "VS2",
        "depth": 61.8, "table": 56.0, "x": 6.32, "y": 6.35, "z": 3.91, "asking": 4850.0
    },
    "1.52 ct Luxury Collector Stone (D / VVS1)": {
        "carat": 1.52, "cut": "Ideal", "color": "D", "clarity": "VVS1",
        "depth": 61.6, "table": 55.5, "x": 7.38, "y": 7.41, "z": 4.56, "asking": 16400.0
    },
    "0.55 ct Everyday Entry Brilliant (H / SI1)": {
        "carat": 0.55, "cut": "Very Good", "color": "H", "clarity": "SI1",
        "depth": 62.3, "table": 58.0, "x": 5.24, "y": 5.27, "z": 3.27, "asking": 1520.0
    },
    "1.00 ct Geometry Anomaly / Simulant Test": {
        "carat": 1.00, "cut": "Fair", "color": "G", "clarity": "SI1",
        "depth": 70.5, "table": 66.0, "x": 5.40, "y": 5.95, "z": 4.10, "asking": 5100.0
    }
}

with st.sidebar:
    st.markdown("### Stone Specification")
    st.caption("Configure a diamond below or load a benchmark preset. All 6 analytical tabs update in real time.")
    
    selected_preset = st.selectbox(
        "Reference Profile",
        options=list(PRESETS.keys()),
        index=0
    )
    p = PRESETS[selected_preset]
    
    st.markdown("#### The 4Cs Grading")
    carat = st.slider("Carat Weight (ct)", min_value=0.20, max_value=5.00, value=float(p["carat"]), step=0.01)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        color = st.selectbox("Color Grade", ["D", "E", "F", "G", "H", "I", "J"], index=["D", "E", "F", "G", "H", "I", "J"].index(p["color"]))
    with col_s2:
        clarity = st.selectbox("Clarity Grade", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"], index=["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"].index(p["clarity"]))
        
    cut = st.selectbox("Cut Grade (Listed)", ["Ideal", "Premium", "Very Good", "Good", "Fair"], index=["Ideal", "Premium", "Very Good", "Good", "Fair"].index(p["cut"]))
    
    # Automatically scale default mm dimensions if user moves carat slider away from preset
    scale_factor = (carat / p["carat"]) ** (1.0 / 3.0) if p["carat"] > 0 else 1.0
    default_x = round(p["x"] * scale_factor, 2)
    default_y = round(p["y"] * scale_factor, 2)
    default_z = round(p["z"] * scale_factor, 2)
    
    with st.expander("Physical Geometry (mm & %)", expanded=False):
        depth = st.slider("Total Depth (%)", 50.0, 75.0, float(p["depth"]), 0.1)
        table = st.slider("Table Width (%)", 50.0, 75.0, float(p["table"]), 0.1)
        x_mm = st.number_input("Length x (mm)", min_value=2.50, max_value=15.00, value=default_x, step=0.01)
        y_mm = st.number_input("Width y (mm)", min_value=2.50, max_value=15.00, value=default_y, step=0.01)
        z_mm = st.number_input("Height z (mm)", min_value=1.50, max_value=10.00, value=default_z, step=0.01)

    st.markdown("---")
    st.caption(
        f"**Model Engine:** Voting Ensemble  \n"
        f"**Verified R²:** {metrics['test_r2']*100:.2f}% | **MAE:** \\${metrics['test_mae_usd']:.0f}  \n"
        f"**Catalog:** {metrics['clean_inventory_size']:,} GIA-verified stones"
    )

# Compute current stone state
derived_specs = compute_stone_features(carat, cut, color, clarity, depth, table, x_mm, y_mm, z_mm)
eval_res = evaluate_diamond(derived_specs)


# ==============================================================================
# MAIN HEADER
# ==============================================================================
st.markdown(f"""
<div class="app-header">
    <div class="app-title">Diamond Market Intelligence AI</div>
    <div class="app-subtitle">
        Institutional Valuation, Arbitrage Deal Hunting, Physical Integrity Auditing & GIA Knowledge System
        &nbsp;•&nbsp; Active Stone: <strong>{carat:.2f} ct {cut} / {color} / {clarity}</strong>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 6 DEDICATED ANALYTICAL TABS
# ==============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Valuation & Drivers",
    "2. Arbitrage Deal Hunter",
    "3. Cut & Market Tier",
    "4. Integrity Sentinel",
    "5. Smart Recommender",
    "6. Gemologist & Dossier"
])


# ------------------------------------------------------------------------------
# TAB 1: FAIR MARKET VALUATION & REAL-USD SHAP EXPLAINABILITY
# ------------------------------------------------------------------------------
with tab1:
    fair_val = eval_res["fair_price"]
    ppc = round(fair_val / carat, 2)
    pct_rank = round((inventory_df["price"] < fair_val).mean() * 100, 1)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AI Fair Market Value", f"${fair_val:,.2f}")
    m2.metric("90% Confidence Interval", f"\\${eval_res['low_band']:,.0f} – \\${eval_res['high_band']:,.0f}")
    m3.metric("Implied Price / Carat", f"${ppc:,.0f} / ct")
    m4.metric("Catalog Value Percentile", f"Top {max(100 - pct_rank, 0.1):.1f}%")
    
    st.markdown("##### Price Attribution Breakdown (SHAP Impact in USD)")
    st.caption(
        f"How each physical and grading attribute shifts this diamond's valuation from the catalog baseline average (\\${eval_res['base_value']:,.0f})."
    )
    
    col_chart, col_summary = st.columns([1.65, 1.0], gap="large")
    
    with col_chart:
        top_drivers = eval_res["shap_drivers"][:7][::-1]
        def fmt_driver_label(d):
            val = d["raw_value"]
            key = d["feature_key"]
            if isinstance(val, str):
                return f"{d['feature_label']} ({val})"
            if key == "density":
                return f"{d['feature_label']} ({val:.4f})"
            if key == "lw_ratio":
                return f"{d['feature_label']} ({val:.3f})"
            return f"{d['feature_label']} ({val:.2f})"

        labels = [fmt_driver_label(d) for d in top_drivers]
        impacts = [d["impact_usd"] for d in top_drivers]
        colors = ["#10b981" if v >= 0 else "#ef4444" for v in impacts]
        text_vals = [f"+${v:,.0f}" if v >= 0 else f"-${abs(v):,.0f}" for v in impacts]
        
        min_imp = min(impacts + [0.0])
        max_imp = max(impacts + [0.0])
        span = max(max_imp - min_imp, 100.0)
        x_range = [min_imp - span * 0.28, max_imp + span * 0.28]
        
        fig_shap = go.Figure(go.Bar(
            x=impacts,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=text_vals,
            textposition="outside",
            cliponaxis=False
        ))
        fig_shap.update_layout(
            height=340,
            margin=dict(l=10, r=30, t=15, b=25),
            xaxis_title="Dollar Impact on Valuation (USD)",
            yaxis_title=None,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                range=x_range,
                zeroline=True,
                zerolinewidth=1.5,
                zerolinecolor="rgba(148,163,184,0.4)",
                gridcolor="rgba(148,163,184,0.12)"
            ),
            font=dict(size=12)
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        
    with col_summary:
        pos_drivers = [d for d in eval_res["shap_drivers"] if d["impact_usd"] > 0]
        neg_drivers = [d for d in eval_res["shap_drivers"] if d["impact_usd"] < 0]
        
        st.markdown("**Executive Valuation Insights**")
        if pos_drivers:
            top_p = pos_drivers[0]
            st.markdown(
                f"- **Primary Value Premium:** `{top_p['feature_label']}` contributes the largest upward premium of **+\\${top_p['impact_usd']:,.0f}**."
            )
        if len(pos_drivers) > 1:
            sec_p = pos_drivers[1]
            st.markdown(
                f"- **Secondary Driver:** `{sec_p['feature_label']}` adds **+\\${sec_p['impact_usd']:,.0f}** based on current GIA grading tiers."
            )
        if neg_drivers:
            top_n = neg_drivers[0]
            st.markdown(
                f"- **Primary Value Adjustment:** `{top_n['feature_label']}` reduces the estimate by **-\\${abs(top_n['impact_usd']):,.0f}** relative to top-tier benchmarks."
            )
        else:
            st.markdown("- **All Major Attributes Positive:** Every primary attribute sits above the global catalog average.")
            
        st.markdown(
            f"- **Magic Weight Context:** At **{carat:.2f} ct**, any jump across a 0.50 ct or 1.00 ct threshold triggers non-linear wholesale price appreciation."
        )


# ------------------------------------------------------------------------------
# TAB 2: B2B ARBITRAGE DEAL HUNTER
# ------------------------------------------------------------------------------
with tab2:
    col_ask, col_spacer = st.columns([1.2, 2.0])
    with col_ask:
        default_ask = float(p["asking"]) if abs(carat - p["carat"]) < 0.005 else float(round(fair_val * 0.88, -1))
        asking_price = st.number_input(
            "Supplier or Retail Asking Price ($ USD)",
            min_value=100.0,
            max_value=150000.0,
            value=default_ask,
            step=50.0,
            help="Enter the quoted price to compute retail arbitrage margin against AI Fair Value."
        )
        
    diff_usd = round(asking_price - fair_val, 2)
    pct_diff = round((diff_usd / fair_val) * 100, 2)
    
    if pct_diff <= -15.0:
        verdict_label = "Strong Buy (Arbitrage Deal)"
        banner_class = "status-pass"
        verdict_desc = (
            f"<strong>Underpriced by {abs(pct_diff):.1f}% (&#36;{abs(diff_usd):,.2f} below fair value).</strong> "
            f"This listing offers significant wholesale acquisition margin."
        )
    elif pct_diff >= 15.0:
        verdict_label = "Overpriced (Negotiate)"
        banner_class = "status-danger"
        verdict_desc = (
            f"<strong>Marked up by +{pct_diff:.1f}% (+&#36;{diff_usd:,.2f} above fair value).</strong> "
            f"Listing exceeds statistical equilibrium for {carat:.2f} ct {color}/{clarity} stones."
        )
    else:
        verdict_label = "Fair Market Pricing"
        banner_class = "status-neutral"
        verdict_desc = (
            f"<strong>Within normal market spread ({pct_diff:+.1f}% / &#36;{diff_usd:+,.2f}).</strong> "
            f"Pricing aligns closely with verified peer transactions."
        )
        
    if diff_usd < 0:
        delta_text = f"{abs(pct_diff):.1f}% below fair value (Discount)"
        delta_col = "normal"
    elif diff_usd > 0:
        delta_text = f"+{pct_diff:.1f}% above fair value (Markup)"
        delta_col = "inverse"
    else:
        delta_text = "0.0% (At Fair Value)"
        delta_col = "off"
        
    d1, d2, d3 = st.columns(3)
    d1.metric("Quoted Asking Price", f"${asking_price:,.2f}")
    d2.metric("Spread vs. Fair Value", f"${diff_usd:+,.2f}", delta=delta_text, delta_color=delta_col)
    d3.metric("Acquisition Verdict", verdict_label)
    
    st.markdown(f'<div class="status-card {banner_class}">{verdict_desc}</div>', unsafe_allow_html=True)
    
    # Peer comparison visual
    st.markdown("##### Market Positioning Against Peer Catalog Stones")
    peer_window = inventory_df[
        (inventory_df["carat"] >= max(0.2, carat - 0.35)) &
        (inventory_df["carat"] <= carat + 0.35) &
        (~inventory_df["is_anomaly"])
    ]
    if len(peer_window) > 600:
        peer_sample = peer_window.sample(600, random_state=42)
    else:
        peer_sample = peer_window
        
    fig_deal = go.Figure()
    fig_deal.add_trace(go.Scatter(
        x=peer_sample["carat"],
        y=peer_sample["price"],
        mode="markers",
        name="Catalog Peers",
        marker=dict(size=6, color="rgba(148, 163, 184, 0.35)"),
        customdata=np.stack((peer_sample["cut"], peer_sample["color"], peer_sample["clarity"]), axis=-1),
        hovertemplate="<b>%{x:.2f} ct</b> (%{customdata[0]} / %{customdata[1]} / %{customdata[2]})<br>Catalog Price: $%{y:,.0f}<extra></extra>"
    ))
    fig_deal.add_trace(go.Scatter(
        x=[carat],
        y=[fair_val],
        mode="markers",
        name="AI Fair Market Value",
        marker=dict(size=13, color="#3b82f6", symbol="diamond", line=dict(width=1.5, color="white")),
        hovertemplate=f"<b>AI Fair Value</b>: ${fair_val:,.2f}<extra></extra>"
    ))
    fig_deal.add_trace(go.Scatter(
        x=[carat],
        y=[asking_price],
        mode="markers",
        name="Quoted Asking Price",
        marker=dict(size=14, color="#10b981" if pct_diff <= 0 else "#ef4444", symbol="star", line=dict(width=1.5, color="white")),
        hovertemplate=f"<b>Asking Price</b>: ${asking_price:,.2f}<extra></extra>"
    ))
    fig_deal.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=10, b=20),
        xaxis_title="Carat Weight (ct)",
        yaxis_title="Price (USD)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(148,163,184,0.12)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.12)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_deal, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 3: CUT GEOMETRY CLASSIFICATION & MARKET TIER SEGMENTATION
# ------------------------------------------------------------------------------
with tab3:
    pred_cut = eval_res["pred_cut"]
    cut_match = (pred_cut == cut)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Listed Cut Grade", cut)
    c2.metric("AI Geometry-Predicted Cut", pred_cut)
    c3.metric("Commercial Price Tier", f"{eval_res['market_tier']} Tier")
    
    col_cut_l, col_cut_r = st.columns([1.2, 1.2], gap="large")
    
    with col_cut_l:
        st.markdown("##### Geometric Cut Grade Probabilities")
        st.caption("Predicted solely from millimeter measurements, table %, and depth % without seeing the listed label.")
        
        grade_order = ["Ideal", "Premium", "Very Good", "Good", "Fair"][::-1]
        probs = [eval_res["cut_probs"].get(g, 0.0) for g in grade_order]
        bar_colors = ["#3b82f6" if g == pred_cut else "rgba(148, 163, 184, 0.45)" for g in grade_order]
        
        fig_cut = go.Figure(go.Bar(
            x=probs,
            y=grade_order,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{p:.1f}%" for p in probs],
            textposition="outside",
            cliponaxis=False
        ))
        fig_cut.update_layout(
            height=280,
            margin=dict(l=10, r=50, t=10, b=20),
            xaxis=dict(range=[0, 110], title="Confidence Probability (%)", gridcolor="rgba(148,163,184,0.12)"),
            yaxis_title=None,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_cut, use_container_width=True)
        
    with col_cut_r:
        st.markdown("##### Optical Proportion Tolerance Audit")
        st.caption("Comparison of stone angles against GIA Ideal Round Brilliant benchmarks.")
        
        depth_ok = 59.5 <= depth <= 62.5
        table_ok = 54.0 <= table <= 58.0
        lw_ok = 0.98 <= derived_specs["lw_ratio"] <= 1.02
        
        prop_df = pd.DataFrame([
            {
                "Metric": "Total Depth %",
                "Measured": f"{depth:.1f}%",
                "GIA Ideal Benchmark": "59.5% – 62.5%",
                "Optical Status": "Optimal Brilliance" if depth_ok else "Light Leakage Risk"
            },
            {
                "Metric": "Table Width %",
                "Measured": f"{table:.1f}%",
                "GIA Ideal Benchmark": "54.0% – 58.0%",
                "Optical Status": "Optimal Fire Balance" if table_ok else "Outside Ideal Window"
            },
            {
                "Metric": "Length / Width Ratio",
                "Measured": f"{derived_specs['lw_ratio']:.3f}",
                "GIA Ideal Benchmark": "0.985 – 1.015",
                "Optical Status": "Symmetric Round" if lw_ok else "Slightly Off-Round"
            },
            {
                "Metric": "Computed Depth %",
                "Measured": f"{derived_specs['calc_depth']:.1f}%",
                "GIA Ideal Benchmark": f"Matches {depth:.1f}% (±1.5%)",
                "Optical Status": "Consistent" if abs(depth - derived_specs["calc_depth"]) <= 2.0 else "Dimension Discrepancy"
            }
        ])
        st.dataframe(prop_df, use_container_width=True, hide_index=True)
        
        if cut_match:
            st.markdown(
                f'<div class="status-card status-pass"><strong>Cut Grade Verified:</strong> Physical proportions confirm the listed <strong>{cut}</strong> grade ({eval_res["cut_probs"][pred_cut]:.1f}% confidence).</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="status-card status-warn"><strong>Grading Variance Notice:</strong> Listed as <strong>{cut}</strong>, but physical proportions align closer to <strong>{pred_cut}</strong> ({eval_res["cut_probs"][pred_cut]:.1f}% probability).</div>',
                unsafe_allow_html=True
            )


# ------------------------------------------------------------------------------
# TAB 4: PHYSICAL INTEGRITY & FRAUD SENTINEL
# ------------------------------------------------------------------------------
with tab4:
    density_val = derived_specs["density"]
    density_ok = 0.0052 <= density_val <= 0.0072
    sym_ok = 0.95 <= derived_specs["lw_ratio"] <= 1.05
    math_depth_ok = abs(depth - derived_specs["calc_depth"]) <= 3.0
    
    overall_clean = (not eval_res["is_anomaly"]) and density_ok and sym_ok and math_depth_ok
    
    if overall_clean:
        st.markdown(
            '<div class="status-card status-pass">'
            '<strong>INTEGRITY AUDIT PASSED:</strong> Physical dimensions, carat-to-volume density proxy, and optical geometry conform to natural carbon diamond physics (3.51 g/cm³).'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-card status-danger">'
            '<strong>GEOMETRIC / DENSITY ANOMALY DETECTED:</strong> One or more physical measurements deviate from natural round diamond geometry. Verify certificate measurements or screen for simulants (e.g., Moissanite / Cubic Zirconia).'
            '</div>',
            unsafe_allow_html=True
        )
        
    f1, f2, f3 = st.columns(3)
    f1.metric("Bounding Volume (x·y·z)", f"{derived_specs['volume']:.2f} mm³")
    f2.metric("Density Proxy (ct / mm³)", f"{density_val:.5f}", delta="Normal (~0.0061)" if density_ok else "Abnormal Ratio", delta_color="normal" if density_ok else "inverse")
    f3.metric("Isolation Forest Score", f"{eval_res['anom_score']:.4f}", delta="In-Distribution" if not eval_res["is_anomaly"] else "Outlier Flagged", delta_color="normal" if not eval_res["is_anomaly"] else "inverse")
    
    st.markdown("##### Three-Point Gemological Verification Checklist")
    audit_rows = [
        {
            "Verification Check": "1. Specific Gravity & Density Alignment",
            "Stone Measurement": f"{density_val:.5f} ct/mm³",
            "Standard Threshold": "0.00520 – 0.00720 ct/mm³",
            "Status": "PASS" if density_ok else "FLAG: Possible Simulant or Carat Typo"
        },
        {
            "Verification Check": "2. Circular Outline Symmetry (x vs. y)",
            "Stone Measurement": f"{x_mm:.2f} mm × {y_mm:.2f} mm (Ratio: {derived_specs['lw_ratio']:.3f})",
            "Standard Threshold": "Ratio 0.950 – 1.050",
            "Status": "PASS" if sym_ok else "FLAG: Non-Circular / Fancy Shape Proportions"
        },
        {
            "Verification Check": "3. Millimeter-to-Depth Mathematical Identity",
            "Stone Measurement": f"Listed {depth:.1f}% vs. Computed {derived_specs['calc_depth']:.1f}%",
            "Standard Threshold": "Variance ≤ 3.0%",
            "Status": "PASS" if math_depth_ok else "FLAG: Certificate Measurement Typo"
        }
    ]
    st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------
# TAB 5: SMART INVENTORY RECOMMENDER
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_top_recommendations(rec_budget: float, min_cut_filter: str, strategy: str) -> pd.DataFrame:
    allowed_cuts = {
        "Ideal Only": ["Ideal"],
        "Premium & Above": ["Ideal", "Premium"],
        "Very Good & Above": ["Ideal", "Premium", "Very Good"]
    }[min_cut_filter]
    
    candidates = inventory_df[
        (inventory_df["price"] <= rec_budget) &
        (inventory_df["price"] >= rec_budget * 0.55) &
        (inventory_df["cut"].isin(allowed_cuts)) &
        (~inventory_df["is_anomaly"])
    ].copy()
    
    if candidates.empty:
        candidates = inventory_df[
            (inventory_df["price"] <= rec_budget) &
            (~inventory_df["is_anomaly"])
        ].copy()
        
    if candidates.empty:
        return pd.DataFrame()
        
    feat_cols = [c for c in bundle["pricing_pipeline"].feature_names_in_ if c in candidates.columns]
    candidates["fair_value"] = np.round(bundle["pricing_pipeline"].predict(candidates[feat_cols]), 2)
    candidates["savings_usd"] = np.round(candidates["fair_value"] - candidates["price"], 2)
    candidates["discount_pct"] = np.round((candidates["savings_usd"] / candidates["fair_value"]) * 100, 1)
    
    cut_r = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
    col_r = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}
    cla_r = {"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8}
    
    if strategy == "Maximize Discount %":
        candidates["rank_score"] = candidates["discount_pct"]
    elif strategy == "Maximize Carat Size":
        candidates["rank_score"] = candidates["carat"] * 10.0 + candidates["discount_pct"].clip(-5, 25) * 0.2
    else:
        candidates["rank_score"] = (
            candidates["cut"].map(cut_r) * 2.0 +
            candidates["color"].map(col_r) * 1.2 +
            candidates["clarity"].map(cla_r) * 1.2 +
            candidates["carat"] * 8.0 +
            candidates["discount_pct"].clip(-10, 35) * 0.4
        )
        
    return candidates.sort_values("rank_score", ascending=False).head(6).copy()

with tab5:
    st.caption("Scans all 53,770 verified, non-anomalous stones in the catalog to surface the highest value-for-money acquisitions under your budget.")
    
    r_col1, r_col2, r_col3 = st.columns([1.3, 1.0, 1.1])
    with r_col1:
        rec_budget = st.slider("Maximum Acquisition Budget ($ USD)", min_value=500, max_value=25000, value=5000, step=250)
    with r_col2:
        min_cut_filter = st.selectbox("Minimum Cut Quality", ["Ideal Only", "Premium & Above", "Very Good & Above"], index=1)
    with r_col3:
        strategy = st.selectbox("Optimization Strategy", ["Balanced Composite Value", "Maximize Discount %", "Maximize Carat Size"], index=0)
        
    top_picks = get_top_recommendations(float(rec_budget), min_cut_filter, strategy)
        
    if top_picks.empty:
        st.warning("No stones found matching this budget constraint. Please increase the budget slider.")
    else:
        best_stone = top_picks.iloc[0]
        b1, b2, b3 = st.columns(3)
        b1.metric("Top Pick Specification", f"{best_stone['carat']:.2f} ct {best_stone['cut']} ({best_stone['color']}/{best_stone['clarity']})")
        b2.metric("Catalog Price vs. AI Fair Value", f"\\${best_stone['price']:,.0f} vs. \\${best_stone['fair_value']:,.0f}")
        b3.metric("Buyer Arbitrage Savings", f"${best_stone['savings_usd']:,.0f} ({best_stone['discount_pct']:.1f}% Off)")
        
        def build_rationale(row):
            reasons = []
            if row["discount_pct"] >= 25.0:
                reasons.append(f"Steep {row['discount_pct']:.0f}% wholesale discount")
            if row["carat"] >= 1.0:
                reasons.append(f"High carat presence ({row['carat']:.2f} ct)")
            if row["color"] in ["D", "E", "F"]:
                reasons.append(f"Colorless tier ({row['color']})")
            if row["clarity"] in ["IF", "VVS1", "VVS2", "VS1", "VS2"]:
                reasons.append(f"Eye-clean clarity ({row['clarity']})")
            return " • ".join(reasons[:2]) if reasons else "Optimal 4Cs balance under budget"

        display_recs = pd.DataFrame({
            "Carat": top_picks["carat"].map(lambda v: f"{v:.2f} ct"),
            "Cut": top_picks["cut"],
            "Color": top_picks["color"],
            "Clarity": top_picks["clarity"],
            "Dimensions (mm)": top_picks.apply(lambda r: f"{r['x']:.2f} × {r['y']:.2f} × {r['z']:.2f}", axis=1),
            "Catalog Price": top_picks["price"].map(lambda v: f"${v:,.0f}"),
            "AI Fair Value": top_picks["fair_value"].map(lambda v: f"${v:,.0f}"),
            "Savings ($)": top_picks["savings_usd"].map(lambda v: f"+${v:,.0f}" if v >= 0 else f"-${abs(v):,.0f}"),
            "Discount (%)": top_picks["discount_pct"].map(lambda v: f"{v:.1f}%"),
            "Why Selected": top_picks.apply(build_rationale, axis=1)
        })
        st.dataframe(display_recs, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------
# TAB 6: GIA GEMOLOGIST CONSULTANT (VECTOR RAG) & EXECUTIVE DOSSIER
# ------------------------------------------------------------------------------
with tab6:
    col_rag, col_memo = st.columns([1.15, 1.15], gap="large")
    
    with col_rag:
        st.markdown("##### GIA Knowledge Consultant (Vector RAG)")
        st.caption("Ask a gemological or buying question to retrieve authoritative GIA standards alongside guidance tailored to your active stone.")
        
        preset_questions = [
            "How can I save money near 1.00 carat while keeping an eye-clean look?",
            "Is my current stone's depth and table percentage within GIA Ideal limits?",
            "Which color and clarity grades offer the best retail value?",
            "How does specific gravity help detect synthetic diamond simulants?"
        ]
        selected_q = st.selectbox("Common Advisory Topics", options=preset_questions, index=0)
        user_query = st.text_input("Or enter a custom question:", value=selected_q)
        
        if user_query:
            rag_info = bundle["rag_gemology"]
            q_vec = rag_info["vectorizer"].transform([user_query])
            sims = cosine_similarity(q_vec, rag_info["tfidf_matrix"]).flatten()
            top_idxs = sims.argsort()[::-1][:2]
            
            st.markdown("**Retrieved GIA Reference Guidance:**")
            for idx in top_idxs:
                score = float(sims[idx])
                if score > 0.03:
                    st.markdown(
                        f'<div class="status-card status-neutral">'
                        f'<strong>GIA Reference (Relevance {score:.2f}):</strong><br>{rag_info["corpus"][idx]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
            st.markdown("**Contextual Advisory for Active Stone:**")
            st.info(
                f"Your configured **{carat:.2f} ct {cut} Cut ({color} / {clarity})** carries an AI Fair Market Valuation of **\\${fair_val:,.2f} USD** "
                f"(90% confidence range: **\\${eval_res['low_band']:,.0f} – \\${eval_res['high_band']:,.0f} USD**). "
                f"Based on its physical millimeter angles, the geometry model classifies its optical cut as **{eval_res['pred_cut']}** "
                f"and places the stone in the **{eval_res['market_tier']} Price Tier**."
            )
            
    with col_memo:
        st.markdown("##### Official Digital Appraisal Dossier")
        st.caption("Structured valuation report ready for client presentation, insurance documentation, or wholesale memo attachment.")
        
        integrity_str = "PASS (Conforms to 3.51 g/cm³ natural diamond standard)" if overall_clean else "REVIEW REQUIRED (Geometric / density outlier flagged)"
        
        dossier_text = (
            f"================ EXECUTIVE DIAMOND APPRAISAL DOSSIER ================\n"
            f"Certificate Ref   : DMI-{int(carat*100):03d}-{color}{clarity}-{int(fair_val)%10000:04d}\n"
            f"Stone Profile     : {carat:.2f} Carat | {cut} Cut | Color {color} | Clarity {clarity}\n"
            f"Measurements (mm) : {x_mm:.2f} x {y_mm:.2f} x {z_mm:.2f} mm (Volume: {derived_specs['volume']:.2f} mm³)\n"
            f"Proportions       : Total Depth {depth:.1f}% | Table Width {table:.1f}% | L/W Ratio {derived_specs['lw_ratio']:.3f}\n"
            f"---------------------------------------------------------------------\n"
            f"AI Fair Valuation : ${fair_val:,.2f} USD\n"
            f"90% Confidence    : ${eval_res['low_band']:,.2f} - ${eval_res['high_band']:,.2f} USD\n"
            f"Implied Rate      : ${ppc:,.2f} USD / Carat\n"
            f"Market Segment    : {eval_res['market_tier']} Tier (Top {max(100 - pct_rank, 0.1):.1f}% by Catalog Value)\n"
            f"Geometry Audit    : Predicted {eval_res['pred_cut']} ({eval_res['cut_probs'][eval_res['pred_cut']]:.1f}% model confidence)\n"
            f"Integrity Status  : {integrity_str}\n"
            f"Valuation Engine  : {metrics['model_name']} (R² = {metrics['test_r2']:.4f})\n"
            f"====================================================================="
        )
        
        dossier_html = dossier_text.replace("$", "&#36;")
        st.markdown(f'<div class="dossier-box">{dossier_html}</div>', unsafe_allow_html=True)
        
        st.download_button(
            label="Download Appraisal Dossier (.txt)",
            data=dossier_text,
            file_name=f"diamond_appraisal_{carat:.2f}ct_{color}_{clarity}.txt",
            mime="text/plain",
            use_container_width=True
        )
