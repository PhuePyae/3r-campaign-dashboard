import json
import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(
    page_title="3R Campaign Dashboard",
    page_icon="♻️",
    layout="wide",
)

DB_PATH = "campaign_database.db"
TABLE_NAME = "campaign_table"

# Sample Dataset (30 Campaigns: 2023–2026)
SAMPLE_DATA = [
    ("Reusable Bag Pledge Week", 2023, 70000, 2800, 190, 1200000.0, 2900000.0, 3400.0),
    ("Cafe Tumbler Discount Program", 2023, 90000, 4100, 330, 1600000.0, 5100000.0, 9800.0),
    ("Apartment Recycling Sorting Guide", 2023, 65000, 2200, 150, 1000000.0, 1300000.0, 6200.0),
    ("Food Waste Awareness Poster Drive", 2023, 110000, 3300, 210, 2100000.0, 3800000.0, 14500.0),
    ("Used Book Exchange Fair", 2023, 45000, 2000, 240, 900000.0, 3300000.0, 2700.0),
    ("Plastic-Free Market Day", 2023, 80000, 3600, 260, 1700000.0, 3100000.0, 4900.0),
    ("Zero-Waste Cafe Cup Return", 2024, 120000, 6500, 520, 2500000.0, 9800000.0, 18500.0),
    ("Reusable Shopping Bag Drive", 2024, 95000, 4200, 310, 1800000.0, 4900000.0, 6200.0),
    ("Neighborhood Compost Starter Kits", 2024, 85000, 3900, 300, 2300000.0, 4200000.0, 21000.0),
    ("Refillable Detergent Stations", 2024, 70000, 3000, 270, 1900000.0, 5600000.0, 5400.0),
    ("Old Phone Take-Back Campaign", 2024, 55000, 1700, 80, 1800000.0, 1400000.0, 1900.0),
    ("Upcycled Furniture Workshop", 2024, 40000, 2100, 190, 1400000.0, 3900000.0, 4300.0),
    ("University Dorm Recycling Challenge", 2024, 100000, 5000, 420, 2000000.0, 5800000.0, 13700.0),
    ("Secondhand Kids Clothes Exchange", 2024, 60000, 2600, 230, 1100000.0, 3700000.0, 3100.0),
    ("Food Waste Reduction Challenge", 2025, 150000, 7800, 640, 3200000.0, 8600000.0, 42000.0),
    ("Second-Hand Clothing Swap", 2025, 80000, 3100, 280, 1500000.0, 5200000.0, 9800.0),
    ("Zero-Waste Restaurant Certification", 2025, 130000, 5600, 410, 3000000.0, 7300000.0, 36500.0),
    ("Bike Repair and Reuse Days", 2025, 50000, 1900, 170, 1000000.0, 2600000.0, 2400.0),
    ("PET Bottle Deposit Awareness", 2025, 160000, 8200, 700, 3400000.0, 9100000.0, 28000.0),
    ("E-Waste Drop-Off Boxes in Apartments", 2025, 75000, 2500, 160, 2200000.0, 3000000.0, 11800.0),
    ("Paper Cup Free Office Month", 2025, 90000, 3700, 280, 1500000.0, 3600000.0, 7100.0),
    ("Green Market Packaging Return", 2025, 105000, 3500, 150, 2600000.0, 2300000.0, 8900.0),
    ("E-Waste Recycling Week", 2026, 60000, 1900, 90, 2000000.0, 1700000.0, 7500.0),
    ("School Recycling Contest", 2026, 110000, 5400, 450, 2200000.0, 6300000.0, 15300.0),
    ("Smart Waste Sorting App Launch", 2026, 180000, 9000, 810, 4000000.0, 10400000.0, 31000.0),
    ("Reusable Container Takeout Program", 2026, 125000, 6000, 520, 2800000.0, 8200000.0, 17600.0),
    ("Textile Recycling Bins Campaign", 2026, 95000, 3400, 240, 2400000.0, 3800000.0, 19500.0),
    ("Community Repair Cafe", 2026, 45000, 1800, 150, 1000000.0, 2400000.0, 2100.0),
    ("Tumbler Return Kiosk Pilot", 2026, 70000, 2300, 100, 2100000.0, 1600000.0, 3300.0),
    ("Food Waste Bin RFID Awareness", 2026, 140000, 6200, 480, 2900000.0, 6900000.0, 38400.0),
]


def safe_divide(numerator, denominator):
    return np.where(denominator > 0, numerator / denominator, 0.0)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign TEXT NOT NULL,
            year INTEGER NOT NULL,
            reach INTEGER NOT NULL,
            clicks INTEGER NOT NULL,
            conversions INTEGER NOT NULL,
            spend REAL NOT NULL,
            revenue REAL NOT NULL,
            waste_kg REAL NOT NULL
        )
        """
    )
    cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
    if cur.fetchone()[0] == 0:
        cur.executemany(
            f"INSERT INTO {TABLE_NAME} (campaign, year, reach, clicks, conversions, spend, revenue, waste_kg) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            SAMPLE_DATA,
        )
    conn.commit()
    conn.close()


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    if not df.empty:
        df["roas"] = safe_divide(df["revenue"], df["spend"])
        df["conversion_rate"] = safe_divide(df["conversions"], df["clicks"]) * 100
        df["ctr"] = safe_divide(df["clicks"], df["reach"]) * 100
        df["cpa"] = safe_divide(df["spend"], df["conversions"])
        df["profit"] = df["revenue"] - df["spend"]
        df["kg_per_million"] = safe_divide(df["waste_kg"], df["spend"] / 1_000_000)
    return df


def add_campaign(campaign, year, reach, clicks, conversions, spend, revenue, waste_kg):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        f"INSERT INTO {TABLE_NAME} (campaign, year, reach, clicks, conversions, spend, revenue, waste_kg) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (campaign, year, reach, clicks, conversions, spend, revenue, waste_kg),
    )
    conn.commit()
    conn.close()


def update_campaign(row_id, campaign, year, reach, clicks, conversions, spend, revenue, waste_kg):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        f"""
        UPDATE {TABLE_NAME}
        SET campaign=?, year=?, reach=?, clicks=?, conversions=?, spend=?, revenue=?, waste_kg=?
        WHERE id=?
        """,
        (campaign, year, reach, clicks, conversions, spend, revenue, waste_kg, row_id),
    )
    conn.commit()
    conn.close()


def delete_campaign(row_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"DELETE FROM {TABLE_NAME} WHERE id=?", (row_id,))
    conn.commit()
    conn.close()


def reload_missing_samples():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"SELECT campaign, year FROM {TABLE_NAME}")
    existing = set(cur.fetchall())
    to_insert = [row for row in SAMPLE_DATA if (row[0], row[1]) not in existing]
    if to_insert:
        cur.executemany(
            f"INSERT INTO {TABLE_NAME} (campaign, year, reach, clicks, conversions, spend, revenue, waste_kg) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            to_insert,
        )
        conn.commit()
    conn.close()
    return len(to_insert)


@st.cache_resource
def train_roas_model(row_count):
    if row_count < 3:
        return None, None
    conn = sqlite3.connect(DB_PATH)
    df_ml = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    if len(df_ml) < 3:
        return None, None
    df_ml["roas"] = safe_divide(df_ml["revenue"], df_ml["spend"])
    X = df_ml[["spend", "clicks", "reach"]]
    y = df_ml["revenue"]
    ml_model = LinearRegression()
    ml_model.fit(X, y)
    return ml_model, df_ml


# Initialize Database
init_db()
df_all = load_data()

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📁 Upload Dataset")
    uploaded_file = st.file_uploader("Upload Campaign CSV or JSON", type=["csv", "json"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                uploaded_df = pd.read_csv(uploaded_file)
            else:
                uploaded_df = pd.read_json(uploaded_file)

            col_map = {
                "Campaign": "campaign",
                "campaign_name": "campaign",
                "Year": "year",
                "Reach": "reach",
                "Clicks": "clicks",
                "Conversions": "conversions",
                "Spend": "spend",
                "spend_krw": "spend",
                "Revenue": "revenue",
                "revenue_krw": "revenue",
                "Waste": "waste_kg",
                "waste_avoided": "waste_kg",
            }
            uploaded_df.rename(columns=col_map, inplace=True)
            required_cols = ["campaign", "year", "reach", "clicks", "conversions", "spend", "revenue", "waste_kg"]
            missing_cols = [c for c in required_cols if c not in uploaded_df.columns]
            if missing_cols:
                st.error(f"Missing columns in uploaded file: {', '.join(missing_cols)}")
            else:
                clean_df = uploaded_df[required_cols].dropna()
                conn = sqlite3.connect(DB_PATH)
                clean_df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
                conn.close()
                st.success(f"Successfully imported {len(clean_df)} campaigns!")
                st.cache_resource.clear()
                st.rerun()
        except Exception as e:
            st.error(f"Error reading file: {e}")

    st.header("📅 Year View")
    years = ["All years"] + sorted(df_all["year"].unique().tolist()) if not df_all.empty else ["All years"]
    selected_year = st.selectbox("Show campaigns from", years)

    st.markdown("---")
    st.header("➕ Add New Campaign")
    with st.form("sidebar_add_form", clear_on_submit=True):
        s_name = st.text_input("Campaign Name", placeholder="e.g. Campus Eco Drive")
        s_year = st.number_input("Year", min_value=2020, max_value=2035, value=2026, step=1)
        s_reach = st.number_input("Reach", min_value=0, value=100000, step=1000)
        s_clicks = st.number_input("Clicks", min_value=0, value=5000, step=100)
        s_conv = st.number_input("Conversions", min_value=0, value=400, step=10)
        s_spend = st.number_input("Spend (KRW)", min_value=0.0, value=2000000.0, step=100000.0)
        s_rev = st.number_input("Revenue (KRW)", min_value=0.0, value=6000000.0, step=100000.0)
        s_waste = st.number_input("Waste Avoided (kg)", min_value=0.0, value=5000.0, step=100.0)
        if st.form_submit_button("Save Campaign", use_container_width=True):
            if s_name.strip():
                add_campaign(s_name.strip(), s_year, s_reach, s_clicks, s_conv, s_spend, s_rev, s_waste)
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error("Please enter a campaign name.")

    with st.expander("🧪 Sample data"):
        st.caption("Adds the 30 sample campaigns not yet in your database.")
        if st.button("Load sample campaigns", use_container_width=True):
            added = reload_missing_samples()
            st.cache_resource.clear()
            st.success(f"Added {added} sample campaign(s).")
            st.rerun()

# Filter Dataframe
df = df_all.copy() if selected_year == "All years" else df_all[df_all["year"] == selected_year].copy()

# ================= MAIN DASHBOARD =================
st.title("♻️ Reduce • Reuse • Recycle — Campaign Performance Dashboard")
st.caption("3R project dashboard | Data stored in SQLite (`campaign_database.db`) | Powered by Pandas, Plotly & Scikit-Learn")

if df.empty:
    st.warning("No campaigns found for the selected filter. Add one in the sidebar or load sample data.")
    st.stop()

st.info(f"📅 Showing: **{selected_year}** ({len(df)} campaign(s))")

# 1. KPI Metrics with full numeric tooltips on Hover
st.subheader("📊 Key Performance Indicators")
total_spend = df["spend"].sum()
total_rev = df["revenue"].sum()
overall_roas = total_rev / total_spend if total_spend > 0 else 0
avg_conv_rate = df["conversion_rate"].mean()
total_profit = total_rev - total_spend
total_waste = df["waste_kg"].sum()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Spend", f"₩{total_spend / 1e6:,.2f}M", help=f"Exact value: ₩{total_spend:,.0f} (South Korean Won)")
k2.metric("Total Revenue", f"₩{total_rev / 1e6:,.2f}M", help=f"Exact value: ₩{total_rev:,.0f} (South Korean Won)")
k3.metric("Overall ROAS", f"{overall_roas:.2f}x", help=f"Exact ratio: {overall_roas:.4f}x")
k4.metric("Avg Conversion Rate", f"{avg_conv_rate:.2f}%", help=f"Exact rate: {avg_conv_rate:.4f}%")
k5.metric(
    "Profit",
    f"₩{total_profit / 1e6:,.2f}M",
    delta="Profitable" if total_profit >= 0 else "Loss",
    help=f"Exact profit: ₩{total_profit:,.0f} (South Korean Won)",
)
k6.metric("Waste Avoided", f"{total_waste / 1e3:,.2f} Tons", help=f"Exact weight: {total_waste:,.0f} kg")

# 2. Annual Overview
if selected_year == "All years":
    st.markdown("---")
    st.subheader("📅 Annual Overview (All Years)")
    ann = df_all.groupby("year", as_index=False).agg({"spend": "sum", "revenue": "sum", "waste_kg": "sum"})
    ann["spend_m"] = ann["spend"] / 1e6
    ann["rev_m"] = ann["revenue"] / 1e6
    ac1, ac2 = st.columns(2)
    with ac1:
        fig_ann_fin = px.bar(
            ann,
            x="year",
            y=["spend_m", "rev_m"],
            barmode="group",
            labels={"value": "Amount (KRW million)", "year": "Year", "variable": "Metric"},
            title="Spend vs Revenue per Year (KRW million)",
            template="plotly_white",
        )
        st.plotly_chart(fig_ann_fin, use_container_width=True)
    with ac2:
        fig_ann_waste = px.bar(
            ann,
            x="year",
            y="waste_kg",
            labels={"waste_kg": "Waste avoided (kg)", "year": "Year"},
            title="Waste Avoided per Year (kg)",
            template="plotly_white",
            color_discrete_sequence=["#2e7d32"],
        )
        st.plotly_chart(fig_ann_waste, use_container_width=True)

# 3. Campaign Visualizations
st.markdown("---")
st.subheader("📈 Visualizations")
df_viz = df.copy()
df_viz["spend_m"] = df_viz["spend"] / 1e6
df_viz["rev_m"] = df_viz["revenue"] / 1e6

v1, v2 = st.columns(2)
with v1:
    fig_sr = px.bar(
        df_viz,
        x="campaign",
        y=["spend_m", "rev_m"],
        barmode="group",
        labels={"value": "Amount (KRW million)", "campaign": "Campaign", "variable": "Metric"},
        title="Spend vs Revenue by Campaign (KRW million)",
        template="plotly_white",
    )
    st.plotly_chart(fig_sr, use_container_width=True)

    fig_waste = px.bar(
        df_viz,
        x="campaign",
        y="waste_kg",
        labels={"waste_kg": "Waste avoided (kg)", "campaign": "Campaign"},
        title="Waste Avoided by Campaign (kg)",
        template="plotly_white",
        color_discrete_sequence=["#00897b"],
    )
    st.plotly_chart(fig_waste, use_container_width=True)

with v2:
    fig_conv = px.bar(
        df_viz,
        x="campaign",
        y="conversions",
        labels={"conversions": "Conversions", "campaign": "Campaign"},
        title="Conversions by Campaign",
        template="plotly_white",
        color_discrete_sequence=["#1e88e5"],
    )
    st.plotly_chart(fig_conv, use_container_width=True)

    fig_eff = px.bar(
        df_viz,
        x="campaign",
        y="kg_per_million",
        labels={"kg_per_million": "kg per ₩1M", "campaign": "Campaign"},
        title="Waste Avoided per ₩1M Spent (kg)",
        template="plotly_white",
        color_discrete_sequence=["#43a047"],
    )
    st.plotly_chart(fig_eff, use_container_width=True)

# ================= ML CLUSTERING SECTION =================
st.markdown("---")
st.subheader("🎯 ML Customer & Campaign Segments (K-Means)")
st.write(
    "This segment utilizes **Scikit-Learn K-Means Unsupervised Learning** to categorize "
    "your campaigns automatically into performance tiers based on **Spend** and **Waste Avoided (kg)**."
)

if len(df_all) >= 3:
    scaler = StandardScaler()
    features = df_all[["spend", "waste_kg"]]
    scaled_features = scaler.fit_transform(features)
    num_clusters = min(3, len(df_all))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df_all["cluster"] = kmeans.fit_predict(scaled_features)
    df_all["cluster"] = df_all["cluster"].apply(lambda x: f"Segment {x + 1}")

    df = df.merge(df_all[["id", "cluster"]], on="id", suffixes=("", "_new"))
    if "cluster_new" in df.columns:
        df["cluster"] = df["cluster_new"]
        df.drop(columns=["cluster_new"], inplace=True)

    fig_cluster = px.scatter(
        df_all,
        x="spend",
        y="waste_kg",
        color="cluster",
        hover_data=["campaign", "year", "roas"],
        labels={"spend": "Campaign Spend (KRW)", "waste_kg": "Waste Avoided (kg)", "cluster": "Performance Segment"},
        title="Campaign Clustering: Budget vs Sustainability Impact",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig_cluster.update_traces(marker=dict(size=12, line=dict(width=1, color="DarkSlateGrey")))
    st.plotly_chart(fig_cluster, use_container_width=True)
else:
    st.info("💡 Add at least 3 campaigns to see ML performance segments (K-Means).")

# 4. Advanced AI Campaign Predictor
st.markdown("---")
st.subheader("🤖 Advanced AI Campaign Predictor & Comparison")
st.write(
    "This Machine Learning module uses **Multiple Linear Regression (Scikit-Learn)** trained on your SQLite database "
    "to predict future campaign **Revenue** and **ROAS** from planned **Spend**, **Clicks**, and **Reach**."
)

ml_model, df_train = train_roas_model(len(df_all))
if ml_model is None:
    st.warning("⚠️ At least 3 campaign records are required to train the AI model.")
else:
    p1, p2, p3 = st.columns(3)
    with p1:
        planned_spend = st.number_input("Target Spend (KRW)", min_value=10000, max_value=20000000, value=2000000, step=100000, format="%d")
    with p2:
        planned_clicks = st.number_input("Estimated Clicks", min_value=10, max_value=100000, value=4500, step=100, format="%d")
    with p3:
        planned_reach = st.number_input("Estimated Reach", min_value=100, max_value=1000000, value=95000, step=5000, format="%d")

    input_df = pd.DataFrame([[planned_spend, planned_clicks, planned_reach]], columns=["spend", "clicks", "reach"])
    pred_rev = max(0.0, float(ml_model.predict(input_df)[0]))
    pred_roas = (pred_rev / planned_spend) if planned_spend > 0 else 0.0

    r1, r2 = st.columns(2)
    r1.metric("🎯 Predicted Revenue", f"₩{int(pred_rev):,}", help=f"Exact predicted amount: ₩{pred_rev:,.2f}")
    r2.metric("📈 Predicted ROAS", f"{pred_roas:.2f}x", help=f"Exact predicted ratio: {pred_roas:.4f}x")

    fig_ai = px.scatter(
        df_train,
        x="spend",
        y="revenue",
        hover_data=["campaign", "year", "clicks", "reach", "roas"],
        labels={"spend": "Campaign Spend (KRW)", "revenue": "Generated Revenue (KRW)"},
        title="Spend vs Revenue: Historical Campaigns vs. AI Prediction",
        template="plotly_white",
    )
    fig_ai.add_scatter(
        x=[planned_spend],
        y=[pred_rev],
        mode="markers+text",
        name="Planned Campaign (AI)",
        text=["🚀 Planned"],
        textposition="top center",
        marker=dict(color="red", size=16, symbol="star"),
    )
    fig_ai.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_ai, use_container_width=True)

# 5. Automated Strategy Recommendations
st.markdown("---")
st.subheader("🧠 Automated Strategy Recommendations")
best_row = df.loc[df["roas"].idxmax()]
worst_row = df.loc[df["roas"].idxmin()]

rec1, rec2 = st.columns(2)
with rec1:
    st.success(
        f"🏆 **Best performer: {best_row['campaign']}**\n\n"
        f"ROAS **{best_row['roas']:.2f}x** | Profit **₩{best_row['profit'] / 1e6:,.2f}M** | Conversion rate **{best_row['conversion_rate']:.2f}%**\n\n"
        "✅ **Recommendation:** Increase this campaign's budget by 15–20% and replicate its messaging across similar channels."
    )
with rec2:
    st.error(
        f"⚠️ **Weakest performer: {worst_row['campaign']}**\n\n"
        f"ROAS **{worst_row['roas']:.2f}x** | Profit **₩{worst_row['profit'] / 1e6:,.2f}M** | Conversion rate **{worst_row['conversion_rate']:.2f}%**\n\n"
        "🔻 **Recommendation:** Run a performance review, pause or reduce budget, and reallocate funds to higher-ROAS initiatives."
    )

# 6. Single Campaign Deep-Dive
st.markdown("---")
st.subheader("🔍 Single Campaign Analysis")
df_ranked = df.sort_values("roas", ascending=False).reset_index(drop=True)
df_ranked["rank"] = df_ranked.index + 1
options = [f"{r['id']} - {r['campaign']}" for _, r in df.iterrows()]
selected_opt = st.selectbox("Choose a campaign to analyze", options)
sel_id = int(selected_opt.split(" - ")[0])
c_row = df_ranked[df_ranked["id"] == sel_id].iloc[0]

st.markdown(f"#### {c_row['campaign']} — ROAS rank #{int(c_row['rank'])} of {len(df_ranked)}")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("ROAS", f"{c_row['roas']:.2f}x", f"{c_row['roas'] - df['roas'].mean():+.2f} vs avg", help=f"Exact Campaign ROAS: {c_row['roas']:.4f}x")
m2.metric("Profit", f"₩{c_row['profit'] / 1e6:,.2f}M", f"₩{(c_row['profit'] - df['profit'].mean()) / 1e3:+,.0f}K vs avg", help=f"Exact Campaign Profit: ₩{c_row['profit']:,.0f}")
m3.metric("Conversion Rate", f"{c_row['conversion_rate']:.2f}%", f"{c_row['conversion_rate'] - df['conversion_rate'].mean():+.2f} pts", help=f"Exact Conversion Rate: {c_row['conversion_rate']:.4f}%")
m4.metric("CTR", f"{c_row['ctr']:.2f}%", f"{c_row['ctr'] - df['ctr'].mean():+.2f} pts", help=f"Exact Click-Through Rate: {c_row['ctr']:.4f}%")
m5.metric("CPA", f"₩{c_row['cpa']:,.0f}", f"₩{c_row['cpa'] - df['cpa'].mean():+,.0f} vs avg", help=f"Exact Cost Per Acquisition: ₩{c_row['cpa']:,.2f}")
m6.metric("Waste Avoided", f"{c_row['waste_kg']:,.0f} kg", f"{c_row['waste_kg'] - df['waste_kg'].mean():+,.0f} kg vs avg", help=f"Exact Waste Avoided: {c_row['waste_kg']:,.0f} kg")

# 7. Manage Campaigns (CRUD)
st.markdown("---")
st.subheader("🗂️ Manage Campaigns (Create • Read • Update • Delete)")
tab_edit, tab_del = st.tabs(["✏️ Update Campaign", "🗑️ Delete Campaign"])

with tab_edit:
    edit_opt = st.selectbox("Select campaign to edit", [f"{r['id']} - {r['campaign']} ({r['year']})" for _, r in df_all.iterrows()], key="edit_sel")
    edit_id = int(edit_opt.split(" - ")[0])
    erow = df_all[df_all["id"] == edit_id].iloc[0]
    with st.form("edit_campaign_form"):
        ec1, ec2, ec3, ec4 = st.columns(4)
        u_name = ec1.text_input("Campaign Name", value=erow["campaign"])
        u_year = ec2.number_input("Year", value=int(erow["year"]), step=1)
        u_reach = ec3.number_input("Reach", value=int(erow["reach"]), step=1000)
        u_clicks = ec4.number_input("Clicks", value=int(erow["clicks"]), step=100)
        ec5, ec6, ec7, ec8 = st.columns(4)
        u_conv = ec5.number_input("Conversions", value=int(erow["conversions"]), step=10)
        u_spend = ec6.number_input("Spend (KRW)", value=float(erow["spend"]), step=100000.0)
        u_rev = ec7.number_input("Revenue (KRW)", value=float(erow["revenue"]), step=100000.0)
        u_waste = ec8.number_input("Waste Avoided (kg)", value=float(erow["waste_kg"]), step=100.0)
        if st.form_submit_button("Update Campaign"):
            update_campaign(edit_id, u_name, u_year, u_reach, u_clicks, u_conv, u_spend, u_rev, u_waste)
            st.cache_resource.clear()
            st.success("Campaign updated!")
            st.rerun()

with tab_del:
    del_opt = st.selectbox("Select campaign to delete", [f"{r['id']} - {r['campaign']} ({r['year']})" for _, r in df_all.iterrows()], key="del_sel")
    del_id = int(del_opt.split(" - ")[0])
    if st.button("🗑️ Permanently Delete Selected Campaign", type="primary"):
        delete_campaign(del_id)
        st.cache_resource.clear()
        st.warning("Campaign deleted.")
        st.rerun()

with st.expander("📄 View complete dataset from SQLite (table view)", expanded=True):
    st.dataframe(df, use_container_width=True, hide_index=True)
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download as CSV", data=csv_data, file_name="3r_campaigns.csv", mime="text/csv")
