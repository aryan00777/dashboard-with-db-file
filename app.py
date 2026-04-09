import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# ---------------- CONFIG ----------------
st.set_page_config(page_title="SaaS Dashboard", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main { background-color:#0e1117; color:white; }
[data-testid="stSidebar"] { background:#111827; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD ----------------
@st.cache_data
def load():
    conn = sqlite3.connect("saas.db")
    df = pd.read_sql("SELECT * FROM users", conn)
    conn.close()
    return df

df = load()

# ---------------- CLEAN ----------------
df = df.dropna(subset=["country","signup_date"])
df = df[df["country"]!="Unknown"]

df["signup_date"] = pd.to_datetime(df["signup_date"])
df["date"] = df["signup_date"].dt.date
df["month"] = df["signup_date"].dt.to_period("M").astype(str)

df["session_time"] = df["session_time"].fillna(df["session_time"].median())

# ---------------- FILTERS ----------------
st.sidebar.title("Filters")

country = st.sidebar.multiselect("Country", df["country"].unique(), df["country"].unique())
channel = st.sidebar.multiselect("Channel", df["channel"].unique(), df["channel"].unique())
plan = st.sidebar.multiselect("Plan", df["plan"].unique(), df["plan"].unique())

df = df[(df["country"].isin(country)) &
        (df["channel"].isin(channel)) &
        (df["plan"].isin(plan))]

# ---------------- KPI ----------------
st.title("🚀 SaaS Intelligence Dashboard")

users = df["user_id"].nunique()
revenue = df["revenue"].sum()
conversion = df["converted"].mean()*100
arpu = revenue/users if users else 0
session = df["session_time"].mean()

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Users",f"{users:,}")
c2.metric("Revenue",f"${revenue:,.0f}")
c3.metric("Conversion",f"{conversion:.2f}%")
c4.metric("ARPU",f"${arpu:.2f}")
c5.metric("Session",f"{session:.1f}")

st.divider()

# ---------------- TABS ----------------
tab1,tab2,tab3,tab4 = st.tabs(["📈 Trends","🌍 Segments","🤖 ML","🧠 Insights"])

# ================= TRENDS =================
with tab1:
    daily = df.groupby("date").size().reset_index(name="users")
    daily["rolling"] = daily["users"].rolling(7).mean()
    daily["cumulative"] = daily["users"].cumsum()

    st.plotly_chart(px.line(daily,x="date",y="users",title="Daily Users"),use_container_width=True)
    st.plotly_chart(px.line(daily,x="date",y="rolling",title="7-Day Avg"),use_container_width=True)
    st.plotly_chart(px.line(daily,x="date",y="cumulative",title="Total Users"),use_container_width=True)

    monthly = df.groupby("month").size().reset_index(name="users")
    st.plotly_chart(px.bar(monthly,x="month",y="users",title="Monthly Growth"),use_container_width=True)

    # -------- FORECAST --------
    st.subheader("🔮 User Forecast")

    x = np.arange(len(daily))
    y = daily["users"].values

    coef = np.polyfit(x, y, 1)
    trend = coef[0]*x + coef[1]

    daily["forecast"] = trend

    fig = px.line(daily,x="date",y=["users","forecast"],title="Forecast vs Actual")
    st.plotly_chart(fig,use_container_width=True)

# ================= SEGMENTS =================
with tab2:
    st.plotly_chart(px.bar(df.groupby("country")["revenue"].sum().reset_index(),
                           x="country",y="revenue",title="Revenue by Country"),use_container_width=True)

    st.plotly_chart(px.bar(df.groupby("channel")["converted"].mean().reset_index(),
                           x="channel",y="converted",title="Conversion by Channel"),use_container_width=True)

    device = df["device"].value_counts().reset_index()
    device.columns=["device","count"]
    st.plotly_chart(px.pie(device,names="device",values="count"),use_container_width=True)

# ================= ML =================
# ================= TAB 3: ML =================
with tab3:
    st.subheader("🤖 Conversion Prediction")

    # ---------------- TRAIN MODEL ----------------
    features = ["session_time", "revenue", "country", "channel", "plan", "device"]

    X = pd.get_dummies(df[features])
    y = df["converted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    st.success(f"Model Accuracy: {accuracy:.2f}")

    st.divider()

    # ---------------- SESSION STATE ----------------
    if "predicted" not in st.session_state:
        st.session_state.predicted = False

    # ---------------- INPUT UI ----------------
    st.markdown("### 🔮 Predict Conversion")

    col1, col2 = st.columns(2)

    with col1:
        session_time = st.slider("Session Time", 1, 300, 50)

    with col2:
        revenue_i = st.slider(
            "Revenue",
            float(df["revenue"].min()),
            float(df["revenue"].max()),
            float(df["revenue"].median())
        )

    col3, col4 = st.columns(2)

    countries = ["All"] + list(df["country"].unique())
    devices = ["All"] + list(df["device"].unique())
    channels = ["All"] + list(df["channel"].unique())
    plans = ["All"] + list(df["plan"].unique())

    with col3:
        c = st.selectbox("Country", countries)

    with col4:
        d = st.selectbox("Device", devices)

    col5, col6 = st.columns(2)

    with col5:
        ch = st.selectbox("Channel", channels)

    with col6:
        p = st.selectbox("Plan", plans)

    # ---------------- BUILD INPUT ----------------
    input_dict = {col: 0 for col in X.columns}
    input_dict["session_time"] = session_time
    input_dict["revenue"] = revenue_i

    def encode(prefix, value):
        if value != "All":
            col = f"{prefix}_{value}"
            if col in input_dict:
                input_dict[col] = 1

    encode("country", c)
    encode("device", d)
    encode("channel", ch)
    encode("plan", p)

    input_df = pd.DataFrame([input_dict])

    # ---------------- PREDICT BUTTON ----------------
    if st.button("Predict Conversion"):
        st.session_state.predicted = True
        st.session_state.pred = model.predict(input_df)[0]
        st.session_state.prob = model.predict_proba(input_df)[0][1]

    # ---------------- SHOW RESULT ----------------
    if st.session_state.predicted:

        st.markdown("### 📊 Result")

        colA, colB = st.columns(2)

        with colA:
            st.metric("Conversion Probability", f"{st.session_state.prob:.2f}")

        with colB:
            if st.session_state.pred == 1:
                st.markdown("### 🟢 ✅ Likely to Convert")
            else:
                st.markdown("### 🔴 ❌ Unlikely to Convert")

        # ---------------- EXPLANATION ----------------
        st.divider()
        st.subheader("🧠 Model Explanation")

        st.write("""
This model predicts whether a user will convert based on past behavior.

### 🔍 What we use:
- Session time → how engaged the user is  
- Revenue → user value  
- Country, Channel, Plan, Device → user profile  

### ⚙️ How it works:
We use a **Random Forest model**, which builds many decision trees.
Each tree makes a prediction, and the final result is based on majority voting.

### 📊 What you see:
- **Probability** → chance user will convert  
- **Prediction** → final decision  

### 💼 Why this matters:
- Identify high-value users  
- Improve marketing targeting  
- Optimize user experience  
""")

        # ---------------- FEATURE IMPORTANCE ----------------
        st.subheader("📊 What drives conversion")

        importance = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False).head(10)

        fig = px.bar(
            importance,
            x="importance",
            y="feature",
            orientation="h",
            title="Top Factors Influencing Conversion"
        )

        st.plotly_chart(fig, use_container_width=True)
     

# ================= INSIGHTS =================
with tab4:
    st.subheader("🧠 Business Insights")

    ch_conv = df.groupby("channel")["converted"].mean().reset_index()
    co_rev = df.groupby("country")["revenue"].sum().reset_index()
    pl_rev = df.groupby("plan")["revenue"].sum().reset_index()

    best_ch = ch_conv.sort_values("converted",ascending=False).iloc[0]["channel"]
    worst_ch = ch_conv.sort_values("converted").iloc[0]["channel"]

    best_co = co_rev.sort_values("revenue",ascending=False).iloc[0]["country"]
    worst_co = co_rev.sort_values("revenue").iloc[0]["country"]

    best_pl = pl_rev.sort_values("revenue",ascending=False).iloc[0]["plan"]

    median = df["session_time"].median()
    high = df[df["session_time"]>median]["converted"].mean()*100
    low = df[df["session_time"]<=median]["converted"].mean()*100

    st.success(f"Best Channel: {best_ch}")
    st.warning(f"Worst Channel: {worst_ch}")

    st.success(f"Top Country: {best_co}")
    st.warning(f"Weak Country: {worst_co}")

    st.success(f"Best Plan: {best_pl}")

    st.info(f"High session converts: {high:.2f}% vs {low:.2f}%")

    st.divider()

    # -------- FUNNEL --------
    st.subheader("📉 Funnel")

    funnel = df["converted"].value_counts().reset_index()
    funnel.columns=["stage","count"]
    funnel["stage"]=funnel["stage"].map({1:"Converted",0:"Dropped"})

    st.plotly_chart(px.bar(funnel,x="stage",y="count",title="Funnel"),use_container_width=True)

    # -------- ACTIONS --------
    st.subheader("🎯 Actions")

    st.success(f"Focus budget on {best_ch}")
    st.warning(f"Fix {worst_ch}")

    st.success(f"Expand in {best_co}")
    st.warning(f"Investigate {worst_co}")

    if high>low:
        st.success("Increase session time (UX improvement)")
    else:
        st.warning("Focus on acquisition quality")