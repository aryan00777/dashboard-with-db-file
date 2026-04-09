import sqlite3
import pandas as pd
import numpy as np

np.random.seed(42)

conn = sqlite3.connect("saas.db")

n = 20000

df = pd.DataFrame({
    "user_id": range(1, n + 1),
    "signup_date": pd.date_range(start="2023-01-01", periods=n, freq="h"),
    "country": np.random.choice(["Denmark", "Sweden", "Norway"], n),
    "channel": np.random.choice(["Organic", "Paid", "Direct"], n, p=[0.5, 0.3, 0.2]),
    "plan": np.random.choice(["Basic", "Premium", "Pro"], n, p=[0.5, 0.3, 0.2]),
    "device": np.random.choice(["Mobile", "Desktop", "Tablet"], n)
})

# Pricing
price_map = {"Basic": 100, "Premium": 300, "Pro": 500}
df["price"] = df["plan"].map(price_map)

# User behavior
df["session_time"] = np.random.gamma(2, 3, n)

# Conversion logic (IMPORTANT — realistic)
channel_boost = df["channel"].map({"Organic": 0.1, "Paid": 0.15, "Direct": 0.05})
plan_boost = df["plan"].map({"Basic": 0.05, "Premium": 0.1, "Pro": 0.15})

prob = 0.25 + channel_boost + plan_boost + np.random.normal(0, 0.05, n)
prob = np.clip(prob, 0.05, 0.9)

df["converted"] = np.random.binomial(1, prob)

# Revenue
df["revenue"] = df["converted"] * df["price"]

# Dirty data (REAL WORLD)
df.loc[np.random.choice(df.index, 300), "session_time"] = None
df.loc[np.random.choice(df.index, 200), "country"] = None

# Save to database
df.to_sql("users", conn, if_exists="replace", index=False)

conn.close()

print("✅ Database created: saas.db")