import numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Last-Mile Delivery Time Predictor", page_icon="🚚")

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

TRAFFIC = {"Low": .9, "Medium": 1.05, "High": 1.25, "Jam": 1.5}
WEATHER = {"Sunny": 1, "Cloudy": 1.03, "Fog": 1.12, "Windy": 1.05, "Stormy": 1.2}
VEHICLE = {"motorcycle": 1, "scooter": 1.05, "van": 1.1, "bicycle": 1.4}

@st.cache_resource
def train():
    rng = np.random.default_rng(42); n = 15000
    d = pd.DataFrame({
        "slat": rng.uniform(12.9, 13.1, n), "slon": rng.uniform(77.5, 77.7, n),
        "traffic": rng.choice(list(TRAFFIC), n), "weather": rng.choice(list(WEATHER), n),
        "vehicle": rng.choice(list(VEHICLE), n), "rating": rng.uniform(3, 5, n).round(1)})
    d["dlat"] = d.slat + rng.uniform(-.1, .1, n); d["dlon"] = d.slon + rng.uniform(-.1, .1, n)
    d["km"] = haversine(d.slat, d.slon, d.dlat, d.dlon)
    t = (15 + 4.5*d.km) * d.traffic.map(TRAFFIC) * d.weather.map(WEATHER) * d.vehicle.map(VEHICLE) * (1.15 - .03*d.rating)
    y = (t + rng.normal(0, 4, n)).clip(10)
    X = pd.get_dummies(d[["km", "rating", "traffic", "weather", "vehicle"]], dtype=int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
    m = RandomForestRegressor(n_estimators=150, max_depth=15, min_samples_leaf=2, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    p = m.predict(Xte)
    res = {"MAE": mean_absolute_error(yte, p), "RMSE": mean_squared_error(yte, p)**.5, "R2": r2_score(yte, p)}
    return m, list(X.columns), res, yte.values[:500], p[:500]

model, cols, res, yt, yp = train()

st.title("🚚 AI-Based Last-Mile Delivery Time Predictor")
c1, c2 = st.columns(2)
slat = c1.number_input("Store latitude", value=12.9716, format="%.4f")
slon = c1.number_input("Store longitude", value=77.5946, format="%.4f")
dlat = c2.number_input("Drop latitude", value=13.0358, format="%.4f")
dlon = c2.number_input("Drop longitude", value=77.6700, format="%.4f")
traffic = c1.selectbox("Traffic", list(TRAFFIC)); weather = c2.selectbox("Weather", list(WEATHER))
vehicle = c1.selectbox("Vehicle", list(VEHICLE)); rating = c2.slider("Partner rating", 1.0, 5.0, 4.5, .1)

if st.button("Predict delivery time", type="primary"):
    km = float(haversine(slat, slon, dlat, dlon))
    row = dict.fromkeys(cols, 0); row.update({"km": km, "rating": rating})
    for k in (f"traffic_{traffic}", f"weather_{weather}", f"vehicle_{vehicle}"):
        if k in row: row[k] = 1
    st.metric("Estimated delivery time", f"{model.predict(pd.DataFrame([row])[cols])[0]:.0f} minutes")
    st.write(f"Distance: **{km:.2f} km**")

st.subheader("Model results")
a, b, c = st.columns(3)
a.metric("MAE (min)", f"{res['MAE']:.2f}"); b.metric("RMSE (min)", f"{res['RMSE']:.2f}"); c.metric("R² score", f"{res['R2']:.3f}")
fig, ax = plt.subplots(figsize=(4, 4)); ax.scatter(yt, yp, s=5, alpha=.4)
ax.plot([yt.min(), yt.max()], [yt.min(), yt.max()], "r--"); ax.set_xlabel("Actual (min)"); ax.set_ylabel("Predicted (min)")
st.pyplot(fig)
st.caption("Model trained on generated sample data with the same fields as the Kaggle Amazon Delivery dataset.")
