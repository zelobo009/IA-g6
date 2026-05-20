"""
app.py — Airbnb Dynamic Pricing POC
Run: streamlit run app.py
"""

import pickle, io, calendar
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from generate_data import generate, LOCATIONS, EVENTS

# ─── Page ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AirPrice · Dynamic Pricing", page_icon="🏠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.listing-card {
    background: #fff;
    border: 1px solid #e8e0d8;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    position: relative;
}
.listing-title { font-weight: 600; font-size: 15px; color: #1a1a1a; margin: 0 0 2px; }
.listing-meta  { font-size: 12px; color: #888; margin: 0; }
.price-tag {
    background: #ff5a5f;
    color: white;
    border-radius: 8px;
    padding: 4px 10px;
    font-weight: 600;
    font-size: 14px;
    display: inline-block;
}
.occ-bar-bg { background: #f0ece6; border-radius: 6px; height: 6px; width: 100%; }
.badge {
    display: inline-block;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 500;
    margin-right: 4px;
}
.badge-high   { background: #fff0f0; color: #d63f3f; }
.badge-med    { background: #fff8e6; color: #b07a00; }
.badge-low    { background: #edfaf3; color: #1d9e75; }
.stat-box {
    background: #faf8f5;
    border: 1px solid #ece6dc;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.stat-num  { font-family: 'DM Serif Display', serif; font-size: 28px; color: #1a1a1a; }
.stat-lbl  { font-size: 12px; color: #999; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ─── State ───────────────────────────────────────────────────────────────────
if "extra_rows"  not in st.session_state: st.session_state.extra_rows  = []
if "listings"    not in st.session_state: st.session_state.listings    = []
if "model_cache" not in st.session_state: st.session_state.model_cache = None
if "data_cache"  not in st.session_state: st.session_state.data_cache  = None

# ─── Model helpers ────────────────────────────────────────────────────────────
CAT = ["event","location"]
NUM = ["month","day_of_week","lead_days","rooms"]
FEATURES = NUM + CAT

def build_pipeline():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", StandardScaler(), NUM),
    ])
    return Pipeline([("pre", pre),
                     ("model", GradientBoostingRegressor(
                         n_estimators=200, max_depth=4,
                         learning_rate=0.05, random_state=42))])

@st.cache_resource(show_spinner=False)
def train_models(n_extra):
    df      = generate(4000, st.session_state.extra_rows or None)
    X       = df[FEATURES]
    y_occ   = df["occupancy"]
    y_price = df["price_eur"]
    occ_p   = build_pipeline(); occ_p.fit(X, y_occ)
    price_p = build_pipeline(); price_p.fit(X, y_price)
    r2_occ   = cross_val_score(occ_p,   X, y_occ,   cv=5, scoring="r2").mean()
    r2_price = cross_val_score(price_p, X, y_price, cv=5, scoring="r2").mean()
    return occ_p, price_p, r2_occ, r2_price, df

def get_models():
    n_extra = len(st.session_state.extra_rows)
    return train_models(n_extra)

def predict_one(occ_p, price_p, month, dow, event, lead, location, rooms):
    X = pd.DataFrame([{"month": month, "day_of_week": dow, "event": event,
                        "lead_days": lead, "location": location, "rooms": rooms}])
    occ   = float(np.clip(occ_p.predict(X)[0], 10, 99))
    price = float(price_p.predict(X)[0])
    return round(occ, 1), round(price, 0)

DOW_NAMES   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def demand_badge(occ):
    if occ >= 80: return "badge-high",  "🔴 Very High"
    if occ >= 60: return "badge-med",   "🟠 High"
    if occ >= 40: return "badge-low",   "🟡 Moderate"
    return "badge-low", "🟢 Low"

def occ_color(occ):
    if occ >= 80: return "#ff5a5f"
    if occ >= 60: return "#ffb400"
    if occ >= 40: return "#00a699"
    return "#767676"

# ─── Load models ─────────────────────────────────────────────────────────────
with st.spinner("Training model…"):
    occ_pipe, price_pipe, r2_occ, r2_price, base_df = get_models()

# ─── Layout ───────────────────────────────────────────────────────────────────
st.markdown("# 🏠 AirPrice")
st.caption("Dynamic pricing & demand forecasting for short-term rentals · ML POC")
st.divider()

tab_dash, tab_listings, tab_predict, tab_data, tab_model = st.tabs([
    "📊 Dashboard", "🏠 My Listings", "🔮 Price Predictor", "➕ Add Training Data", "🧠 Model Info"
])

# ══════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════
with tab_dash:
    st.subheader("Market Overview")

    # KPI row
    k1,k2,k3,k4 = st.columns(4)
    avg_price = base_df["price_eur"].mean()
    avg_occ   = base_df["occupancy"].mean()
    n_locs    = base_df["location"].nunique()
    peak_month = MONTH_NAMES[base_df.groupby("month")["occupancy"].mean().idxmax()-1]

    with k1:
        st.markdown(f'<div class="stat-box"><div class="stat-num">€{avg_price:.0f}</div><div class="stat-lbl">Avg nightly price</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{avg_occ:.0f}%</div><div class="stat-lbl">Avg occupancy</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{n_locs}</div><div class="stat-lbl">Locations tracked</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{peak_month}</div><div class="stat-lbl">Peak month</div></div>', unsafe_allow_html=True)

    st.markdown("")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Average occupancy by month**")
        monthly = base_df.groupby("month")["occupancy"].mean()
        fig, ax = plt.subplots(figsize=(6,3))
        colors  = [occ_color(v) for v in monthly.values]
        ax.bar(MONTH_NAMES, monthly.values, color=colors, edgecolor="none", width=0.7)
        ax.set_ylim(0,100); ax.set_ylabel("Occupancy (%)")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        st.markdown("**Average price by location**")
        loc_price = base_df.groupby("location")["price_eur"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(6,3))
        ax.barh(loc_price.index, loc_price.values, color="#ff5a5f", edgecolor="none")
        ax.set_xlabel("Avg price (€)")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("**Occupancy heatmap — Day of week × Month**")
    pivot = base_df.groupby(["month","day_of_week"])["occupancy"].mean().unstack()
    pivot.index = [MONTH_NAMES[m-1] for m in pivot.index]
    pivot.columns = DOW_NAMES
    fig, ax = plt.subplots(figsize=(10,4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=30, vmax=90)
    ax.set_xticks(range(7)); ax.set_xticklabels(DOW_NAMES)
    ax.set_yticks(range(12)); ax.set_yticklabels(MONTH_NAMES)
    plt.colorbar(im, ax=ax, label="Occupancy %")
    for i in range(12):
        for j in range(7):
            ax.text(j, i, f"{pivot.values[i,j]:.0f}", ha="center", va="center", fontsize=7, color="black")
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════
# TAB 2 — MY LISTINGS
# ══════════════════════════════════════════════════════════
with tab_listings:
    st.subheader("My Listings")

    with st.expander("➕ Add a new listing", expanded=len(st.session_state.listings)==0):
        with st.form("add_listing"):
            c1,c2 = st.columns(2)
            name     = c1.text_input("Listing name", placeholder="e.g. Cozy Loft near Metro")
            location = c2.selectbox("Location", list(LOCATIONS.keys()))
            rooms    = c1.slider("Number of rooms", 1, 5, 2)
            desc     = c2.text_area("Short description", placeholder="Beautiful flat with terrace…", height=68)
            submitted = st.form_submit_button("Add listing", use_container_width=True)
            if submitted and name:
                st.session_state.listings.append({
                    "name": name, "location": location,
                    "rooms": rooms, "desc": desc
                })
                st.success(f"'{name}' added!")
                st.rerun()

    if not st.session_state.listings:
        st.info("No listings yet — add one above to see price forecasts.")
    else:
        view_month = st.selectbox("Preview prices for month", range(1,13),
                                   format_func=lambda m: MONTH_NAMES[m-1],
                                   key="listing_month")
        view_event = st.selectbox("Local event", EVENTS, format_func=str.capitalize, key="listing_event")

        for i, listing in enumerate(st.session_state.listings):
            # Predict for each day of week
            occs, prices = [], []
            for dow in range(7):
                o, p = predict_one(occ_pipe, price_pipe,
                                   view_month, dow, view_event, 14,
                                   listing["location"], listing["rooms"])
                occs.append(o); prices.append(p)

            avg_occ_l   = np.mean(occs)
            avg_price_l = np.mean(prices)
            badge_cls, badge_txt = demand_badge(avg_occ_l)

            col_info, col_chart = st.columns([1.4, 1])
            with col_info:
                st.markdown(f"""
                <div class="listing-card">
                  <p class="listing-title">🏠 {listing['name']}</p>
                  <p class="listing-meta">📍 {listing['location']} &nbsp;·&nbsp; 🛏 {listing['rooms']} room{"s" if listing["rooms"]>1 else ""}</p>
                  <p class="listing-meta" style="margin-top:4px;color:#555;">{listing.get('desc','')}</p>
                  <div style="margin-top:10px;">
                    <span class="price-tag">€{avg_price_l:.0f}/night</span>
                    &nbsp;<span class="badge {badge_cls}">{badge_txt}</span>
                  </div>
                  <div style="margin-top:8px;">
                    <div style="font-size:11px;color:#999;margin-bottom:3px;">Avg occupancy {avg_occ_l:.0f}%</div>
                    <div class="occ-bar-bg"><div style="background:{occ_color(avg_occ_l)};width:{avg_occ_l}%;height:6px;border-radius:6px;"></div></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"🗑 Remove", key=f"del_{i}"):
                    st.session_state.listings.pop(i); st.rerun()

            with col_chart:
                fig, ax = plt.subplots(figsize=(4, 2.5))
                bar_colors = [occ_color(o) for o in occs]
                ax.bar(DOW_NAMES, prices, color=bar_colors, edgecolor="none", width=0.6)
                ax.set_ylabel("€/night", fontsize=9)
                ax.set_title(f"{MONTH_NAMES[view_month-1]} forecast", fontsize=10)
                ax.spines[["top","right"]].set_visible(False)
                for b, p in zip(ax.patches, prices):
                    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"€{p:.0f}",
                            ha="center", va="bottom", fontsize=7)
                plt.tight_layout(); st.pyplot(fig); plt.close()

        # Calendar view for first listing
        if st.session_state.listings:
            st.markdown("---")
            st.markdown("**📅 Monthly price calendar**")
            cal_listing_idx = st.selectbox("Select listing", range(len(st.session_state.listings)),
                                            format_func=lambda i: st.session_state.listings[i]["name"],
                                            key="cal_listing")
            cal_listing = st.session_state.listings[cal_listing_idx]
            cal_month   = st.selectbox("Month", range(1,13), index=view_month-1,
                                        format_func=lambda m: MONTH_NAMES[m-1], key="cal_month")
            cal_event   = st.selectbox("Event", EVENTS, format_func=str.capitalize, key="cal_event")

            # Build calendar grid
            year = 2025
            cal  = calendar.monthcalendar(year, cal_month)
            rows = []
            for week in cal:
                row = []
                for day in week:
                    if day == 0:
                        row.append(("", "", 0))
                    else:
                        d = pd.Timestamp(year, cal_month, day)
                        dow_idx = d.weekday()
                        o, p = predict_one(occ_pipe, price_pipe, cal_month, dow_idx,
                                           cal_event, 14, cal_listing["location"], cal_listing["rooms"])
                        row.append((day, f"€{p:.0f}", o))
                rows.append(row)

            fig, ax = plt.subplots(figsize=(10, len(rows)*1.1 + 0.5))
            ax.set_xlim(0,7); ax.set_ylim(0, len(rows))
            ax.axis("off")
            for c, dname in enumerate(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]):
                ax.text(c+0.5, len(rows)-0.1, dname, ha="center", va="top",
                        fontsize=10, fontweight="bold", color="#555")
            for r, week in enumerate(rows):
                for c, (day, price_str, occ) in enumerate(week):
                    y = len(rows) - r - 1
                    if day:
                        color = occ_color(occ)
                        rect = plt.Rectangle((c+0.05, y+0.05), 0.9, 0.88,
                                             facecolor=color+"22", edgecolor=color, linewidth=1.2, zorder=1)
                        ax.add_patch(rect)
                        ax.text(c+0.15, y+0.78, str(day), fontsize=9, color="#333", va="top")
                        ax.text(c+0.5, y+0.35, price_str, ha="center", fontsize=10,
                                fontweight="bold", color=color)
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════
# TAB 3 — PRICE PREDICTOR
# ══════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("🔮 Single Night Price Predictor")
    st.caption("Adjust parameters and see model output instantly")

    cp1, cp2 = st.columns([1, 1.6])

    with cp1:
        p_location = st.selectbox("Location", list(LOCATIONS.keys()), key="p_loc")
        p_rooms    = st.slider("Rooms", 1, 5, 2, key="p_rooms")
        p_month    = st.selectbox("Month", range(1,13), format_func=lambda m: MONTH_NAMES[m-1], key="p_month")
        p_dow      = st.selectbox("Day of week", range(7), format_func=lambda d: DOW_NAMES[d], key="p_dow")
        p_event    = st.selectbox("Local event", EVENTS, format_func=str.capitalize, key="p_event")
        p_lead     = st.slider("Lead time (days)", 1, 90, 14, key="p_lead")

    pred_occ, pred_price = predict_one(occ_pipe, price_pipe,
                                       p_month, p_dow, p_event,
                                       p_lead, p_location, p_rooms)
    badge_cls, badge_txt = demand_badge(pred_occ)

    with cp2:
        st.markdown(f"""
        <div style="background:#fff9f5;border:1.5px solid #ffd4c4;border-radius:16px;padding:1.5rem;margin-bottom:1rem;">
          <p style="font-size:13px;color:#999;margin:0;">Suggested nightly rate</p>
          <p style="font-family:'DM Serif Display',serif;font-size:48px;color:#ff5a5f;margin:4px 0;">€{pred_price:.0f}</p>
          <span class="badge {badge_cls}" style="font-size:13px;">{badge_txt}</span>
          <p style="font-size:13px;color:#555;margin-top:8px;">Predicted occupancy: <b>{pred_occ:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

        # Lead time curve
        st.markdown("**How lead time affects price**")
        leads  = range(1, 91)
        lprices = [predict_one(occ_pipe, price_pipe, p_month, p_dow, p_event, l, p_location, p_rooms)[1]
                   for l in leads]
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.plot(list(leads), lprices, color="#ff5a5f", lw=2)
        ax.axvline(p_lead, color="#484848", lw=1.5, ls="--",
                   label=f"Selected: {p_lead}d → €{pred_price:.0f}")
        ax.set_xlabel("Lead time (days)"); ax.set_ylabel("€/night")
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=9); plt.tight_layout(); st.pyplot(fig); plt.close()

        # Event comparison
        st.markdown("**Event impact**")
        ev_occs = {ev: predict_one(occ_pipe, price_pipe, p_month, p_dow, ev, p_lead, p_location, p_rooms)[0]
                   for ev in EVENTS}
        fig, ax = plt.subplots(figsize=(5, 2.2))
        cols_ev = ["#ff5a5f" if ev == p_event else "#ffc4be" for ev in EVENTS]
        ax.barh([e.capitalize() for e in EVENTS], [ev_occs[e] for e in EVENTS],
                color=cols_ev, edgecolor="none")
        ax.set_xlabel("Predicted occupancy (%)")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════
# TAB 4 — ADD TRAINING DATA
# ══════════════════════════════════════════════════════════
with tab_data:
    st.subheader("➕ Add Real Observations")
    st.caption("Add actual booking data to retrain and improve the model")

    with st.form("add_data"):
        d1,d2,d3 = st.columns(3)
        r_month    = d1.selectbox("Month",    range(1,13), format_func=lambda m: MONTH_NAMES[m-1])
        r_dow      = d1.selectbox("Day",      range(7),    format_func=lambda d: DOW_NAMES[d])
        r_event    = d2.selectbox("Event",    EVENTS,      format_func=str.capitalize)
        r_lead     = d2.slider("Lead days",   1, 90, 14)
        r_location = d3.selectbox("Location", list(LOCATIONS.keys()))
        r_rooms    = d3.slider("Rooms",       1, 5, 2)
        r_occ      = d1.number_input("Actual occupancy (%)", 0.0, 100.0, 75.0, step=1.0)
        r_price    = d2.number_input("Actual price (€)",     0.0, 2000.0, 100.0, step=5.0)
        note       = d3.text_input("Note (optional)")
        add_btn    = st.form_submit_button("Add observation & retrain", use_container_width=True)

        if add_btn:
            row = {"month": r_month, "day_of_week": r_dow, "event": r_event,
                   "lead_days": r_lead, "location": r_location, "rooms": r_rooms,
                   "occupancy": r_occ, "price_eur": r_price}
            st.session_state.extra_rows.append(row)
            train_models.clear()
            st.success(f"Added! Model will retrain with {len(st.session_state.extra_rows)} extra observation(s).")
            st.rerun()

    if st.session_state.extra_rows:
        st.markdown(f"**{len(st.session_state.extra_rows)} custom observation(s) added:**")
        st.dataframe(pd.DataFrame(st.session_state.extra_rows), use_container_width=True)
        if st.button("🗑 Clear all custom data"):
            st.session_state.extra_rows = []
            train_models.clear()
            st.rerun()
    else:
        st.info("No custom observations yet. The model runs on synthetic data.")

# ══════════════════════════════════════════════════════════
# TAB 5 — MODEL INFO
# ══════════════════════════════════════════════════════════
with tab_model:
    st.subheader("🧠 Model Details")

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Occupancy R²",    f"{r2_occ:.3f}")
    m2.metric("Price R²",        f"{r2_price:.3f}")
    m3.metric("Training rows",   f"{4000 + len(st.session_state.extra_rows):,}")
    m4.metric("Custom obs.",     len(st.session_state.extra_rows))

    st.markdown("**Algorithm:** Gradient Boosting Regressor (scikit-learn) — 200 estimators, depth 4")
    st.markdown("**Features:** month, day of week, event type, lead time, location, number of rooms")
    st.markdown("**Two models trained:** one predicts occupancy %, one predicts nightly price (€)")

    # Feature importance
    st.markdown("---")
    st.markdown("**Feature importances — occupancy model**")
    ohe_names = (occ_pipe.named_steps["pre"]
                 .named_transformers_["cat"]
                 .get_feature_names_out(CAT).tolist())
    feat_names   = ohe_names + NUM
    importances  = occ_pipe.named_steps["model"].feature_importances_
    idx = np.argsort(importances)[-15:]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    cols = ["#ff5a5f" if importances[i] > np.median(importances) else "#ffc4be" for i in idx]
    ax.barh([feat_names[i] for i in idx], importances[idx], color=cols, edgecolor="none")
    ax.set_xlabel("Importance"); ax.set_title("Top features driving occupancy prediction")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Actual vs predicted
    st.markdown("**Model fit — actual vs predicted (test set sample)**")
    sample = base_df.sample(300, random_state=1)
    X_s    = sample[FEATURES]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, y_true, pipe, label, unit in zip(
        axes,
        [sample["occupancy"], sample["price_eur"]],
        [occ_pipe, price_pipe],
        ["Occupancy","Price"],
        ["%","€"]
    ):
        y_hat = pipe.predict(X_s)
        ax.scatter(y_true, y_hat, alpha=0.25, s=12, color="#ff5a5f")
        lo, hi = min(y_true.min(), y_hat.min()), max(y_true.max(), y_hat.max())
        ax.plot([lo,hi],[lo,hi], "k--", lw=1)
        ax.set_xlabel(f"Actual {label} ({unit})"); ax.set_ylabel(f"Predicted")
        ax.set_title(f"{label} · R²={r2_score(y_true, y_hat):.3f}")
        ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()
