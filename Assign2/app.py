import pickle, subprocess, os, calendar
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, mean_absolute_error, r2_score, accuracy_score
)
from generate_data import generate, LOCATIONS, EVENTS, PROPERTY_TYPES

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Airbnb Predictor", page_icon="🏠", layout="wide")
st.cache_data.clear()
st.cache_resource.clear()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.metric-card {
    background: #f7f4f0;
    border: 1px solid #e8e0d5;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-num  { font-size: 26px; font-weight: 700; color: #1a1a1a; margin: 0; }
.metric-lbl  { font-size: 11px; color: #999; margin: 4px 0 0; text-transform: uppercase; letter-spacing: 0.05em; }

.listing-card {
    background: #ffffff;
    border: 1px solid #e8e0d5;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.8rem;
}
.listing-name { font-size: 15px; font-weight: 600; color: #1a1a1a; margin: 0 0 3px; }
.listing-meta { font-size: 12px; color: #888; margin: 0; }

.price-pill {
    background: #ff5a5f;
    color: #fff;
    border-radius: 8px;
    padding: 3px 12px;
    font-weight: 700;
    font-size: 15px;
    display: inline-block;
    font-family: 'DM Mono', monospace;
}
.badge {
    display: inline-block;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 600;
    margin-left: 6px;
}
.badge-high { background: #fff0f0; color: #d63f3f; }
.badge-med  { background: #fff8e6; color: #b07a00; }
.badge-low  { background: #edfaf3; color: #1d9e75; }

.occ-track { background: #f0ece6; border-radius: 6px; height: 5px; width: 100%; margin-top: 6px; }
.section-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #aaa;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
QUANT_COLS = ["month","day_of_week","lead_days","competition","rooms",
              "review_score","is_holiday","has_parking","has_pool"]
QUAL_COLS  = ["event","location","district_type","property_type"]
FEATURES   = QUANT_COLS + QUAL_COLS

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
DOW_NAMES   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

# ─── Session state ────────────────────────────────────────────────────────────
if "listings"     not in st.session_state: st.session_state.listings     = []
if "model_loaded" not in st.session_state: st.session_state.model_loaded = False

# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    with open("airbnb_models.pkl", "rb") as f:
        m = pickle.load(f)
    return m["price_mdl"], m["occ_mdl"], m["X_test"], m["y_price_test"], m["y_occ_test"]

try:
    price_mdl, occ_mdl, X_test, y_price_test, y_occ_test = load_models()
    model_ok = True
except FileNotFoundError:
    model_ok = False


# ─── Helpers ──────────────────────────────────────────────────────────────────
def make_row(month, dow, event, lead, location, rooms, prop_type,
             review=4.2, parking=0, pool=0):
    info = LOCATIONS[location]
    return pd.DataFrame([{
        "month":         month,
        "day_of_week":   dow,
        "lead_days":     lead,
        "competition":   info["competition"],
        "rooms":         rooms,
        "review_score":  review,
        "is_holiday":    0,
        "has_parking":   parking,
        "has_pool":      pool,
        "event":         event,
        "location":      location,
        "district_type": info["type"],
        "property_type": prop_type,
    }])

def predict(month, dow, event, lead, location, rooms, prop_type,
            review=4.2, parking=0, pool=0):
    X     = make_row(month, dow, event, lead, location, rooms, prop_type, review, parking, pool)
    price = float(price_mdl.predict(X)[0])
    occ   = int(occ_mdl.predict(X)[0])
    prob  = float(occ_mdl.predict_proba(X)[0][1])
    return round(price, 0), occ, round(prob, 3)

def demand_badge(prob):
    if prob >= 0.75: return "badge-high", "🔴 High demand"
    if prob >= 0.50: return "badge-med",  "🟠 Moderate"
    return "badge-low", "🟢 Low demand"

def occ_color(prob):
    if prob >= 0.75: return "#ff5a5f"
    if prob >= 0.50: return "#ffb400"
    return "#00a699"

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🏠 Airbnb Predictor App")
st.caption("Dynamic pricing & booking demand prediction for Airbnb hosts in Portugal")

if not model_ok:
    st.error("No model found. Run `python train_model.py` first to generate `airbnb_models.pkl`.")
    st.stop()

st.divider()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_dash, tab_listings, tab_predict, tab_model = st.tabs([
    "📊 Dashboard", "🏠 My Listings", "🔮 Price Predictor", "🧠 Model Info"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.markdown('<p class="section-label">Market Overview · Portugal Airbnb</p>', unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def get_market_data():
        rows = []
        for loc, info in LOCATIONS.items():
            for m in range(1, 13):
                for dow in range(7):
                    price, occ, prob = predict(m, dow, "none", 14, loc, 2, "Apartment")
                    rows.append({"location": loc, "month": m, "dow": dow,
                                 "price": price, "prob": prob, "type": info["type"]})
        return pd.DataFrame(rows)


    mdf = get_market_data()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="metric-card"><p class="metric-num">€{mdf["price"].mean():.0f}</p><p class="metric-lbl">Avg nightly price</p></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="metric-card"><p class="metric-num">{mdf["prob"].mean()*100:.0f}%</p><p class="metric-lbl">Avg booking probability</p></div>', unsafe_allow_html=True)
    with k3:
        best_loc = mdf.groupby("location")["prob"].mean().idxmax()
        st.markdown(f'<div class="metric-card"><p class="metric-num" style="font-size:16px">{best_loc}</p><p class="metric-lbl">Highest demand location</p></div>', unsafe_allow_html=True)
    with k4:
        best_month = MONTH_NAMES[int(mdf.groupby("month")["prob"].mean().idxmax()) - 1]
        st.markdown(f'<div class="metric-card"><p class="metric-num">{best_month}</p><p class="metric-lbl">Peak month</p></div>', unsafe_allow_html=True)

    st.markdown("")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Booking probability by month**")
        monthly = mdf.groupby("month")["prob"].mean()
        fig, ax = plt.subplots(figsize=(6, 3))
        colors  = [occ_color(v) for v in monthly.values]
        ax.bar(MONTH_NAMES, monthly.values * 100, color=colors, edgecolor="none", width=0.7)
        ax.set_ylabel("Booking probability (%)"); ax.set_ylim(0, 100)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        st.markdown("**Average nightly price by location**")
        loc_price = mdf.groupby("location")["price"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(loc_price.index, loc_price.values, color="#ff5a5f", edgecolor="none")
        ax.set_xlabel("Avg price (€)")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("**Demand heatmap — Day of week × Month**")
    pivot = mdf.groupby(["month","dow"])["prob"].mean().unstack()
    pivot.index = [MONTH_NAMES[m-1] for m in pivot.index]
    pivot.columns = DOW_NAMES
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0.3, vmax=0.9)
    ax.set_xticks(range(7)); ax.set_xticklabels(DOW_NAMES)
    ax.set_yticks(range(12)); ax.set_yticklabels(MONTH_NAMES)
    plt.colorbar(im, ax=ax, label="Booking probability")
    for i in range(12):
        for j in range(7):
            ax.text(j, i, f"{pivot.values[i,j]:.0%}", ha="center", va="center", fontsize=7)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MY LISTINGS
# ══════════════════════════════════════════════════════════════════════════════
with tab_listings:
    st.markdown('<p class="section-label">Manage your properties</p>', unsafe_allow_html=True)

    with st.expander("➕ Add a new listing", expanded=len(st.session_state.listings) == 0):
        with st.form("add_listing"):
            c1, c2, c3 = st.columns(3)
            name      = c1.text_input("Listing name", placeholder="e.g. Sunny flat near beach")
            location  = c2.selectbox("Location", list(LOCATIONS.keys()))
            prop_type = c3.selectbox("Property type", PROPERTY_TYPES)
            rooms     = c1.slider("Rooms", 1, 5, 2)
            review    = c2.slider("Review score", 1.0, 5.0, 4.2, step=0.1)
            parking   = c3.checkbox("Has parking")
            pool      = c3.checkbox("Has pool")
            desc      = c1.text_area("Description", placeholder="Cozy place with terrace…", height=60)
            submitted = st.form_submit_button("Add listing", use_container_width=True)
            if submitted and name:
                st.session_state.listings.append({
                    "name": name, "location": location, "prop_type": prop_type,
                    "rooms": rooms, "review": review,
                    "parking": int(parking), "pool": int(pool), "desc": desc
                })
                st.success(f"'{name}' added!")
                st.rerun()

    if not st.session_state.listings:
        st.info("No listings yet — add one above.")
    else:
        view_month = st.selectbox("Preview for month", range(1, 13),
                                   format_func=lambda m: MONTH_NAMES[m-1], key="lm")
        view_event = st.selectbox("Local event", EVENTS, format_func=str.capitalize, key="le")

        for i, lst in enumerate(st.session_state.listings):
            week_prices, week_probs = [], []
            for dow in range(7):
                p, o, pr = predict(view_month, dow, view_event, 14,
                                   lst["location"], lst["rooms"], lst["prop_type"],
                                   lst["review"], lst["parking"], lst["pool"])
                week_prices.append(p); week_probs.append(pr)

            avg_price = np.mean(week_prices)
            avg_prob  = np.mean(week_probs)
            bcls, btxt = demand_badge(avg_prob)

            col_info, col_chart = st.columns([1.3, 1])

            with col_info:
                st.markdown(f"""
                <div class="listing-card">
                  <p class="listing-name">🏠 {lst['name']}</p>
                  <p class="listing-meta">📍 {lst['location']} &nbsp;·&nbsp; 🛏 {lst['rooms']} room{"s" if lst["rooms"]>1 else ""} &nbsp;·&nbsp; {lst['prop_type']}</p>
                  <p class="listing-meta">⭐ {lst['review']} &nbsp;{"🅿️" if lst["parking"] else ""} {"🏊" if lst["pool"] else ""}</p>
                  <p class="listing-meta" style="color:#666;margin-top:5px;">{lst.get('desc','')}</p>
                  <div style="margin-top:10px;">
                    <span class="price-pill">€{avg_price:.0f}/night</span>
                    <span class="badge {bcls}">{btxt}</span>
                  </div>
                  <div class="occ-track">
                    <div style="background:{occ_color(avg_prob)};width:{avg_prob*100:.0f}%;height:5px;border-radius:6px;"></div>
                  </div>
                  <p style="font-size:11px;color:#aaa;margin-top:3px;">Booking probability {avg_prob*100:.0f}%</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("🗑 Remove listing", key=f"del_{i}"):
                    st.session_state.listings.pop(i); st.rerun()

            with col_chart:
                fig, ax = plt.subplots(figsize=(4, 2.5))
                bar_colors = [occ_color(p) for p in week_probs]
                ax.bar(DOW_NAMES, week_prices, color=bar_colors, edgecolor="none", width=0.6)
                ax.set_ylabel("€/night", fontsize=9)
                ax.set_title(f"{MONTH_NAMES[view_month-1]} · {view_event.capitalize()}", fontsize=10)
                ax.spines[["top","right"]].set_visible(False)
                for b, p in zip(ax.patches, week_prices):
                    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                            f"€{p:.0f}", ha="center", va="bottom", fontsize=7)
                plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown("**📅 Monthly price calendar**")
        cal_idx = st.selectbox("Listing", range(len(st.session_state.listings)),
                                format_func=lambda i: st.session_state.listings[i]["name"],
                                key="cal_idx")
        cal_month = st.selectbox("Month", range(1, 13), index=view_month-1,
                                  format_func=lambda m: MONTH_NAMES[m-1], key="cal_month")
        cal_event = st.selectbox("Event", EVENTS, format_func=str.capitalize, key="cal_event")
        lst = st.session_state.listings[cal_idx]

        year   = 2025
        cal    = calendar.monthcalendar(year, cal_month)
        fig, ax = plt.subplots(figsize=(10, len(cal)*1.1 + 0.6))
        ax.set_xlim(0, 7); ax.set_ylim(0, len(cal)); ax.axis("off")
        for c, dn in enumerate(DOW_NAMES):
            ax.text(c+0.5, len(cal)-0.08, dn, ha="center", va="top",
                    fontsize=10, fontweight="bold", color="#555")
        for r, week in enumerate(cal):
            for c, day in enumerate(week):
                y = len(cal) - r - 1
                if day:
                    d   = pd.Timestamp(year, cal_month, day)
                    dow = d.weekday()
                    p, o, prob = predict(cal_month, dow, cal_event, 14,
                                         lst["location"], lst["rooms"], lst["prop_type"],
                                         lst["review"], lst["parking"], lst["pool"])
                    color = occ_color(prob)
                    rect  = plt.Rectangle((c+0.05, y+0.05), 0.9, 0.88,
                                          facecolor=color+"22", edgecolor=color,
                                          linewidth=1.2, zorder=1)
                    ax.add_patch(rect)
                    ax.text(c+0.15, y+0.80, str(day), fontsize=9, color="#444", va="top")
                    ax.text(c+0.50, y+0.38, f"€{p:.0f}", ha="center", fontsize=10,
                            fontweight="bold", color=color)
                    ax.text(c+0.50, y+0.14, f"{prob*100:.0f}%", ha="center",
                            fontsize=8, color="#777")
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRICE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown('<p class="section-label">Single night prediction</p>', unsafe_allow_html=True)

    cp1, cp2 = st.columns([1, 1.5])

    with cp1:
        p_location  = st.selectbox("Location",      list(LOCATIONS.keys()), key="p_loc")
        p_prop      = st.selectbox("Property type", PROPERTY_TYPES,         key="p_prop")
        p_rooms     = st.slider("Rooms",       1, 5,    2,   key="p_rooms")
        p_review    = st.slider("Review score",1.0, 5.0, 4.2, step=0.1, key="p_review")
        p_month     = st.selectbox("Month", range(1,13), format_func=lambda m: MONTH_NAMES[m-1], key="p_month")
        p_dow       = st.selectbox("Day of week", range(7), format_func=lambda d: DOW_NAMES[d], key="p_dow")
        p_event     = st.selectbox("Local event", EVENTS, format_func=str.capitalize, key="p_event")
        p_lead      = st.slider("Lead time (days)", 1, 90, 14, key="p_lead")
        p_parking   = st.checkbox("Has parking", key="p_park")
        p_pool      = st.checkbox("Has pool",    key="p_pool")

    pred_price, pred_occ, pred_prob = predict(
        p_month, p_dow, p_event, p_lead, p_location,
        p_rooms, p_prop, p_review, int(p_parking), int(p_pool)
    )
    bcls, btxt = demand_badge(pred_prob)

    with cp2:
        st.markdown(f"""
        <div style="background:#fff9f5;border:1.5px solid #ffd4c4;border-radius:16px;padding:1.5rem;margin-bottom:1rem;">
          <p style="font-size:12px;color:#aaa;margin:0;text-transform:uppercase;letter-spacing:.06em;">Suggested nightly rate</p>
          <p style="font-family:'DM Mono',monospace;font-size:52px;color:#ff5a5f;margin:4px 0;font-weight:700;">€{pred_price:.0f}</p>
          <span class="badge {bcls}" style="font-size:13px;">{btxt}</span>
          <p style="font-size:13px;color:#666;margin-top:10px;">
            Booking probability: <b>{pred_prob*100:.1f}%</b><br>
            Prediction: <b>{"✅ Likely booked" if pred_occ == 1 else "❌ Likely vacant"}</b>
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Price vs booking lead time**")
        leads   = range(1, 91)
        lprices = [predict(p_month, p_dow, p_event, l, p_location, p_rooms, p_prop,
                           p_review, int(p_parking), int(p_pool))[0] for l in leads]
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.plot(list(leads), lprices, color="#ff5a5f", lw=2)
        ax.axvline(p_lead, color="#484848", lw=1.5, ls="--",
                   label=f"{p_lead} days → €{pred_price:.0f}")
        ax.set_xlabel("Lead time (days)"); ax.set_ylabel("€/night")
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=9); plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("**Booking probability by event**")
        ev_probs = {ev: predict(p_month, p_dow, ev, p_lead, p_location,
                                p_rooms, p_prop, p_review,
                                int(p_parking), int(p_pool))[2]
                    for ev in EVENTS}
        fig, ax = plt.subplots(figsize=(5, 2.2))
        cols_ev = ["#ff5a5f" if ev == p_event else "#ffc4be" for ev in EVENTS]
        ax.barh([e.capitalize() for e in EVENTS],
                [ev_probs[e]*100 for e in EVENTS],
                color=cols_ev, edgecolor="none")
        ax.set_xlabel("Booking probability (%)")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown('<p class="section-label">Model diagnostics</p>', unsafe_allow_html=True)

    st.markdown("**Retrain model**")
    st.caption("Tune the parameters below and click **Regenerate & retrain** — nothing runs until you do.")

    # ── All controls inside a single form ────────────────────────────────────
    with st.form("retrain_form"):
        with st.expander("⚙️ Data generation & training parameters", expanded=True):

            st.markdown('<p class="section-label">Data generation</p>', unsafe_allow_html=True)
            rc1, rc2, rc3, rc4 = st.columns(4)
            n_rows           = rc1.slider("Training rows",           1000,  20000, 10000, step=500)
            noise_std        = rc2.slider("Noise std (σ)",           0.0,   12.0,  3.0,   step=0.5)
            prob_clip_min    = rc3.slider("Booking prob. clip min",  0.05,  0.45,  0.05,  step=0.05)
            prob_clip_max    = rc4.slider("Booking prob. clip max",  0.55,  0.95,  0.95,  step=0.05)
            test_size        = rc1.slider("Test split size",         0.10,  0.30,  0.25,  step=0.05)

            st.markdown('<p class="section-label">Seasonality</p>', unsafe_allow_html=True)
            sc1, sc2 = st.columns(2)
            season_mult      = sc1.slider("Seasonality strength multiplier", 0.5, 2.0, 1.5, step=0.1)
            peak_month_boost = sc2.slider("Peak month boost (Jul/Aug)",      0,   20,  0,   step=1)

            st.markdown('<p class="section-label">Demand drivers</p>', unsafe_allow_html=True)
            dc1, dc2, dc3, dc4 = st.columns(4)
            event_mult        = dc1.slider("Event impact multiplier",    0.5,  2.0,  1.0,  step=0.1)
            weekend_mult      = dc2.slider("Weekend premium multiplier", 0.5,  2.0,  1.0,  step=0.1)
            holiday_boost_val = dc3.slider("Holiday boost",              0,    30,   10,   step=1)
            competition_coef  = dc4.slider("Competition sensitivity",    -20,  -4,   -8,   step=1)

            st.markdown('<p class="section-label">Listing attributes</p>', unsafe_allow_html=True)
            la1, la2 = st.columns(2)
            review_weight    = la1.slider("Review score weight",           0,    30,   18,   step=1)
            room_price_incr  = la2.slider("Room price increment (€/room)", 5,    30,   14,   step=1)

            st.markdown('<p class="section-label">Pricing</p>', unsafe_allow_html=True)
            pc1, pc2 = st.columns(2)
            base_price_mult   = pc1.slider("Base price multiplier",    0.5,  2.0,  1.0,  step=0.1)
            lead_time_penalty = pc2.slider("Lead time penalty coef",  -0.30, -0.05, -0.10, step=0.01)

            st.markdown('<p class="section-label">Model</p>', unsafe_allow_html=True)
            n_trees = st.slider("Number of trees", 50, 500, 200, step=50)

        retrain_submitted = st.form_submit_button("🔄 Regenerate data & retrain", use_container_width=False)

    # ── Retrain logic — only runs when form is submitted ─────────────────────
        if retrain_submitted:
                with st.spinner("Generating data and retraining…"):
                    from generate_data import generate
                    from train_model import train_and_save_models

                    # 1. Generate new data using UI parameters
                    df_new = generate(
                        n=n_rows,
                        noise_std=noise_std,
                        prob_clip_min=prob_clip_min,
                        prob_clip_max=prob_clip_max,
                        season_mult=season_mult,
                        peak_month_boost=peak_month_boost,
                        event_mult=event_mult,
                        weekend_mult=weekend_mult,
                        holiday_boost_val=holiday_boost_val,
                        competition_coef=competition_coef,
                        review_weight=review_weight,
                        room_price_incr=room_price_incr,
                        base_price_mult=base_price_mult,
                        lead_time_penalty=lead_time_penalty
                    )
                    df_new.to_csv("airbnb_data.csv", index=False)

                    # 2. Train and save models
                    train_and_save_models(
                        csv_path="airbnb_data.csv", 
                        test_size=test_size, 
                        n_estimators=n_trees
                    )

                    # 3. Clear cache so Streamlit loads the new models
                    load_models.clear()
                    get_market_data.clear()

                    price_mdl, occ_mdl, X_test, y_price_test, y_occ_test = load_models()
                    st.session_state["model_version"] = st.session_state.get("model_version", 0) + 1

                    st.success(f"✅ Retrained on {n_rows:,} rows")
                    st.rerun()

    st.divider()

    # ── Evaluate on test set ──────────────────────────────────────────────────
    yp_pred_e = price_mdl.predict(X_test)
    yo_pred_e = occ_mdl.predict(X_test)
    yo_prob_e = occ_mdl.predict_proba(X_test)[:, 1]
    yp_test_e = y_price_test
    yo_test_e = y_occ_test

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(f'<div class="metric-card"><p class="metric-num">€{mean_absolute_error(yp_test_e, yp_pred_e):.2f}</p><p class="metric-lbl">Price MAE</p></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="metric-card"><p class="metric-num">{r2_score(yp_test_e, yp_pred_e):.3f}</p><p class="metric-lbl">Price R²</p></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="metric-card"><p class="metric-num">{accuracy_score(yo_test_e, yo_pred_e):.3f}</p><p class="metric-lbl">Occ. Accuracy</p></div>', unsafe_allow_html=True)
    with m4:
        from sklearn.metrics import f1_score
        st.markdown(f'<div class="metric-card"><p class="metric-num">{f1_score(yo_test_e, yo_pred_e):.3f}</p><p class="metric-lbl">Occ. F1 Score</p></div>', unsafe_allow_html=True)

    st.markdown("")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Confusion matrix — occupancy model**")
        cm   = confusion_matrix(yo_test_e, yo_pred_e)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["Not booked", "Booked"])
        fig, ax = plt.subplots(figsize=(4, 3.5))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Confusion Matrix", fontsize=11)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("**Classification report**")
        report = classification_report(yo_test_e, yo_pred_e,
                                        target_names=["Not booked","Booked"])
        st.code(report, language=None)

    with col_b:
        st.markdown("**Actual vs predicted price**")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.scatter(yp_test_e, yp_pred_e, alpha=0.2, s=10, color="#ff5a5f")
        lo = min(yp_test_e.min(), yp_pred_e.min())
        hi = max(yp_test_e.max(), yp_pred_e.max())
        ax.plot([lo,hi],[lo,hi], "k--", lw=1)
        ax.set_xlabel("Actual price (€)"); ax.set_ylabel("Predicted price (€)")
        ax.set_title(f"R² = {r2_score(yp_test_e, yp_pred_e):.3f}", fontsize=11)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("**Predicted booking probability distribution**")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.hist(yo_prob_e[yo_test_e==0], bins=30, alpha=0.6, color="#00a699", label="Not booked")
        ax.hist(yo_prob_e[yo_test_e==1], bins=30, alpha=0.6, color="#ff5a5f", label="Booked")
        ax.set_xlabel("Predicted probability"); ax.set_ylabel("Count")
        ax.set_title("Probability separation", fontsize=11)
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()
    st.markdown("**Feature importances**")
    fi_col1, fi_col2 = st.columns(2)

    def plot_importances(pipe, step_name, title, ax):
        try:
            ohe_names = (pipe.named_steps["pre"]
                         .named_transformers_["cat"]
                         .get_feature_names_out(QUAL_COLS).tolist())
        except Exception:
            try:
                ohe_names = (pipe.named_steps["preprocessor"]
                             .named_transformers_["cat"]
                             .get_feature_names_out(QUAL_COLS).tolist())
            except Exception:
                ohe_names = []
        feat_names   = QUANT_COLS + ohe_names
        importances  = pipe.named_steps[step_name].feature_importances_
        n            = min(12, len(feat_names))
        idx          = np.argsort(importances)[-n:]
        colors       = ["#ff5a5f" if importances[i] > np.median(importances) else "#ffc4be"
                        for i in idx]
        ax.barh([feat_names[i] for i in idx], importances[idx],
                color=colors, edgecolor="none")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Importance")
        ax.spines[["top","right"]].set_visible(False)

    with fi_col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        try:
            plot_importances(price_mdl, "regression", "Price model", ax)
        except Exception:
            plot_importances(price_mdl, "reg", "Price model", ax)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with fi_col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        try:
            plot_importances(occ_mdl, "classifier", "Occupancy model", ax)
        except Exception:
            plot_importances(occ_mdl, "clf", "Occupancy model", ax)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()
    st.markdown("**Model details**")
    st.markdown(f"""
    - **Algorithm:** Random Forest (scikit-learn)
    - **Price model:** Regressor → predicts nightly price in €
    - **Occupancy model:** Classifier → predicts whether a night will be booked (0/1) + booking probability
    - **Features:** {', '.join(QUANT_COLS + QUAL_COLS)}
    - **Training data:** 10,000 synthetic rows · 18 Portuguese districts
    - **Base Train/test split:** 75% / 25%
    """)