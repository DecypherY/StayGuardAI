from pathlib import Path
import calendar
import json

import joblib
import pandas as pd
import streamlit as st

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "stayguard_model.joblib"
METRICS_PATH = ROOT_DIR / "outputs" / "model_metrics.json"

st.set_page_config(
    page_title="StayGuard AI | Risk Intelligence",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. LOAD MODEL AND METRICS
# ============================================================


@st.cache_resource
def load_model():
    saved_data = joblib.load(MODEL_PATH)

    return (saved_data["model"], saved_data["threshold"], saved_data["features"])


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r") as file:
        return json.load(file)


try:
    model, threshold, features = load_model()
    metrics = load_metrics()

except FileNotFoundError:
    st.error("Model or metrics file not found. " "Run python src/train.py first.")
    st.stop()


# ============================================================
# 3. CUSTOM STYLING
# ============================================================

st.markdown(
    """
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stMetric"] {
    background-color: #171C27;
    border: 1px solid #303849;
    padding: 18px;
    border-radius: 12px;
}

div[data-testid="stForm"] {
    border: 1px solid #303849;
    border-radius: 12px;
    padding: 20px;
}

.section-label {
    color: #9CA3AF;
    font-size: 13px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.main-heading {
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 5px;
}

.sub-heading {
    color: #9CA3AF;
    font-size: 15px;
    margin-bottom: 25px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 4. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏨 StayGuard AI")

    st.caption("HOTEL RISK INTELLIGENCE")

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "Select a page",
        ["Dashboard", "Risk Predictor", "Model Information"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### Model Status")

    st.success("Model loaded successfully")

    st.caption(f"Decision threshold: {threshold:.2%}")

    st.divider()

    st.caption("StayGuard AI | Machine Learning Project")


# ============================================================
# 5. CALCULATE MODEL METRICS
# ============================================================

tn, fp, fn, tp = metrics["confusion_matrix"][0] + metrics["confusion_matrix"][1]

total = tn + fp + fn + tp

accuracy = (tn + tp) / total if total else 0

precision = tp / (tp + fp) if (tp + fp) else 0

recall = tp / (tp + fn) if (tp + fn) else 0


# ============================================================
# 6. DASHBOARD PAGE
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="section-label">OVERVIEW / ANALYTICS</div>', unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-heading">Cancellation Risk Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sub-heading">'
        "Monitor model performance and understand booking cancellation risk."
        "</div>",
        unsafe_allow_html=True,
    )

    # Performance cards

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    col2.metric("Precision", f"{precision:.1%}")

    col3.metric("Recall", f"{recall:.1%}")

    col4.metric("F1 Score", f"{metrics['test_f1']:.3f}")

    st.divider()

    # Model performance

    left, right = st.columns([1.2, 1])

    with left:

        st.subheader("Confusion Matrix")

        confusion_df = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Actual: No Cancellation", "Actual: Cancellation"],
            columns=["Predicted: No Cancellation", "Predicted: Cancellation"],
        )

        st.dataframe(confusion_df, use_container_width=True)

        st.caption("Counts from the final test dataset.")

    with right:

        st.subheader("Model Performance")

        st.metric("Average Precision", f"{metrics['average_precision']:.3f}")

        st.metric("Test Accuracy", f"{accuracy:.1%}")

        st.metric("Selected Threshold", f"{threshold:.1%}")

        st.metric("Validation F1", f"{metrics['validation_f1']:.3f}")

    st.divider()

    st.subheader("Understanding the model")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**What the model does**")

        st.write(
            "Analyzes booking characteristics and estimates "
            "the likelihood score of a last-minute cancellation."
        )

    with col2:

        st.markdown("**How to use it**")

        st.write(
            "Enter booking information in the Risk Predictor "
            "to receive a model-generated risk estimate."
        )

    st.info(
        "This model was trained on historical Hotel Booking Demand "
        "data. Its predictions are experimental and should not be "
        "treated as guaranteed cancellation probabilities."
    )


# ============================================================
# 7. RISK PREDICTOR PAGE
# ============================================================

elif page == "Risk Predictor":

    st.markdown(
        '<div class="section-label">PREDICTION / BOOKING ANALYSIS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-heading">Booking Risk Predictor</div>', unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-heading">'
        "Enter booking details to estimate last-minute cancellation risk."
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # BOOKING FORM
    # --------------------------------------------------------

    with st.form("booking_form"):

        st.subheader("01 — Hotel Information")

        col1, col2 = st.columns(2)

        with col1:

            hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])

            lead_time = st.number_input("Lead Time (days)", min_value=0, value=30)

            arrival_date_year = st.selectbox("Arrival Year", [2015, 2016, 2017])

            arrival_date_month = st.selectbox(
                "Arrival Month",
                [
                    "January",
                    "February",
                    "March",
                    "April",
                    "May",
                    "June",
                    "July",
                    "August",
                    "September",
                    "October",
                    "November",
                    "December",
                ],
            )

            month_number = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ].index(arrival_date_month) + 1

            max_day = calendar.monthrange(arrival_date_year, month_number)[1]

            arrival_date_day_of_month = st.number_input(
                "Arrival Day", min_value=1, max_value=max_day, value=min(15, max_day)
            )

        with col2:

            market_segment = st.selectbox(
                "Market Segment",
                [
                    "Online TA",
                    "Offline TA/TO",
                    "Direct",
                    "Corporate",
                    "Groups",
                    "Complementary",
                    "Aviation",
                    "Undefined",
                ],
            )

            distribution_channel = st.selectbox(
                "Distribution Channel",
                ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"],
            )

            deposit_type = st.selectbox(
                "Deposit Type", ["No Deposit", "Non Refund", "Refundable"]
            )

            customer_type = st.selectbox(
                "Customer Type", ["Transient", "Transient-Party", "Contract", "Group"]
            )

        st.divider()

        st.subheader("02 — Guest Information")

        col1, col2, col3 = st.columns(3)

        with col1:

            adults = st.number_input("Adults", min_value=0, value=2)

            children = st.number_input("Children", min_value=0, value=0)

        with col2:

            babies = st.number_input("Babies", min_value=0, value=0)

            is_repeated_guest = st.selectbox(
                "Repeated Guest",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )

        with col3:

            country = (
                st.text_input(
                    "Country Code",
                    value="PRT",
                    help="Enter a three-letter country code.",
                )
                .strip()
                .upper()
            )

        st.divider()

        st.subheader("03 — Stay Information")

        col1, col2 = st.columns(2)

        with col1:

            stays_in_weekend_nights = st.number_input(
                "Weekend Nights", min_value=0, value=1
            )

            stays_in_week_nights = st.number_input("Week Nights", min_value=0, value=2)

            meal = st.selectbox("Meal Type", ["BB", "HB", "SC", "FB", "Undefined"])

        with col2:

            reserved_room_type = st.selectbox(
                "Reserved Room Type", ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
            )

            adr = st.number_input(
                "Average Daily Rate", min_value=0.0, value=100.0, step=10.0
            )

        st.divider()

        st.subheader("04 — Booking History")

        col1, col2 = st.columns(2)

        with col1:

            previous_cancellations = st.number_input(
                "Previous Cancellations", min_value=0, value=0
            )

        with col2:

            previous_bookings_not_canceled = st.number_input(
                "Previous Bookings Not Cancelled", min_value=0, value=0
            )

        st.divider()

        submitted = st.form_submit_button(
            "Analyze Booking Risk", use_container_width=True, type="primary"
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if submitted:

        if adults + children + babies == 0:

            st.error("A booking must include at least one guest.")

        elif not country:

            st.error("Please enter a country code.")

        elif len(country) != 3 or not country.isalpha():

            st.error("Please enter a valid three-letter country code.")

        else:

            input_data = {
                "hotel": hotel,
                "lead_time": lead_time,
                "arrival_date_year": arrival_date_year,
                "arrival_date_month": arrival_date_month,
                "arrival_date_day_of_month": arrival_date_day_of_month,
                "stays_in_weekend_nights": stays_in_weekend_nights,
                "stays_in_week_nights": stays_in_week_nights,
                "adults": adults,
                "children": children,
                "babies": babies,
                "meal": meal,
                "country": country,
                "market_segment": market_segment,
                "distribution_channel": distribution_channel,
                "is_repeated_guest": is_repeated_guest,
                "previous_cancellations": previous_cancellations,
                "previous_bookings_not_canceled": previous_bookings_not_canceled,
                "reserved_room_type": reserved_room_type,
                "deposit_type": deposit_type,
                "customer_type": customer_type,
                "adr": adr,
            }

            input_df = pd.DataFrame([input_data], columns=features)

            probability = model.predict_proba(input_df)[0, 1]

            prediction = int(probability >= threshold)

            st.divider()

            st.subheader("Prediction Results")

            col1, col2 = st.columns(2)

            col1.metric("Estimated Model Score", f"{probability:.1%}")

            col2.metric("Decision Threshold", f"{threshold:.1%}")

            st.progress(float(probability))

            if prediction == 1:

                st.error(
                    "HIGHER MODEL-CLASSIFIED RISK — "
                    "This booking exceeds the selected decision threshold."
                )

                st.write(
                    "Suggested action: Review the booking and consider "
                    "a routine confirmation with the customer."
                )

            else:

                st.success(
                    "LOWER MODEL-CLASSIFIED RISK — "
                    "This booking is below the selected decision threshold."
                )

                st.write("Suggested action: Continue with normal booking procedures.")

            st.caption(
                "The score is generated by a model trained on historical "
                "data. It is not a calibrated real-world probability or "
                "a guarantee of cancellation."
            )


# ============================================================
# 8. MODEL INFORMATION PAGE
# ============================================================

elif page == "Model Information":

    st.markdown(
        '<div class="section-label">TECHNICAL / MODEL DETAILS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-heading">Model Information</div>', unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-heading">'
        "Technical details and evaluation of the StayGuard AI model."
        "</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Model Architecture")

    st.write("**Algorithm:** Random Forest Classifier")

    st.write("**Number of trees:** 200")

    st.write("**Minimum samples per leaf:** 5")

    st.write("**Class balancing:** Enabled")

    st.write("**Preprocessing:** Median imputation and One-Hot Encoding")

    st.write("**Validation strategy:** Training / validation / testing split")

    st.divider()

    st.subheader("Evaluation Metrics")

    st.json(metrics)

    st.divider()

    st.subheader("Project Limitations")

    st.warning("""
        - The dataset contains historical hotel bookings from 2015–2017.
        - The target uses reservation_status_date as a proxy for
          cancellation timing.
        - The target may include no-show cases.
        - Random splitting does not fully simulate future deployment.
        - Model scores have not been established as calibrated
          real-world cancellation probabilities.
        """)

    st.caption(
        "StayGuard AI is an educational machine learning project "
        "and not a validated operational decision system."
    )
