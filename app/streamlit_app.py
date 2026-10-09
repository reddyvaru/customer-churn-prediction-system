import os
import requests
import streamlit as st


try:
    API_URL = st.secrets["CHURN_API_URL"]
except (KeyError, FileNotFoundError):
    API_URL = os.getenv(
        "CHURN_API_URL",
        "http://127.0.0.1:8000/predict"
    )

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction & Retention Intelligence")
st.write(
    "Enter customer information to estimate churn probability "
    "and identify customers who may need retention support."
)

st.info(
    "The model estimates churn probability. This is not the "
    "probability that its prediction is correct."
)

with st.form("customer_form"):

    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=72, value=6
        )
        monthly_charges = st.number_input(
            "Monthly Charges", min_value=0.0, value=90.0
        )
        total_charges = st.number_input(
            "Total Charges", min_value=0.0, value=540.0
        )
        cltv = st.number_input(
            "Customer Lifetime Value (CLTV)",
            min_value=0,
            value=4000
        )

    with col2:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox(
            "Senior Citizen", ["No", "Yes"]
        )
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

    with col3:
        internet_service = st.selectbox(
            "Internet Service", ["Fiber optic", "DSL", "No"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )

    st.subheader("Internet Services")

    col4, col5 = st.columns(2)

    with col4:
        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )

    with col5:
        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        use_container_width=True
    )


if submitted:

    payload = {
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "cltv": cltv,
        "partner": partner,
        "dependents": dependents,
        "senior_citizen": senior_citizen,
        "gender": gender,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "contract": contract,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method
    }

    try:
        with st.spinner("Analyzing customer information..."):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

        st.divider()
        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Churn Probability",
                f"{result['churn_probability_percent']:.2f}%"
            )

        with col2:
            st.metric(
                "Predicted Outcome",
                result["predicted_class"]
            )

        with col3:
            st.metric(
                "Risk Level",
                result["risk_level"]
            )

        st.progress(
            min(
                max(result["churn_probability_percent"] / 100, 0.0),
                1.0
            )
        )

        st.caption(
            f"Decision threshold: "
            f"{result['threshold_used']:.2f} "
            f"({result['threshold_used'] * 100:.0f}%)"
        )

        if result["predicted_value"] == 1:
            st.warning(
                "This customer has been flagged for potential churn. "
                "Consider reviewing the customer's needs before "
                "taking retention action."
            )
        else:
            st.success(
                "This customer is below the current churn threshold."
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the prediction API. "
            "Make sure FastAPI is running on port 8000."
        )

    except requests.exceptions.RequestException as error:
        st.error(f"Prediction request failed: {error}")