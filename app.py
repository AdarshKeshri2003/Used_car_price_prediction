
import requests
import streamlit as st


# --------------------------------------------------
# App configuration
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="centered",
)

st.title("🚗 Used Car Price Prediction")
st.write("Enter the car details to predict its selling price.")


# --------------------------------------------------
# Car details form
# --------------------------------------------------

with st.form("car_prediction_form"):

    st.subheader("Car Information")

    name = st.text_input(
        "Car name",
        placeholder="Example: Maruti Swift Dzire VDI",
        help="The first word is used as the brand.",
    )

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input(
            "Manufacturing year",
            min_value=1900,
            max_value=2100,
            value=2018,
            step=1,
        )

        km_driven = st.number_input(
            "Kilometers driven",
            min_value=0,
            value=50000,
            step=1000,
        )

        fuel = st.selectbox(
            "Fuel type",
            ["Diesel", "Petrol", "CNG", "LPG", "Electric"],
        )

        seller_type = st.selectbox(
            "Seller type",
            ["Individual", "Dealer", "Trustmark Dealer"],
        )

        transmission = st.selectbox(
            "Transmission",
            ["Manual", "Automatic"],
        )

        owner = st.selectbox(
            "Owner",
            [
                "First Owner",
                "Second Owner",
                "Third Owner",
                "Fourth & Above Owner",
                "Test Drive Car",
            ],
        )

    with col2:
        mileage = st.number_input(
            "Mileage (km/l)",
            min_value=0.0,
            value=20.0,
            step=0.1,
        )

        engine = st.number_input(
            "Engine capacity (CC)",
            min_value=0.0,
            value=1248.0,
            step=50.0,
        )

        max_power = st.number_input(
            "Maximum power (bhp)",
            min_value=0.0,
            value=74.0,
            step=1.0,
        )

        torque_value = st.number_input(
            "Torque value",
            min_value=0.0,
            value=190.0,
            step=1.0,
            help="Enter the numeric torque value, without units.",
        )

        seats = st.number_input(
            "Number of seats",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
        )

    submitted = st.form_submit_button(
        "Predict Car Price",
        use_container_width=True,
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submitted:

    if not name.strip():
        st.error("Please enter the car name.")

    else:
        # Match the notebook's brand extraction:
        # car["brand"] = car["name"].str.split().str[0]
        brand = name.strip().split()[0]

        car_data = {
            "year": int(year),
            "km_driven": float(km_driven),
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "mileage": float(mileage),
            "engine": float(engine),
            "max_power": float(max_power),
            "torque_value": float(torque_value),
            "seats": float(seats),
            "brand": brand,
        }

        try:
            with st.spinner("Predicting car price..."):
                response = requests.post(
                    API_URL,
                    json=car_data,
                    timeout=30,
                )

            if response.ok:
                result = response.json()
                price = result["predicted_price"]

                st.success("Prediction completed!")

                st.metric(
                    label="Predicted Selling Price",
                    value=f"₹{price:,.2f}",
                )

                st.caption(
                    "This is a model estimate, not a guaranteed selling price."
                )

            else:
                try:
                    error_detail = response.json().get(
                        "detail",
                        response.text,
                    )
                except ValueError:
                    error_detail = response.text

                st.error(
                    f"API error ({response.status_code}): "
                    f"{error_detail}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to FastAPI. "
                "Start the backend first."
            )

        except requests.exceptions.Timeout:
            st.error("The API request timed out.")

        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")