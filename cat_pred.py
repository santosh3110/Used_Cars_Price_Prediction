import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import snowflake.connector
from dotenv import load_dotenv
import os

load_dotenv()

uri = os.getenv("uri")
client = MongoClient(uri, server_api=ServerApi('1'))
db=client['used_cars_price']
collection= db["price_pred"]


# Load dataset for dynamic options
@st.cache_data
def load_data():
    df_feat = pd.read_csv("features.csv")  
    return df_feat

df_feat = load_data()

# Extract unique values for dropdowns
brands = df_feat["brand"].unique().tolist()
models = df_feat["model"].unique().tolist()
fuel_types = df_feat["fuel_type"].unique().tolist()
transmissions = df_feat["transmission"].unique().tolist()
ext_colors = df_feat["ext_col"].unique().tolist()
int_colors = df_feat["int_col"].unique().tolist()

@st.cache_resource
def load_model_and_scaler():
    with open("cat_model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)  # Load the trained StandardScaler
    return model, scaler

model, scaler = load_model_and_scaler()

def preprocessing_input_data(data):
    df=pd.DataFrame([data])
    num_cols = ["model_year", "milage", "horsepower", "engine_size", "cylinders"]
    df[num_cols] = scaler.transform(df[num_cols])
    return df

def predict_data(data):
    df=preprocessing_input_data(data)
    prediction=model.predict(df)
    return prediction[0]

# Store data in Snowflake
def store_data_in_snowflake(data):
    try:
        conn = snowflake.connector.connect(
            account=os.getenv("account"),
            user=os.getenv("user"),
            password=os.getenv("password"),
            role="ACCOUNTADMIN",
            warehouse="COMPUTE_WH",
            database="My_USED_CAR_DB",
            schema="PUBLIC"
        )
        cur = conn.cursor()
        
        # Prepare SQL query for insertion
        insert_query = f"""
        INSERT INTO used_car_prices (brand, model, model_year, mileage, fuel_type, transmission, 
                                     ext_col, int_col, accident, clean_title, horsepower, engine_size, 
                                     cylinders, predicted_price) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        values = (
            data["brand"], data["model"], data["model_year"], data["milage"], data["fuel_type"],
            data["transmission"], data["ext_col"], data["int_col"], data["accident"],
            data["clean_title"], data["horsepower"], data["engine_size"], data["cylinders"], 
            data["predicted price"]
        )
        
        cur.execute(insert_query, values)
        conn.commit()
        cur.close()
        conn.close()
        st.success("✅ Data successfully stored in **Snowflake**!")
    except Exception as e:
        st.error(f"❌ Error storing data in Snowflake: {e}")


def main():
    st.title("🚗 Used Car Price Prediction")
    st.write("Fill in the details to predict the price of the used car:")
    
   # Brand selection
    brand = st.selectbox("Brand", brands)

    # Filter models based on selected brand
    models_filtered = df_feat[df_feat["brand"] == brand]["model"].unique().tolist()
    model = st.selectbox("Model", models_filtered)

    # Filter other attributes based on brand & model
    filtered_data = df_feat[(df_feat["brand"] == brand) & (df_feat["model"] == model)]

    # Set min/max values dynamically
    model_year = st.slider("Model Year", 
                        min_value=int(filtered_data["model_year"].min()), 
                        max_value=int(filtered_data["model_year"].max()), 
                        value=int(filtered_data["model_year"].median()))

    milage = st.number_input("Mileage (in km)", 
                            min_value=int(filtered_data["milage"].min()), 
                            max_value=int(filtered_data["milage"].max()), 
                            value=int(filtered_data["milage"].median()), 
                            step=1000)

    fuel_type = st.selectbox("Fuel Type", filtered_data["fuel_type"].unique().tolist())
    transmission = st.selectbox("Transmission", filtered_data["transmission"].unique().tolist())
    ext_col = st.selectbox("Exterior Color", filtered_data["ext_col"].unique().tolist())
    int_col = st.selectbox("Interior Color", filtered_data["int_col"].unique().tolist())

    accident = st.radio("Has the car been in an accident?", ["Yes", "No"])
    clean_title = st.radio("Does it have a clean title?", ["Yes", "No"])

    horsepower = st.slider("Horsepower", 
                            min_value=int(filtered_data["horsepower"].min()), 
                            max_value=int(filtered_data["horsepower"].max()), 
                            value=int(filtered_data["horsepower"].median()), 
                            step=10)

    engine_size = st.number_input("Engine Size (L)", 
                                min_value=float(filtered_data["engine_size"].min()), 
                                max_value=float(filtered_data["engine_size"].max()), 
                                value=float(filtered_data["engine_size"].median()), 
                                step=0.1)

    cylinders = st.slider("Cylinders", 
                        min_value=int(filtered_data["cylinders"].min()), 
                        max_value=int(filtered_data["cylinders"].max()), 
                        value=int(filtered_data["cylinders"].median()), 
                        step=1)




# Prediction button
    if st.button("🔮 Predict Price"):
        input_data = {
            "brand": brand,
            "model": model,
            "model_year": model_year,
            "milage": milage,  
            "fuel_type": fuel_type,
            "transmission": transmission,
            "ext_col": ext_col,
            "int_col": int_col,
            "accident": accident,
            "clean_title": clean_title,
            "horsepower": horsepower,
            "engine_size": engine_size,
            "cylinders": cylinders,
        }

        predicted_price = predict_data(input_data)
        input_data["predicted price"]=predicted_price
        collection.insert_one(input_data)
        store_data_in_snowflake(input_data)
        st.success(f"💰 Estimated Price: **${predicted_price:,.2f}**")


if __name__ == "__main__":
    main()