Overview
This project involves building a regression model to predict the prices of used cars based on various features. The goal is to develop a model that can accurately predict the prices of used cars based on historical data. The project consists of three main components: data preprocessing, model development, and deployment.

Dataset
The dataset used for this project consists of 188533 rows and 13 columns. The columns include:

Brand: 57 unique brands
Model: 1897 unique models
Model Year: 34 unique model years
Mileage: The total distance traveled by the car
Fuel Type: 9 unique fuel types
Transmission: 52 unique transmission types
Exterior Color: 319 unique exterior colors
Interior Color: 156 unique interior colors
Accident: Whether the car has been in an accident
Clean Title: Whether the car has a clean title
Horsepower: The horsepower of the car's engine
Engine Size: The size of the car's engine
Cylinders: The number of cylinders in the car's engine
Price: The price of the car
The dataset is a mix of categorical and numerical features, which requires careful preprocessing to ensure that the model can effectively learn from the data.

Models Used
Two regression models were used in this project:

CatBoost Regressor: This is a gradient boosting model that is known for its high performance and ease of use. It is particularly well-suited for handling categorical features in the dataset.
KNN Regressor: This is a simple yet effective model that uses the k-nearest neighbors algorithm to make predictions.
After training and evaluating both models, the CatBoost Regressor turned out to be the best performer, with an RMSE of 63817.83993 on the test data in the Kaggle competition.

Streamlit App and Database Integration
To deploy the model and collect user input data, a Streamlit app was built. The app allows users to input the features of a used car and predict its price. The app is connected to two databases:

MongoDB: This is used to store the user input data and the corresponding predictions.
Snowflake: This is a cloud-based data warehouse that is used to store the user input data and the predictions.
The Streamlit app provides a simple and intuitive interface for users to input the features of a used car and get a predicted price. The app also stores the user input data and the predictions in the MongoDB and Snowflake databases for future reference and analysis.
