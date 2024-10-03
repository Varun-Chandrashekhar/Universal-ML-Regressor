# Import Libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import io
import zipfile
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import NotFittedError
import warnings
warnings.filterwarnings('ignore')

# Set Streamlit page configuration
st.set_page_config(page_title='Universal ML Regressor', layout='wide')

# Create a title and a sub-title
st.title('Universal ML Regressor')
st.write('An advanced machine learning web app for regression applications.')

st.write("Import a CSV dataset. The CSV file must have the target variable in the rightmost column and the remaining predictors to the left.")

# Sidebar for user inputs
st.sidebar.header('User Input Parameters')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/BostonHousing.csv)
""")

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

split_ratio = st.sidebar.slider('Test Set Percentage', 10, 50, 20, 5)
split_ratio = split_ratio / 100

# Model hyperparameters
st.sidebar.subheader('Model Hyperparameters')

# Parameters for Polynomial Regression
poly_degree = st.sidebar.slider('Polynomial Degree', 2, 5, 2)

# Parameters for Random Forest
n_estimators = st.sidebar.slider('Random Forest: Number of Trees (n_estimators)', 10, 200, 100, 10)

# Parameters for KNN
n_neighbors = st.sidebar.slider('KNN: Number of Neighbors (n_neighbors)', 1, 20, 5, 1)

# Parameters for SVR
svr_kernel = st.sidebar.selectbox('SVR: Kernel', ('linear', 'poly', 'rbf', 'sigmoid'))

if uploaded_file is not None:
    # Read the dataset
    dataset = pd.read_csv(uploaded_file)

    # Display dataset
    st.subheader('Dataset')
    st.write('Shape of dataset:', dataset.shape)
    st.dataframe(dataset.head())

    # Data Preprocessing
    st.subheader('Data Preprocessing')

    # Handle missing values
    if dataset.isnull().values.any():
        st.write('Missing values detected. Filling missing values with column mean.')
        dataset = dataset.fillna(dataset.mean())

    # Encode categorical variables
    if dataset.select_dtypes(include=['object']).shape[1] > 0:
        st.write('Categorical variables detected. Encoding categorical variables.')
        dataset = pd.get_dummies(dataset)

    # Feature Scaling option
    scaling_option = st.selectbox('Feature Scaling', ('None', 'Standardization', 'Normalization'))

    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    if scaling_option == 'Standardization':
        scaler = StandardScaler()
        dataset.iloc[:, :-1] = scaler.fit_transform(dataset.iloc[:, :-1])
    elif scaling_option == 'Normalization':
        scaler = MinMaxScaler()
        dataset.iloc[:, :-1] = scaler.fit_transform(dataset.iloc[:, :-1])

    # Split data into features and target
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    # Data Visualization
    st.subheader('Data Visualization')

    # Correlation Heatmap
    st.write('Correlation Heatmap:')
    plt.figure(figsize=(10, 8))
    sns.heatmap(dataset.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    st.pyplot(plt)

    # Feature Importance Placeholder
    feature_importance = {}

    # Splitting the dataset into Training and Testing Set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=0)

    # Initialize dictionaries to store models and results
    models = {}
    results = {}
    trained_models = []

    # List of regression models
    model_list = [
        'Linear Regression',
        'Lasso Regression',
        'Ridge Regression',
        'Elastic Net',
        'Polynomial Regression',
        'Support Vector Machine',
        'K-Nearest Neighbors',
        'Decision Tree',
        'Random Forest',
        'Extra Trees Regressor',
        'Gradient Boosting Regressor',
        'AdaBoost Regressor',
        'XGBoost',
        # 'CatBoost Regressor'  # Uncomment if CatBoost is installed
    ]

    # Train each model
    for model_option in model_list:
        try:
            if model_option == 'Linear Regression':
                from sklearn.linear_model import LinearRegression
                regressor = LinearRegression()
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                feature_importance[model_option] = regressor.coef_

            elif model_option == 'Lasso Regression':
                from sklearn.linear_model import Lasso
                regressor = Lasso()
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                feature_importance[model_option] = regressor.coef_

            elif model_option == 'Ridge Regression':
                from sklearn.linear_model import Ridge
                regressor = Ridge()
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                feature_importance[model_option] = regressor.coef_

            elif model_option == 'Elastic Net':
                from sklearn.linear_model import ElasticNet
                regressor = ElasticNet()
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                feature_importance[model_option] = regressor.coef_

            elif model_option == 'Polynomial Regression':
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.linear_model import LinearRegression
                poly_reg = PolynomialFeatures(degree=poly_degree)
                X_poly_train = poly_reg.fit_transform(X_train)
                X_poly_test = poly_reg.transform(X_test)
                regressor = LinearRegression()
                regressor.fit(X_poly_train, y_train)
                y_pred = regressor.predict(X_poly_test)
                # Coefficients for polynomial features
                feature_importance[model_option] = regressor.coef_

            elif model_option == 'Support Vector Machine':
                from sklearn.svm import SVR
                regressor = SVR(kernel=svr_kernel)
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                # SVR doesn't have feature importances

            elif model_option == 'K-Nearest Neighbors':
                from sklearn.neighbors import KNeighborsRegressor
                regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                # KNN doesn't have feature importances

            elif model_option == 'Decision Tree':
                from sklearn.tree import DecisionTreeRegressor
                regressor = DecisionTreeRegressor(random_state=0)
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                feature_importance[model_option] = regressor.feature_importances_

            elif model_option == 'Random Forest':
                from sklearn.ensemble import RandomForestRegressor
                regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                feature_importance[model_option] = regressor.feature_importances_

            elif model_option == 'Extra Trees Regressor':
                from sklearn.ensemble import ExtraTreesRegressor
                regressor = ExtraTreesRegressor(n_estimators=100, random_state=0)
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                feature_importance[model_option] = regressor.feature_importances_

            elif model_option == 'Gradient Boosting Regressor':
                from sklearn.ensemble import GradientBoostingRegressor
                regressor = GradientBoostingRegressor(random_state=0)
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                feature_importance[model_option] = regressor.feature_importances_

            elif model_option == 'AdaBoost Regressor':
                from sklearn.ensemble import AdaBoostRegressor
                regressor = AdaBoostRegressor(random_state=0)
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                # AdaBoost may have feature importances
                feature_importance[model_option] = regressor.feature_importances_

            elif model_option == 'XGBoost':
                from xgboost import XGBRegressor
                regressor = XGBRegressor()
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                feature_importance[model_option] = regressor.feature_importances_

            # Uncomment the following if CatBoost is installed
            # elif model_option == 'CatBoost Regressor':
            #     from catboost import CatBoostRegressor
            #     regressor = CatBoostRegressor(verbose=0)
            #     regressor.fit(X_train, y_train)
            #     y_pred = regressor.predict(X_test)
            #     feature_importance[model_option] = regressor.feature_importances_

            # Evaluate the model
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # Store the results
            results[model_option] = {'R-squared': r2, 'MSE': mse, 'MAE': mae}
            models[model_option] = regressor
            trained_models.append(model_option)
        except Exception as e:
            # If the model fails, store N/A
            results[model_option] = {'R-squared': 'N/A', 'MSE': 'N/A', 'MAE': 'N/A'}
            models[model_option] = None

    # Display the results in a table
    st.subheader('Model Evaluation Results')
    results_df = pd.DataFrame(results).T
    results_df = pd.DataFrame(results).T
    
    results_df.replace('N/A', np.nan, inplace=True)

    # Display the results and highlight the maximum values
    st.subheader('Model Evaluation Results')
    st.dataframe(results_df.style.highlight_max(axis=0, subset=['R-squared', 'MSE', 'MAE'], color='lightgreen'))
    st.dataframe(results_df.style.highlight_max(axis=0))

    # Model Comparison Charts
    st.subheader('Model Comparison Charts')
    metrics = ['R-squared', 'MSE', 'MAE']
    for metric in metrics:
        plt.figure()
        plt.title(f'Model Comparison - {metric}')
        sns.barplot(x=results_df.index, y=results_df[metric])
        plt.xticks(rotation=45)
        st.pyplot(plt)

    # Feature Importance Visualization
    st.subheader('Feature Importance')
    selected_feature_model = st.selectbox('Select Model for Feature Importance', trained_models)
    if selected_feature_model in feature_importance:
        importances = feature_importance[selected_feature_model]
        indices = np.argsort(importances)[::-1]
        names = [X.columns[i] for i in indices]

        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importance - {selected_feature_model}')
        plt.bar(range(X.shape[1]), importances[indices])
        plt.xticks(range(X.shape[1]), names, rotation=90)
        st.pyplot(plt)
    else:
        st.write(f'Feature importance not available for {selected_feature_model}.')

    # Allow the user to select a model to download
    st.subheader('Select a Model to Download')
    selected_model = st.selectbox('Select Model', trained_models)

    if selected_model:
        regressor = models[selected_model]

        # Save the trained model to a bytes object
        model_buffer = io.BytesIO()
        pickle.dump(regressor, model_buffer)
        model_buffer.seek(0)

        # Create the code template based on the selected model
        code_template = f'''
import pickle
import pandas as pd

# Load the trained model
with open('regressor.pkl', 'rb') as file:
    regressor = pickle.load(file)

# Load your input data
data = pd.read_csv('input.csv')  # Replace 'input.csv' with your data file

# Data Preprocessing
# Handle missing values
data = data.fillna(data.mean())

# Encode categorical variables
data = pd.get_dummies(data)

# Ensure the features match the training data
from sklearn.model_selection import train_test_split
X = data.values  # Ensure this matches the training data format

'''

        if scaling_option == 'Standardization':
            code_template += '''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
'''
        elif scaling_option == 'Normalization':
            code_template += '''
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
'''

        if selected_model == 'Polynomial Regression':
            code_template += f'''
# Prepare the data with Polynomial Features
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree={poly_degree})
X = poly_reg.fit_transform(X)
'''

        code_template += '''
# Make predictions
predictions = regressor.predict(X)

# Save predictions to a CSV file
pd.DataFrame(predictions, columns=['Predictions']).to_csv('predictions.csv', index=False)
'''

        # Create requirements.txt content
        requirements = '''
numpy
pandas
scikit-learn
xgboost
seaborn
matplotlib
'''

        # Create Procfile content
        procfile_content = '''
web: streamlit run app.py
'''

        # Create a ZIP file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
            # Add model file
            zip_file.writestr('regressor.pkl', model_buffer.getvalue())
            # Add code template
            zip_file.writestr('predict.py', code_template)
            # Add requirements.txt
            zip_file.writestr('requirements.txt', requirements)
            # Add Procfile
            zip_file.writestr('Procfile', procfile_content)

        zip_buffer.seek(0)

        # Provide the ZIP file for download
        st.subheader('Download Your Model and Code')
        st.write('Click the button below to download a ZIP file containing your trained model, prediction code, and supporting files.')

        # Generate download link for ZIP file
        st.download_button(
            label='Download ZIP File',
            data=zip_buffer,
            file_name='model_and_code.zip',
            mime='application/zip'
        )

        # Instructions for using the files
        st.subheader('Instructions for Using the Downloaded Files')
        st.markdown('''
1. **Extract** the contents of the ZIP file.
2. **Place** your input data file (e.g., `input.csv`) in the same directory.
3. **Install** the required packages using `pip install -r requirements.txt`.
4. **Run** the prediction script using `python predict.py`.
5. **Check** the `predictions.csv` file for the output.
''')

else:
    st.info('Awaiting CSV file upload. Please upload a file to proceed.')

# Footer
st.write('---')
st.write('Developed by Varun Chandrashekhar')
