import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from io import BytesIO
import seaborn as sns
from pylab import rcParams
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
import geopy.geocoders
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import folium
from collections import Counter
from folium.features import DivIcon
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import CategoricalNB
import io
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


# Custom CSS styles
st.markdown(
    """
    <style>
    body {
        background-color: white;
        color: grey;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2F5496;
    }
    .header {
        color: #2F5496;
    }
    .text {
        font-family: Arial, sans-serif;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


header = st.container()

#@st.cache_data
#def get_data():
#    df_crime_num_2 = pd.read_csv("data/crime_data.csv")
#    return df_crime_num_2

with header:
    st.title('The NZ Police Victimisation Dataset')
    st.write("""
    Crime is a hot topic at the moment, with these models we will see if there were any insights we could gain from analysing crime data from NZ police website.
    The initial data pulled is from 2019-2022 and contained almost one millions lines of information.
    """)

dataset = st.expander('The Data', expanded=False)
with dataset:
    st.write("""The Data Frame:""")
    #df_crime_num_2  = get_data()
    df_crime_num_2 = pd.read_csv(r"C:\Users\Michael\streamlit\nzpolicedata\data\crime_data.csv")
    st.write(df_crime_num_2.head(5))

features = st.expander('Data Sources', expanded=False)
with features:
    #st.header('Features')
    st.write("""
    This dataframe was used by merging several datasets. Police crime data, sourced from the NZ Police
    website, Police numbers sourced from the NZ Police Annual Report and, Household Living Costs and the Unemployment
    Rate are sourced from NZ Statistics. It was then cleaned up and merged together adding several other variables.
    We will use the below variables to see if we can predict victimisations with multiple linear regression.
    """)
    st.markdown('* **Police Numbers by District (Region)**')
    st.markdown('* **Household Living Costs**')
    st.markdown('* **Unemployment Rate (%)**')
    st.markdown('* **Police District**')
#sidebar
st.sidebar.header('Select Conditions')
st.sidebar.write(""" #### The Linear Regression Model """)

model = st.container()
with model:
    st.header('Linear Regression Model')
    st.write("""
    Predicting Victimisations by selecting the variables; Police numbers by district,
    Unemployment Rate, District (region), and Household living costs.
    """)


# Select available variables
available_variables = ['District_Police_Num', 'hlc_index_amt_mean', 'Unemployment rate (%)', 'Boundary_Class']

# Define a dictionary for variable renaming
variable_mapping = {
    'District_Police_Num': 'Police Numbers by District',
    'hlc_index_amt_mean': 'Household Living Costs',
    'Unemployment rate (%)': 'Unemployment Rate (%)',
    'Boundary_Class': 'Police District'
}

# Multi-select dropdown for variable selection
selected_variables = st.sidebar.multiselect('Select Variables', available_variables, format_func=lambda x: variable_mapping.get(x, x))


#Create Model

# Encode categorical variables using one-hot encoding or label encoding if necessary

# Select available variables
available_variables = ['District_Police_Num', 'hlc_index_amt_mean', 'Unemployment rate (%)', 'Boundary_Class']

# Multi-select dropdown for variable selection
#selected_variables = st.sidebar.multiselect('Select Variables', available_variables)

if len(selected_variables) > 0:
    # Filter the data based on selected variables
    X = df_crime_num_2[selected_variables]
    y = df_crime_num_2['Victimisations']

    # Interactive sliders for adjusting test size
    test_size = st.sidebar.slider("Adjust Test Size", min_value=0.1, max_value=0.5, step=0.1, value=0.3)

    # Build and train the linear regression model
    model = LinearRegression()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r_squared = model.score(X_test, y_test)

    # Get the number of samples and features
    n = X_test.shape[0]
    p = X_test.shape[1]

    # Calculate adjusted R-squared
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

    # Analyze the coefficients
    coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})

    # Remove index column
    coefficients_without_index = coefficients.set_index('Feature')

    # Rename the feature names
    coefficients_without_index.rename(index=variable_mapping, inplace=True)

    # Print the evaluation metrics and coefficients
    st.write("Mean Squared Error:", mse)
    st.write("R-squared:", r_squared)
    st.write("Adjusted R-squared:", adjusted_r_squared)
    st.write("Coefficients:")
    st.dataframe(coefficients_without_index)



    # Scatter plot of actual vs predicted values
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, color='blue', alpha=0.6, edgecolor='black')
    ax.plot(y_test, y_test, color='red', linewidth=2)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Actual vs Predicted Values')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add the regression line
    ax.plot(y_pred, y_pred, color='red')

    # Calculate the confidence interval
    error = y_pred - y_test
    confidence_interval = 1.96 * np.std(error)  # 95% confidence interval

    # Plot the confidence interval
    ax.fill_between(y_pred, y_pred - confidence_interval, y_pred + confidence_interval, color='gray', alpha=0.3)

    # Save the figure as a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Display the saved figure in the Streamlit app
    st.image(buffer, use_column_width=True)

else:
    st.write("Please select at least one variable.")



if len(selected_variables) > 0:

    # Calculate the residuals
    residuals = y_test - y_pred

    # Create a DataFrame for residuals
    residuals_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Residuals': residuals})

    # Scatter plot of predicted values vs residuals
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, color='blue', alpha=0.6, edgecolor='black')
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals Plot')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save the figure as a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Display the saved figure in the Streamlit app
    st.image(buffer, use_column_width=True)

    # Display the residuals DataFrame
    st.write("Residuals:")
    st.dataframe(residuals_df.head(5))
else:
    st.write("")



# NAIVE BAYES SECTION... MICHAEL
st.write("# Naive Bayes Model:")

dataset2 = st.expander('The NB Data', expanded=False)
with dataset2:
    st.write("""The Naive Bayes Data Frame:""")
    #df_crime_num_2  = get_data()
    df_crime = pd.read_csv(r"C:\Users\Michael\streamlit\nzpolicedata\data\NBDataset.csv")
    df_crime = df_crime.drop("Unnamed: 0", axis=1)
    st.write(df_crime.head(5))




# Features used in the models
st.write("<b>We are showing the top 5 Districts with the Most accurate Predictions for whether the crimes in our data set are 'Serious crimes'</b> ", unsafe_allow_html=True)
st.write("<b>This has been done Using Categorical Naive Bayes and Binary Classifications</b> ", unsafe_allow_html=True)
serious_crimes_array = ["Abduction", "Harassment and Other Related Offences Against a Person", "Acts Intended to Cause Injury", "Sexual Assault and Related Offences", "Aggravated Robbery"]

st.write("<b>Serious Crimes include:</b> ", unsafe_allow_html=True)
for i in serious_crimes_array:
    st.markdown(f'* {i}')

st.write("<span style='color: blue;'><b>Please Select Some Input Variables in the left-hand pane:</b></span>", unsafe_allow_html=True)



#Naive bayes sidebar menu drop down.
# Create a dropdown menu in the sidebar
# Create a dropdown menu in the sidebar that allows multiple selections
st.sidebar.write(""" #### The Naive Bayes Model """)
selected_variables5 = st.sidebar.multiselect('Select Input variables', ['Occurrence Hour Of Day', 'Month', 'Time_Of_Day_Class','Day_Of_Week_Class'])

selected_input_vars = []
# Update the content based on the selected variables
if 'Occurrence Hour Of Day' in selected_variables5:
    selected_input_vars.append('Occurrence Hour Of Day')
if 'Month' in selected_variables5:
    selected_input_vars.append('Month')
if 'Time_Of_Day_Class' in selected_variables5:
    selected_input_vars.append('Time_Of_Day_Class')
if 'Day_Of_Week_Class' in selected_variables5:
    selected_input_vars.append('Day_Of_Week_Class')


def oversample_df(dataframe, os):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values
    
    if os:
        ros = RandomOverSampler()
        X, y= ros.fit_resample(X,y)
    data = np.hstack((X, np.reshape(y,(-1,1))))
    return data, X, y


# Calculate the Top 5 Districts Based on columns_to_include
columns_to_include = selected_input_vars
#NB Categorical calculation
def calcTopFive():
    
    target = 'Serious_Class'

    column = columns_to_include
    columns_used = column

    #Keep track of the top 5 models produced
    top_models = []
    #loop through each district 

    for territorial_authority in df_crime['Territorial Authority'].unique():
        df_ta = df_crime[df_crime['Territorial Authority'] == territorial_authority].reset_index(drop=True)
        df_nb = pd.DataFrame()
        df_nb = pd.concat([df_ta[column] for column in columns_to_include], axis=1)
        df_nb[target] = df_ta[target]
    
        column = columns_to_include
        columns_used = column

        X = df_nb[columns_to_include].values
        y = df_nb[target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Reset the index of X_train and X_test
        X_train = pd.DataFrame(X_train, columns=columns_to_include)
        X_test = pd.DataFrame(X_test, columns=columns_to_include)

        # Initialize and train the Naive Bayes model
        nb_model = CategoricalNB()
        nb_model.fit(X_train, y_train)

        # Make predictions
        y_pred = nb_model.predict(X_test)
        
        accuracy = nb_model.score(X_test, y_test)
        
        from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

        # Compute the confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Compute the predicted probabilities for class 1
        y_pred_prob = nb_model.predict_proba(X_test)[:, 1]

    
        # Record the model result
        model_result = {
            'Territorial Authority': territorial_authority,
            'Columns Used': columns_used.copy(),
            'Accuracy': accuracy,
            'Confusion Matrix': cm,
            "Classification Report": classification_report(y_test, y_pred),
            "ROC": roc_curve(y_test, y_pred_prob),
            "AUC": roc_auc_score(y_test, y_pred_prob),
            "y_test": y_test,
            "y_pred": y_pred
        }
        
        
        # Add the model result to the top_models list
        top_models.append(model_result)

        # Sort the top_models list based on accuracy in descending order
        top_models.sort(key=lambda x: x['Accuracy'], reverse=True)

        # Keep only the top 5 models
        top_models = top_models[:5]

    ###OUTPUT
    # Print the top 5 model results
    # Print the top 5 model results

    st.write("<b>The following features have been included as input features for predicting if a Crime committed is a 'Serious Crime' or not:</b> ", unsafe_allow_html=True)

    for i in columns_to_include:
        st.markdown(f'* {i}')


    st.write("# Top 5 Model Results:")

    # Display model results
    for i, model_result in enumerate(top_models):
        # Print model information
        st.header(str(i+1)+ ". " +  model_result['Territorial Authority'])
        st.subheader("Accuracy: " +  str(model_result['Accuracy']))

        # Classification Report.
        st.text(model_result['Classification Report'])
        
        # Create a Streamlit figure using `st.pyplot()`
        # Set the desired figure width and height
        figure_width = 600
        figure_height = 400

        # Create the figure with the desired figsize
        fig, ax = plt.subplots(figsize=(figure_width/80, figure_height/80))
        sns.heatmap(model_result["Confusion Matrix"], annot=True, cmap='Blues', fmt='d', ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        # Save the figure to a buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Display the figure using st.image() with the desired figure size
        st.image(buffer, width=figure_width, caption='Confusion Matrix')

        # Assuming you have the necessary variables: fpr, tpr, auc
        fpr, tpr, thresholds = model_result["ROC"]
        auc = model_result["AUC"]
        # Create a Streamlit figure using `st.pyplot()`
        # Set the desired image width and height
        image_width = 600
        image_height = 400

        # Create the plot with the desired figsize
        fig, ax = plt.subplots(figsize=(image_width/80, image_height/80))
        plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')

        # Save the plot to a buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Display the plot using st.image() with the desired image size
        st.image(buffer, width=image_width)



if len(selected_input_vars) > 0:
    calcTopFive()