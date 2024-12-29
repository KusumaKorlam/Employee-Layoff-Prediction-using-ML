import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is None:
        # Load the local file if no uploaded file is provided
        data = pd.read_excel("C:/Users/likhi/OneDrive/Desktop/MINI/employee_layoff_dataset_large.xlsx")
    else:
        # Read the uploaded file if provided
        data = pd.read_excel(uploaded_file)
    return data
def main(): 
    data = load_data()
    
    # Preprocessing the dataset
    current_year = 2024
    data['Joining Date'] = pd.to_datetime(data['Joining Date'], errors='coerce')
    data['Last Promotion Date'] = pd.to_datetime(data['Last Promotion Date'], errors='coerce')
    
    data['Years at Company'] = current_year - data['Joining Date'].dt.year
    data['Years Since Last Promotion'] = current_year - data['Last Promotion Date'].dt.year
    
    # Drop original date columns and irrelevant columns
    data = data.drop(['Employee Name', 'Employee ID', 'Joining Date', 'Last Promotion Date', 'Key Projects Handled', 'Skills'], axis=1)
    
    # Preserve the original values for company names, job roles, regions, and departments
    company_names_list = data['Company Name'].unique().tolist()[:15]  # Get first 15 unique company names
    job_roles_list = data['Job Role'].unique().tolist()
    regions_list = data['Region'].unique().tolist()
    departments_list = data['Department'].unique().tolist()  # Extract valid department options from dataset
    employment_status_list = data['Employment Status'].unique().tolist()  # Extract valid employment status options from dataset
    
    # Initialize the label encoder
    label_encoders = {}
    
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column].astype(str))
    
    # Handle missing values
    data.fillna(data.mean(numeric_only=True), inplace=True)
    
    # Split the dataset into features and target
    X = data.drop('Layoff Status', axis=1)
    y = data['Layoff Status']
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Streamlit App Interface
    st.title("Employee Layoff Predictor")
    
    # Load and display an open-source image
    st.image("https://img.freepik.com/free-photo/woman-carrying-cardboard-box-side-view_23-2149445765.jpg?size=626&ext=jpg&ga=GA1.1.2001254907.1729491964&semt=ais_hybrid", caption="Facing a Layoff")
    
    # Sidebar for user type selection
    user_type = st.sidebar.selectbox("Are you an Employee or HR?", ["Employee", "HR"])
    
    # Define the column names expected during scaling and prediction
    expected_columns = X.columns
    
    if user_type == "Employee":
        st.header("Employee Layoff Prediction")
        
        # Create input fields for employees
        gender = st.selectbox("Gender", ['Select', 'Male', 'Female'])  # Added 'Select' as first option
        marital_status = st.selectbox("Marital Status", ['Select', 'Single', 'Married'])  # Added 'Select'
        
        # Provide options for categorical fields dynamically from the dataset
        company_name = st.selectbox("Company Name", ['Select'] + sorted(company_names_list))  # Added 'Select'
        region = st.selectbox("Region", ['Select'] + sorted(regions_list))  # Added 'Select'
        job_role = st.selectbox("Job Role", ['Select'] + sorted(job_roles_list))  # Added 'Select'
        department = st.selectbox("Department", ['Select'] + sorted(departments_list))  # Added 'Select'
        employment_type = st.selectbox("Employment Type", ['Select', 'Full-time', 'Part-time', 'Contract'])  # Added 'Select'
        voluntary_exit = st.selectbox("Voluntary Exit", ['Select', 'Yes', 'No'])  # Added 'Select'
        employment_status = st.selectbox("Employment Status", ['Select'] + sorted(employment_status_list))  # Added 'Select'
        
        # Input fields for employee CTC and projects completed
        ctc_inr = st.number_input("Current Salary (CTC in INR)", min_value=0)
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=50)
        years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=50)
        number_of_projects = st.number_input("Number of Projects Completed in Company", min_value=0)
        
        # Check if all fields have values before proceeding
        if st.button("Predict Layoff Probability"):
            if 'Select' in [gender, marital_status, company_name, region, job_role, department, employment_type, voluntary_exit, employment_status]:
                st.warning("Please make sure all fields are filled correctly.")
            else:
                # Create a DataFrame with the input values
                input_data = pd.DataFrame([[gender, marital_status, company_name, region, job_role, department, employment_type, voluntary_exit, employment_status, years_at_company, years_since_last_promotion, number_of_projects, ctc_inr]],
                                          columns=['Gender', 'Marital Status', 'Company Name', 'Region', 'Job Role', 'Department', 
                                                   'Employment Type', 'Voluntary Exit', 'Employment Status', 'Years at Company', 
                                                   'Years Since Last Promotion', 'Number of Projects Completed', 'Current Salary (CTC in INR)'])
    
                # Encode the input data using the same label encoders used in training
                for column in input_data.columns:
                    if column in label_encoders:
                        input_data[column] = label_encoders[column].transform(input_data[column].astype(str))
    
                # Align the input data with the expected columns
                input_data = input_data.reindex(columns=expected_columns, fill_value=0)
    
                # Standardize the input features using the scaler trained earlier
                input_data_scaled = scaler.transform(input_data)
    
                # Prediction
                prediction = model.predict(input_data_scaled)
                layoff_prob = model.predict_proba(input_data_scaled)[0][1]
    
                # Performance metrics calculation (hypothetical)
                if years_at_company > 0 and number_of_projects > 0:
                    performance_score = number_of_projects / years_at_company
                else:
                    performance_score = 0
    
                if number_of_projects > 0 and ctc_inr > 0:
                    salary_to_projects_ratio = ctc_inr / number_of_projects
                else:
                    salary_to_projects_ratio = 0
    
                # Display prediction and suggestions
                st.write(f"Layoff Probability: {layoff_prob * 100:.2f}%")
                st.write(f"Performance Score: {performance_score:.2f}")
    
                # Display layoff reasons and suggestions based on layoff probability
                if layoff_prob > 0.70:
                    st.write("There is a high probability of being laid off.")
                    st.write("Reasons for Layoff:")
                    st.write("1. Low performance score.")
                    st.write("2. Lack of recent promotions.")
                    st.write("3. Low number of projects completed relative to salary.")
                elif layoff_prob > 0.55:
                    st.write("You are at risk. It is recommended to improve as soon as possible.")
                    st.write("Suggestions:")
                    st.write("1. Increase the number of projects handled.")
                    st.write("2. Upskill in your domain.")
                    st.write("3. Seek opportunities for promotion.")
                else:
                    st.write("There is a low probability of being laid off. Keep up the good work!")

    elif user_type == "HR":
        st.header("HR Dashboard")
    
        # Upload Excel file for HR data
        uploaded_file = st.file_uploader("Upload Employee Data (Excel file)", type=['xlsx'])
        
        if uploaded_file is not None:
            hr_data = load_data(uploaded_file)
            st.subheader("Employee Data Overview")
            st.write(hr_data.describe())
    
            # Check if 'Layoff Status' exists before attempting to plot
            if 'Layoff Status' not in hr_data.columns:
                st.error("The uploaded dataset must contain a 'Layoff Status' column.")
                return
    
            # Ensure 'Current Salary (CTC in INR)' column exists
            if 'Salary (Annual)' not in hr_data.columns:
                st.error("The uploaded dataset must contain a 'Current Salary (CTC in INR)' column.")
                return
            # Preprocess HR data
            hr_data['Joining Date'] = pd.to_datetime(hr_data['Joining Date'], errors='coerce')
            hr_data['Last Promotion Date'] = pd.to_datetime(hr_data['Last Promotion Date'], errors='coerce')
    
            current_year = 2024
            hr_data['Years at Company'] = current_year - hr_data['Joining Date'].dt.year
            hr_data['Years Since Last Promotion'] = current_year - hr_data['Last Promotion Date'].dt.year
            
            # Drop original date columns and irrelevant columns
            columns_to_drop = ['Joining Date', 'Last Promotion Date', 'Key Projects Handled', 'Skills']
            hr_data = hr_data.drop(columns=columns_to_drop, errors='ignore')
    
            # Ensure the same categorical columns are encoded
            for column in hr_data.select_dtypes(include=['object']).columns:
                if column in label_encoders:
                    hr_data[column] = label_encoders[column].transform(hr_data[column].astype(str))
    
            # Handle missing values
            hr_data.fillna(hr_data.mean(numeric_only=True), inplace=True)
    
            # Align the columns with the training data
            X_hr = hr_data.reindex(columns=X.columns, fill_value=0)
    
            # Standardize the features
            X_hr_scaled = scaler.transform(X_hr)
    
            # Calculate and display layoff probability
            hr_data['Layoff Probability'] = model.predict_proba(X_hr_scaled)[:, 1]

            if 'Department' in hr_data.columns and 'Department' in label_encoders:
                hr_data['Department'] = label_encoders['Department'].inverse_transform(hr_data['Department'])
            
            # The rest of your HR dashboard code continues...
            # Visualization: Count of Layoff Status
            st.subheader("Layoff Status Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x='Layoff Status', data=hr_data, ax=ax)
            st.pyplot(fig)
    
            # Visualization: Average Salary by Layoff Status
            st.subheader("Average Salary by Layoff Status")
            avg_salary = hr_data.groupby('Layoff Status')['Salary (Annual)'].mean().reset_index()
            fig2, ax2 = plt.subplots()
            sns.barplot(x='Layoff Status', y='Salary (Annual)', data=avg_salary, ax=ax2)
            st.pyplot(fig2)
    
            # Calculate and display employees at risk of being laid off
            st.subheader("Employees at Risk of Layoff")
            features = X.columns.tolist()  # Get the feature names used during training
            hr_data_filtered = hr_data[features].copy()  # Keep only the necessary columns

        # Handle any missing values
            hr_data_filtered.fillna(0, inplace=True)
            hr_data_filtered = hr_data_filtered.reindex(columns=features, fill_value=0)
            hr_data['Layoff Probability'] = model.predict_proba(X_hr_scaled)[:, 1]

            at_risk_employees = hr_data[hr_data['Layoff Probability'] > 0.75].copy()
            if at_risk_employees.empty:
                st.write("No employees are currently at risk of being laid off.")
            else:
                st.write("List of Employees at Risk:")
                # Define the columns you want to display
                display_columns = ['Employee ID', 'Employee Name', 'Contact Number', 'Department',
                                   'Salary (Annual)', 'Layoff Probability']
                
                # Check which columns are available in the DataFrame
                available_columns = [col for col in display_columns if col in at_risk_employees.columns]
    
                if not available_columns:
                    st.warning("No relevant employee information is available in the dataset.")
                else:
                    st.dataframe(at_risk_employees[available_columns])
                    at_risk_employees = hr_data[hr_data['Layoff Probability'] > 0.75]
        
                if at_risk_employees.empty:
                    st.write("No employees are currently at risk of being laid off.")
                else:
                    st.write("List of Employees at Risk:")
                    csv = at_risk_employees[available_columns].to_csv(index=False).encode('utf-8')
                    st.download_button("Download At-Risk Employees Data", csv, "at_risk_employees.csv", "text/csv")

                # Visualization: Layoff Probability Distribution for at-risk employees
                st.subheader("Layoff Probability Distribution for At-Risk Employees")
                fig3, ax3 = plt.subplots()
                sns.histplot(at_risk_employees['Layoff Probability'], bins=10, ax=ax3)
                st.pyplot(fig3)
                                # Visualization: Scatter Plot for Salary vs Performance Score
if __name__=="__main__":
    main()

