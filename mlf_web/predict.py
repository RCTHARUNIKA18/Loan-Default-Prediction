import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_model_and_features():
    """
    Load the trained model and feature information
    """
    try:
        model = joblib.load('loan_default_model.pkl')
        feature_info = joblib.load('feature_info.pkl')
        return model, feature_info
    except FileNotFoundError:
        print("Model files not found. Please run model.py first to train the model.")
        sys.exit(1)

def prepare_loan_application(application_data, feature_info):
    """
    Prepare a loan application for prediction
    """
    # Create a DataFrame for the application
    if isinstance(application_data, dict):
        # Convert single application dict to DataFrame
        application_df = pd.DataFrame([application_data])
    else:
        # Already a DataFrame
        application_df = application_data.copy()
    
    # Extract feature lists
    numeric_features = feature_info['numeric_features']
    categorical_features = feature_info['categorical_features']
    
    # Handle missing features
    for feature in numeric_features + categorical_features:
        if feature not in application_df.columns:
            application_df[feature] = np.nan
    
    # Clean data similar to training
    # Convert string percentages to float
    for col in application_df.columns:
        if application_df[col].dtype == object:
            # Try to convert percentage strings to float
            try:
                # Check if column contains percentage values
                if application_df[col].str.contains('%').any():
                    application_df[col] = application_df[col].str.rstrip('%').astype(float) / 100
            except:
                pass
    
    # Convert categorical Yes/No columns to binary
    for col in application_df.columns:
        if application_df[col].dtype == object:
            try:
                application_df[col] = application_df[col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'YES': 1, 'NO': 0})
            except:
                pass
    
    # Handle date columns if they exist
    date_columns = ['DOB']
    for col in date_columns:
        if col in application_df.columns:
            try:
                application_df[col] = pd.to_datetime(application_df[col])
                # Extract year, month from date
                application_df[f'{col}_year'] = application_df[col].dt.year
                application_df[f'{col}_month'] = application_df[col].dt.month
                # Drop original date column
                application_df = application_df.drop(col, axis=1)
            except:
                pass
    
    # Calculate derived features that might be useful
    if 'LoanAmount' in application_df.columns and 'Annual Income' in application_df.columns:
        application_df['Loan_to_Income_Ratio'] = application_df['LoanAmount'] / application_df['Annual Income']
    
    if 'Credit Score' in application_df.columns and 'Existing Loans' in application_df.columns:
        application_df['Credit_Score_to_Loans'] = application_df['Credit Score'] / (application_df['Existing Loans'] + 1)
    
    return application_df

def predict_default_risk(application_data):
    """
    Predict default risk for a loan application
    """
    # Load model and feature information
    model, feature_info = load_model_and_features()
    
    # Prepare application data
    prepared_data = prepare_loan_application(application_data, feature_info)
    
    # Make prediction
    try:
        # Probability of default (0 = default, 1 = no default)
        default_probability = model.predict_proba(prepared_data)[0]
        
        # Get approval probability (class 1)
        approval_probability = default_probability[1]
        
        # Determine risk category
        if approval_probability >= 0.8:
            risk_category = "Low Risk"
        elif approval_probability >= 0.6:
            risk_category = "Moderate Risk"
        elif approval_probability >= 0.4:
            risk_category = "High Risk" 
        else:
            risk_category = "Very High Risk"
        
        return {
            'approval_probability': approval_probability,
            'risk_category': risk_category,
            'approval_prediction': 'Approved' if approval_probability > 0.5 else 'Declined'
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return {
            'error': str(e),
            'approval_prediction': 'Unable to predict'
        }

def analyze_rejection_factors(application_data):
    """
    Analyze factors that might lead to rejection
    """
    rejection_factors = []
    
    # Credit score check
    if 'Credit Score' in application_data:
        credit_score = application_data['Credit Score']
        if credit_score < 700:
            rejection_factors.append(f"Low credit score ({credit_score})")
    
    # Loan-to-income ratio check
    if 'Annual Income' in application_data and 'LoanAmount' in application_data:
        loan_amount = application_data['LoanAmount']
        income = application_data['Annual Income']
        if income > 0:
            lti_ratio = loan_amount / income
            if lti_ratio > 0.5:
                rejection_factors.append(f"High loan-to-income ratio ({lti_ratio:.2f})")
    
    # Existing loans check
    if 'Existing Loans' in application_data:
        existing_loans = application_data['Existing Loans']
        if existing_loans > 1:
            rejection_factors.append(f"Multiple existing loans ({existing_loans})")
    
    # Job stability check
    if 'Job Stability(years)' in application_data:
        job_stability = application_data['Job Stability(years)']
        if job_stability < 2:
            rejection_factors.append(f"Low job stability ({job_stability} years)")
    
    # Past defaults check
    if 'Defaults' in application_data:
        defaults = application_data['Defaults']
        if defaults > 0:
            rejection_factors.append(f"Previous defaults in credit history ({defaults})")
    
    return rejection_factors

def get_loan_type_input():
    """
    Get loan type input from user
    """
    print("\nSelect Loan Type:")
    print("1. Gold Loan")
    print("2. Car Loan")
    print("3. Education Loan")
    
    while True:
        choice = input("Enter your choice (1-3): ")
        if choice in ['1', '2', '3']:
            return int(choice)
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def get_user_input(loan_type):
    """
    Get user input for loan application based on loan type
    """
    application = {}
    
    # Common fields for all loan types
    print("\n=== Basic Information ===")
    application['Full_Name'] = input("Full Name: ")
    
    # Get age with validation
    while True:
        try:
            application['Age'] = int(input("Age: "))
            if 18 <= application['Age'] <= 80:
                break
            else:
                print("Age should be between 18 and 80.")
        except ValueError:
            print("Please enter a valid number for age.")
    
    print("\n=== Financial Information ===")
    
    # Get credit score with validation
    while True:
        try:
            application['Credit Score'] = int(input("Credit Score (300-900): "))
            if 300 <= application['Credit Score'] <= 900:
                break
            else:
                print("Credit score should be between 300 and 900.")
        except ValueError:
            print("Please enter a valid number for credit score.")
    
    # Get annual income with validation
    while True:
        try:
            application['Annual Income'] = float(input("Annual Income (₹): "))
            if application['Annual Income'] > 0:
                break
            else:
                print("Annual income should be greater than 0.")
        except ValueError:
            print("Please enter a valid number for annual income.")
    
    # Get employment type
    print("\nEmployment Type:")
    print("1. Salaried")
    print("2. Self-Employed")
    print("3. Business Owner")
    print("4. Other")
    
    emp_choice = input("Select employment type (1-4): ")
    emp_types = {
        '1': 'Salaried',
        '2': 'Self-Employed',
        '3': 'Business Owner',
        '4': 'Other'
    }
    application['Employment Type'] = emp_types.get(emp_choice, 'Other')
    
    # Get job stability
    while True:
        try:
            application['Job Stability(years)'] = float(input("Job Stability (years): "))
            if application['Job Stability(years)'] >= 0:
                break
            else:
                print("Job stability should be a non-negative number.")
        except ValueError:
            print("Please enter a valid number for job stability.")
    
    # Get existing loans
    while True:
        try:
            application['Existing Loans'] = int(input("Number of Existing Loans: "))
            if application['Existing Loans'] >= 0:
                break
            else:
                print("Number of existing loans should be a non-negative number.")
        except ValueError:
            print("Please enter a valid number for existing loans.")
    
    # Get defaults
    while True:
        try:
            application['Defaults'] = int(input("Number of Past Defaults: "))
            if application['Defaults'] >= 0:
                break
            else:
                print("Number of defaults should be a non-negative number.")
        except ValueError:
            print("Please enter a valid number for defaults.")
    
    # Get loan amount
    while True:
        try:
            application['Loan Amount Requested'] = float(input("Loan Amount Requested (₹): "))
            if application['Loan Amount Requested'] > 0:
                break
            else:
                print("Loan amount should be greater than 0.")
        except ValueError:
            print("Please enter a valid number for loan amount.")
    
    # Set LoanAmount to same value for compatibility with both column names
    application['LoanAmount'] = application['Loan Amount Requested']
    
    # Get repayment duration
    while True:
        try:
            application['RepaymentDuration'] = int(input("Repayment Duration (years): "))
            if application['RepaymentDuration'] > 0:
                break
            else:
                print("Repayment duration should be greater than 0.")
        except ValueError:
            print("Please enter a valid number for repayment duration.")
    
    # Loan type specific fields
    if loan_type == 1:  # Gold Loan
        print("\n=== Gold Loan Specific Information ===")
        
        while True:
            try:
                application['Gold Purity (karats)'] = float(input("Gold Purity (karats, 18-24): "))
                if 18 <= application['Gold Purity (karats)'] <= 24:
                    break
                else:
                    print("Gold purity should be between 18 and 24 karats.")
            except ValueError:
                print("Please enter a valid number for gold purity.")
        
        while True:
            try:
                application['Min Gold Weight (g)'] = float(input("Gold Weight (grams): "))
                if application['Min Gold Weight (g)'] > 0:
                    break
                else:
                    print("Gold weight should be greater than 0.")
            except ValueError:
                print("Please enter a valid number for gold weight.")
        
        # Yes/No questions
        application['Gold Valuation Certificate'] = 1 if input("Do you have a Gold Valuation Certificate? (Yes/No): ").lower() in ['yes', 'y'] else 0
        application['Gold Insurance'] = 1 if input("Do you have Gold Insurance? (Yes/No): ").lower() in ['yes', 'y'] else 0
        
        print("\nGold Storage:")
        print("1. Bank Vault")
        print("2. Personal Locker")
        print("3. Other")
        storage_choice = input("Select gold storage option (1-3): ")
        storage_options = {
            '1': 'Bank Vault',
            '2': 'Personal Locker',
            '3': 'Other'
        }
        application['Gold Storage'] = storage_options.get(storage_choice, 'Other')
        
        application['Purpose of Loan'] = input("Purpose of Loan: ")
        
    elif loan_type == 2:  # Car Loan
        print("\n=== Car Loan Specific Information ===")
        
        application['Car brand'] = input("Car Brand: ")
        
        while True:
            try:
                application['Car Value'] = float(input("Car Value (₹): "))
                if application['Car Value'] > 0:
                    break
                else:
                    print("Car value should be greater than 0.")
            except ValueError:
                print("Please enter a valid number for car value.")
        
        while True:
            try:
                application['Down Payment Made'] = float(input("Down Payment Amount (₹): "))
                if 0 <= application['Down Payment Made'] <= application['Car Value']:
                    break
                else:
                    print("Down payment should be between 0 and car value.")
            except ValueError:
                print("Please enter a valid number for down payment.")
        
        application['Car insurance available'] = 1 if input("Do you have Car Insurance? (Yes/No): ").lower() in ['yes', 'y'] else 0
        
        while True:
            try:
                application['Preferred EMI Amount'] = float(input("Preferred EMI Amount (₹): "))
                if application['Preferred EMI Amount'] > 0:
                    break
                else:
                    print("EMI amount should be greater than 0.")
            except ValueError:
                print("Please enter a valid number for EMI amount.")
        
        while True:
            try:
                application['Existing EMIs'] = float(input("Existing EMI Payments (₹/month): "))
                if application['Existing EMIs'] >= 0:
                    break
                else:
                    print("Existing EMIs should be a non-negative number.")
            except ValueError:
                print("Please enter a valid number for existing EMIs.")
        
        application['Marital Status'] = input("Marital Status (Single/Married/Other): ")
        
        while True:
            try:
                application['Number of Dependents'] = int(input("Number of Dependents: "))
                if application['Number of Dependents'] >= 0:
                    break
                else:
                    print("Number of dependents should be a non-negative number.")
            except ValueError:
                print("Please enter a valid number for dependents.")
        
    elif loan_type == 3:  # Education Loan
        print("\n=== Education Loan Specific Information ===")
        
        application['Institution'] = input("Institution Name: ")
        application['Degree'] = input("Degree (e.g., B.Tech, MBA): ")
        application['CourseName'] = input("Course Name (e.g., Computer Science): ")
        
        print("\nCourse Type:")
        print("1. Undergraduate")
        print("2. Postgraduate")
        print("3. Diploma")
        print("4. Certificate")
        course_choice = input("Select course type (1-4): ")
        course_types = {
            '1': 'Undergraduate',
            '2': 'Postgraduate',
            '3': 'Diploma',
            '4': 'Certificate'
        }
        application['CourseType'] = course_types.get(course_choice, 'Other')
        
        while True:
            try:
                application['CourseDuration'] = float(input("Course Duration (years): "))
                if application['CourseDuration'] > 0:
                    break
                else:
                    print("Course duration should be greater than 0.")
            except ValueError:
                print("Please enter a valid number for course duration.")
        
        while True:
            try:
                application['TotalCost'] = float(input("Total Course Cost (₹): "))
                if application['TotalCost'] > 0:
                    break
                else:
                    print("Total cost should be greater than 0.")
            except ValueError:
                print("Please enter a valid number for total cost.")
        
        while True:
            try:
                application['TuitionFees'] = float(input("Tuition Fees (₹): "))
                if application['TuitionFees'] > 0:
                    break
                else:
                    print("Tuition fees should be greater than 0.")
            except ValueError:
                print("Please enter a valid number for tuition fees.")
        
        while True:
            try:
                application['Marks12'] = float(input("12th Standard Marks (%): "))
                if 0 <= application['Marks12'] <= 100:
                    break
                else:
                    print("Marks should be between 0 and 100.")
            except ValueError:
                print("Please enter a valid number for marks.")
        
        print("\nAdmission Status:")
        print("1. Confirmed")
        print("2. Provisional")
        print("3. Applied")
        admission_choice = input("Select admission status (1-3): ")
        admission_status = {
            '1': 'Confirmed',
            '2': 'Provisional',
            '3': 'Applied'
        }
        application['AdmissionStatus'] = admission_status.get(admission_choice, 'Applied')
        
        while True:
            try:
                application['FamilyIncome'] = float(input("Family Annual Income (₹): "))
                if application['FamilyIncome'] >= 0:
                    break
                else:
                    print("Family income should be a non-negative number.")
            except ValueError:
                print("Please enter a valid number for family income.")
        
        application['ITRAvailable'] = 1 if input("Is Income Tax Return (ITR) Available? (Yes/No): ").lower() in ['yes', 'y'] else 0
        
        application['Gender'] = input("Gender (Male/Female/Other): ")
    
    return application

def main():
    """
    Main function to run the loan default prediction system with user input
    """
    print("\n===== Loan Default Risk Prediction System =====\n")
    
    # Check if model exists
    if not os.path.exists('loan_default_model.pkl'):
        print("Model not found. Please run model.py first to train the model.")
        return
    
    print("This system will predict the default risk for a loan application.")
    print("Please provide the information as requested.\n")
    
    while True:
        # Get loan type
        loan_type = get_loan_type_input()
        
        # Get user input for the loan application
        loan_name = {1: "Gold Loan", 2: "Car Loan", 3: "Education Loan"}
        print(f"\nCollecting information for {loan_name[loan_type]} application...")
        
        application = get_user_input(loan_type)
        
        # Make prediction
        print("\nProcessing your application...")
        prediction = predict_default_risk(application)
        
        # Print key application details
        print("\n--- Loan Application Details ---")
        print(f"Applicant: {application['Full_Name']}")
        print(f"Age: {application['Age']}")
        print(f"Credit Score: {application['Credit Score']}")
        print(f"Loan Type: {loan_name[loan_type]}")
        print(f"Loan Amount: ₹{application['Loan Amount Requested']:,.2f}")
        print(f"Annual Income: ₹{application['Annual Income']:,.2f}")
        print(f"Existing Loans: {application['Existing Loans']}")
        
        # Print prediction results
        print("\nPrediction Results:")
        print(f"Approval Prediction: {prediction['approval_prediction']}")
        print(f"Approval Probability: {prediction['approval_probability']:.2%}")
        print(f"Risk Category: {prediction['risk_category']}")
        
        # If high risk or declined, analyze rejection factors
        if prediction['risk_category'] in ['High Risk', 'Very High Risk'] or prediction['approval_prediction'] == 'Declined':
            rejection_factors = analyze_rejection_factors(application)
            if rejection_factors:
                print("\nPotential rejection factors:")
                for factor in rejection_factors:
                    print(f"  - {factor}")
        
        # Ask if user wants to continue
        choice = input("\nDo you want to evaluate another loan application? (Yes/No): ")
        if choice.lower() not in ['yes', 'y']:
            break
    
    print("\nThank you for using the Loan Default Risk Prediction System.")

if __name__ == "__main__":
    main()