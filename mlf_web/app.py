from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
import uuid
from model import predict_loan_approval, train_model

app = Flask(__name__, static_folder='loan_mlf')
app.secret_key = 'your-secret-key-change-this-in-production'  # Change this to a secure secret key
CORS(app)

# Load ML model and feature info
MODEL_PATH = 'loan_default_model.pkl'
FEATURE_INFO_PATH = 'feature_info.pkl'

def load_model():
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_INFO_PATH):
            print("Model and feature info loaded successfully")
            return True
        else:
            print("Model files not found, training new model...")
            train_model('Loan_Dataset.csv')
            return True
    except Exception as e:
        print(f"Error loading/training model: {e}")
        return False

# Initialize model
model_loaded = load_model()

# Define the path for the JSON file storage
APPLICATIONS_FILE = 'applications.json'

# Initialize a counter for auto-incrementing IDs
application_id_counter = 0

# Function to load applications from the JSON file
def load_applications():
    global application_id_counter
    if os.path.exists(APPLICATIONS_FILE):
        with open(APPLICATIONS_FILE, 'r') as f:
            try:
                applications_list = json.load(f)
                # Convert submitted_at strings back to datetime objects if needed for sorting/filtering, 
                # but keeping as string is fine for storage and display
                
                # Ensure applications have integer IDs and update counter
                max_id = 0
                for app in applications_list:
                    # If application has a UUID, keep it for now but assign a new int ID if needed
                    # For simplicity now, let's just ensure all have an 'id' field and find max int ID
                    if 'id' not in app or isinstance(app['id'], str):
                         # Assign a temporary negative ID or handle as needed; we'll prioritize finding max int ID
                         pass # We'll assign new integer IDs later if needed
                    elif isinstance(app['id'], int):
                        max_id = max(max_id, app['id'])
                application_id_counter = max_id

                return applications_list
            except json.JSONDecodeError:
                # Return an empty list if the file is empty or corrupted
                return []
    else:
        return []

# Function to save applications to the JSON file
def save_applications(applications_list):
    with open(APPLICATIONS_FILE, 'w') as f:
        # Convert datetime objects to strings before saving if necessary (not needed here as we store as string)
        json.dump(applications_list, f, indent=4)

# Load applications when the app starts
applications = load_applications()

# Mock user database (replace with a real database in production)
USERS = {
    'loan_user': {'password': 'officer123', 'role': 'officer'},
    'branch_mgr': {'password': 'manager456', 'role': 'manager'},
    'admin': {'password': 'admin123', 'role': 'admin'}
}

# Mock ML prediction function when models are not available
def mock_prediction(loan_type, data):
    """Mock prediction for testing when ML models are not available"""
    
    # Simple rule-based mock predictions for demonstration
    credit_score = float(data.get('credit_score', 300))
    income = float(data.get('income', 0))
    loan_amount = float(data.get('loan_amount', 0))
    
    # Calculate basic risk score
    income_to_loan_ratio = income / loan_amount if loan_amount > 0 else 0
    
    if credit_score >= 700 and income_to_loan_ratio >= 3:
        prediction = "Approved"
        probability = 0.85
        risk_level = "Low Risk"
        reason = "Good credit score and adequate income"
    elif credit_score >= 600 and income_to_loan_ratio >= 2:
        prediction = "Approved"
        probability = 0.65
        risk_level = "Moderate Risk"
        reason = "Acceptable credit score and income"
    else:
        prediction = "Declined"
        probability = 0.35
        risk_level = "High Risk"
        reason = "Credit score or income below requirements"
    
    return {
        "prediction": prediction,
        "probability": probability,
        "risk_level": risk_level,
        "reason": reason
    }

def predict_loan_default(loan_type, data):
    """Make predictions using our ML model"""
    try:
        if not model_loaded:
            print("Model not loaded, using mock prediction")
            # Use the provided data for mock prediction
            return mock_prediction(loan_type, data)

        # Use the provided data directly for prediction
        ml_data = data # Use the data dictionary passed to the function

        # Make prediction using our model
        # The predict_loan_approval function in model.py expects a DataFrame, so convert data
        applicant_df = pd.DataFrame([ml_data])

        prediction_result = predict_loan_approval(MODEL_PATH, FEATURE_INFO_PATH, applicant_df)
        
        print("Raw prediction result from model.py:", prediction_result)
        
        if 'error' in prediction_result:
            print(f"Error in prediction: {prediction_result['error']}")
            return mock_prediction(loan_type, data)

        # Format prediction result
        return {
            "prediction": "Approved" if prediction_result['approved'] else "Declined",
            "risk_level": "Low Risk" if prediction_result['approved'] else "High Risk",
            "approval_probability": prediction_result['approval_probability'],
            "decline_probability": prediction_result['decline_probability'],
            # Use the reason from the prediction_result if available, otherwise use a default
            "prediction_reason": prediction_result.get('reason', f"Prediction based on application data. Status: {'Approved' if prediction_result['approved'] else 'Declined'}")
        }

    except Exception as e:
        print(f"Error making prediction: {e}")
        import traceback
        traceback.print_exc()
        # Return a dictionary indicating an error, possibly with a fallback reason
        return {
            'prediction': 'Error',
            'risk_level': 'Unknown',
            'approval_probability': 0.0,
            'decline_probability': 0.0,
            'prediction_reason': f"Error during prediction: {str(e)}. Could not get a specific reason."
        }

@app.route('/')
def index():
    return send_from_directory('loan_mlf', 'login.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if username in USERS and USERS[username]['password'] == password:
        session['user'] = username
        session['role'] = USERS[username]['role']
        return jsonify({'success': True, 'role': USERS[username]['role']})
    return jsonify({'success': False, 'message': 'Invalid credentials'})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# Route to handle Car Loan form submission
@app.route('/submit_loan/car', methods=['POST'])
def submit_car_loan():
    global applications, application_id_counter
    try:
        data = request.json if request.is_json else request.form.to_dict()
        
        print(f"[DEBUG] Submitting car loan. User in session: {session.get('user')}")
        
        # Construct ml_data dictionary with all expected features for the combined model
        # Based on features defined in model.py -> select_features and clean_data
        ml_data = {
            # --- Features from select_features in model.py --- #
            # Important Features:
            'Credit Score': int(data.get('creditScore', data.get('Family_Credit_Score', 300))), # Map 'creditScore', also check 'Family_Credit_Score'
            'Annual Income': int(data.get('annualIncome', data.get('income', 0))), # Map 'annualIncome', also check 'income'
            'Income': int(data.get('annualIncome', data.get('income', 0))), # Explicitly adding 'Income' as it appeared in the error
            'Age': int(data.get('age', 25)), # Map 'age'
            'Job Stability(years)': int(data.get('jobStability', 0)), # Map 'jobStability'
            'Existing Loans': int(data.get('existingLoans', 0)), # Default to 0 for car loan
            'Existing EMIs': int(data.get('existingEMIs', data.get('Existing_Family_Loans_EMIs', 0))), # Map 'existingEMIs', also check 'Existing_Family_Loans_EMIs'
            'LoanAmount': int(data.get('loanAmount', 0)), # Map 'loanAmount', duplicate for robustness
            'Loan Amount Requested': int(data.get('loanAmount', 0)), # Map 'loanAmount'
            'Employment Type': data.get('employmentType', data.get('Parent_Guardian_Occupation', 'Salaried')), # Map 'employmentType', also check 'Parent_Guardian_Occupation'
            'Defaults': data.get('loanDefaults', data.get('Previous_Loan_Defaults_in_Family', 'No')), # Map 'loanDefaults', also check 'Previous_Loan_Defaults_in_Family'
            'RepaymentDuration': int(data.get('loanTenure', 0)), # Map 'loanTenure'
            'Gender': data.get('gender', 'Unknown'), # Map 'gender', default
            'Marital Status': data.get('maritalStatus', 'Single'), # Map 'maritalStatus'
            'Number of Dependents': int(data.get('dependents', data.get('Family_Dependents', 0))), # Map 'dependents', also check 'Family_Dependents'
            'ITRAvailable': data.get('ITRAvailable', 'No'), # Default
            'Purpose of Loan': data.get('loanPurpose', 'Other'), # Map 'loanPurpose'

            # Gold loan specific features (provide defaults):
            'Gold Purity (karats)': 0.0, 
            'Min Gold Weight (g)': 0.0, 
            'Gold Valuation Certificate': 'No', 
            'Gold Insurance': 'No', 
            'Gold Storage': 'Bank',

            # Car loan specific features:
            'Car brand': data.get('carBrand', 'Unknown'), # Map 'carBrand'
            'Car insurance available': data.get('carInsurance', 'No'), # Map 'carInsurance'
            'Down Payment Made': int(data.get('downPayment', 0)), # Map 'downPayment'

            # Educational loan specific features (provide defaults):
            'CourseType': 'N/A',
            'Institution': 'Unknown', 
            'Degree': 'Unknown', 
            'Marks12': 0.0, 
            'ExamScore': 0, 
            'AdmissionStatus': 'Unknown', 
            'TotalCost': 0, 
            'TuitionFees': 0, 
            'FamilyContribution': 0,

            # --- Engineered Features from clean_data in model.py --- #
            # These will be recalculated in clean_data, but including here with placeholders for completeness if needed before clean_data
            'Loan_to_Income_Ratio': 0.0, 
            'Credit_Score_to_Loans': 0.0,

            # Additional fields that might be expected based on other forms, provide defaults:
            'Bank_Account_last_4_digits': data.get('bankAccount', ''), # Map 'bankAccount'
            'Category': 'Other',
            'Course_Level': 'N/A',
            'Institution_Type': 'N/A',
            'Institution_Ranking': int(data.get('Institution_Ranking', 0)),
            'Previous_Academic_Grade': float(data.get('Previous_Academic_Grade', 0.0)),
            'Course_Duration_Years': int(data.get('Course_Duration_Years', 0)),
            'Study_Location': 'N/A',
            'Parent_Guardian_Annual_Income': int(data.get('Parent_Guardian_Annual_Income', 0)),
            'Parent_Guardian_Occupation': data.get('Parent_Guardian_Occupation', 'N/A'),
            'Family_Credit_Score': int(data.get('Family_Credit_Score', 0)),
            'Existing_Family_Loans_EMIs': int(data.get('Existing_Family_Loans_EMIs', 0)),
            'Property_Ownership': data.get('Property_Ownership', 'No'),
            'Family_Dependents': int(data.get('Family_Dependents', 0)),
            'Self_Family_Contribution': int(data.get('Self_Family_Contribution', 0)),
            'Scholarship_Amount': int(data.get('Scholarship_Amount', 0)),
            'Moratorium_Period': data.get('Moratorium_Period', 'None'),
            'Preferred_Repayment_Tenure_Years': int(data.get('Preferred_Repayment_Tenure_Years', 0)),
            'Collateral_Available': data.get('Collateral_Available', 'No'),
            'Previous_Loan_Defaults_in_Family': data.get('Previous_Loan_Defaults_in_Family', 'No'),
            'Work_Experience_Years': int(data.get('Work_Experience_Years', 0)),
            'Future_Career_Plans': data.get('future_career_plan', 'Unknown'), # Ensure this field is mapped
            # Note: 'Student_Full_Name' is not used in prediction, so not included here
            # Note: 'Loan_Type' is handled by the backend route '/submit_loan/education'
        }
        
        # Get ML prediction
        prediction_result = predict_loan_default('car', ml_data)
        
        # Generate auto-incrementing ID
        application_id_counter += 1
        new_application_id = application_id_counter

        # Create new application entry
        new_application = {
            'id': new_application_id,
            'applicant_name': session.get('user', 'N/A'), # Use logged-in username as applicant name
            'loan_type': 'car',
            'loan_amount': data.get('loanAmount', 'N/A'),
            'submitted_at': datetime.now().isoformat(),
            'ml_prediction': prediction_result.get('prediction', 'N/A'),
            'prediction_probability': prediction_result.get('approval_probability', 0.0) * 100, # Convert to percentage
            'risk_level': prediction_result.get('risk_level', 'Unknown'),
            'status': 'Pending', # Default status
            'action': 'Pending Review', # Default action
            'data': data # Store original form data
        }

        applications.append(new_application)
        save_applications(applications)

        return jsonify({'success': True, 'message': 'Application submitted successfully!', 'application_id': new_application_id, 'prediction_result': prediction_result})
        
    except Exception as e:
        print(f"Error submitting car loan application: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# Route to handle Education Loan form submission
@app.route('/submit_loan/education', methods=['POST'])
def submit_education_loan():
    global applications, application_id_counter
    try:
        data = request.json if request.is_json else request.form.to_dict()
        
        print(f"[DEBUG] Submitting education loan. User in session: {session.get('user')}")
        
        # Construct ml_data dictionary with all expected features for the combined model
        # Based on features defined in model.py -> select_features and clean_data
        ml_data = {
            # --- Features from select_features in model.py --- #
            # Important Features:
            'Credit Score': int(data.get('Family_Credit_Score', data.get('creditScore', 300))), # Map 'Family_Credit_Score', also check 'creditScore'
            'Annual Income': int(data.get('Parent_Guardian_Annual_Income', data.get('annualIncome', data.get('income', 0)))), # Map 'Parent_Guardian_Annual_Income', also check others
            'Income': int(data.get('Parent_Guardian_Annual_Income', data.get('annualIncome', data.get('income', 0)))), # Explicitly adding 'Income'
            'Age': int(data.get('Student_Age', data.get('age', 25))), # Map 'Student_Age', also check 'age'
            'Job Stability(years)': int(data.get('Work_Experience_Years', data.get('jobStability', 0))), # Map 'Work_Experience_Years', also check 'jobStability'
            'Existing Loans': int(data.get('Existing_Family_Loans_EMIs', data.get('existingLoans', 0))), # Map 'Existing_Family_Loans_EMIs', also check 'existingLoans'
            'Existing EMIs': int(data.get('Existing_Family_Loans_EMIs', data.get('existingEMIs', 0))), # Map 'Existing_Family_Loans_EMIs', also check 'existingEMIs'
            'LoanAmount': int(data.get('Loan_Amount_Requested', data.get('loanAmount', 0))), # Map 'Loan_Amount_Requested', also check 'loanAmount'
            'Loan Amount Requested': int(data.get('Loan_Amount_Requested', data.get('loanAmount', 0))), # Map 'Loan_Amount_Requested', also check 'loanAmount'
            'Employment Type': data.get('Parent_Guardian_Occupation', data.get('employmentType', 'Salaried')), # Map 'Parent_Guardian_Occupation', also check 'employmentType'
            'Defaults': data.get('Previous_Loan_Defaults_in_Family', data.get('loanDefaults', 'No')), # Map 'Previous_Loan_Defaults_in_Family', also check 'loanDefaults'
            'RepaymentDuration': int(data.get('Preferred_Repayment_Tenure_Years', data.get('loanTenure', 0))), # Map 'Preferred_Repayment_Tenure_Years', also check 'loanTenure'
            'Gender': data.get('Gender', 'Unknown'), # Map 'Gender'
            'Marital Status': data.get('Marital Status', 'Single'), # Map 'Marital Status', default
            'Number of Dependents': int(data.get('Family_Dependents', data.get('dependents', 0))), # Map 'Family_Dependents', also check 'dependents'
            'ITRAvailable': data.get('ITRAvailable', 'No'), # Default
            'Purpose of Loan': data.get('Future_Career_Plans', data.get('loanPurpose', 'Other')), # Map 'Future_Career_Plans', also check 'loanPurpose'

            # Gold loan specific features (provide defaults):
            'Gold Purity (karats)': 0.0, 
            'Min Gold Weight (g)': 0.0, 
            'Gold Valuation Certificate': 'No', 
            'Gold Insurance': 'No', 
            'Gold Storage': 'Bank',

            # Car loan specific features (provide defaults):
            'Car brand': 'N/A', 
            'Car insurance available': 'No', 
            'Down Payment Made': 0,

            # Educational loan specific features:
            'CourseType': data.get('Course_Type', 'N/A'), # Map 'Course_Type'
            'Institution': data.get('Institution', 'Unknown'), # Map 'Institution'
            'Degree': data.get('Course_Level', 'Unknown'), # Map 'Course_Level' to 'Degree'
            'Marks12': float(data.get('Previous_Academic_Grade', 0.0)), # Map 'Previous_Academic_Grade'
            'ExamScore': int(data.get('Entrance_Exam_Score', 0)), # Map 'Entrance_Exam_Score'
            'AdmissionStatus': data.get('AdmissionStatus', 'Unknown'), # Map 'AdmissionStatus', default
            'TotalCost': int(data.get('Total_Course_Fee', 0)), # Map 'Total_Course_Fee'
            'TuitionFees': int(data.get('TuitionFees', 0)), # Map 'TuitionFees', default
            'FamilyContribution': int(data.get('Self_Family_Contribution', 0)), # Map 'Self_Family_Contribution'

            # --- Engineered Features from clean_data in model.py --- #
            # These will be recalculated in clean_data, but including here with placeholders for completeness if needed before clean_data
            'Loan_to_Income_Ratio': 0.0, 
            'Credit_Score_to_Loans': 0.0,

            # Additional fields that might be expected based on other forms, provide defaults:
            'Nationality': data.get('Nationality', ''), # Map 'Nationality'
            'Bank_Account_last_4_digits': data.get('Bank_Account_last_4_digits', ''), # Map 'Bank_Account_last_4_digits'
            'Category': data.get('Category', 'Other'), # Map 'Category'
            'Institution_Type': data.get('Institution_Type', 'N/A'), # Map 'Institution_Type'
            'Institution_Ranking': int(data.get('Institution_Ranking', 0)), # Map 'Institution_Ranking'
            'Study_Location': data.get('Study_Location', 'N/A'), # Map 'Study_Location'
            'Property_Ownership': data.get('Property_Ownership', 'No'), # Map 'Property_Ownership'
            'Self_Family_Contribution': int(data.get('Self_Family_Contribution', 0)), # Map 'Self_Family_Contribution'
            'Scholarship_Amount': int(data.get('Scholarship_Amount', 0)), # Map 'Scholarship_Amount'
            'Moratorium_Period': data.get('Moratorium_Period', 'None'), # Map 'Moratorium_Period'
            'Co_applicant_Required': data.get('Co_applicant_Required', 'No'), # Map 'Co_applicant_Required'
            'Collateral_Available': data.get('Collateral_Available', 'No'), # Map 'Collateral_Available'
            'Work_Experience_Years': int(data.get('Work_Experience_Years', 0)), # Map 'Work_Experience_Years'
            'Future_Career_Plans': data.get('future_career_plan', 'Unknown'), # Ensure this field is mapped
            # Note: 'Student_Full_Name' is not used in prediction, so not included here
            # Note: 'Loan_Type' is handled by the backend route '/submit_loan/education'
        }
        
        # Get ML prediction
        prediction_result = predict_loan_default('education', ml_data)
        
        # Generate auto-incrementing ID
        application_id_counter += 1
        new_application_id = application_id_counter

        # Create new application entry
        new_application = {
            'id': new_application_id,
            'applicant_name': session.get('user', 'N/A'), # Use logged-in username as applicant name
            'loan_type': 'education',
            'loan_amount': data.get('loanAmount', 'N/A'),
            'submitted_at': datetime.now().isoformat(),
            'ml_prediction': prediction_result.get('prediction', 'N/A'),
            'prediction_probability': prediction_result.get('approval_probability', 0.0) * 100, # Convert to percentage
            'risk_level': prediction_result.get('risk_level', 'Unknown'),
            'status': 'Pending', # Default status
            'action': 'Pending Review', # Default action
            'data': data # Store original form data
        }

        applications.append(new_application)
        save_applications(applications)

        return jsonify({'success': True, 'message': 'Application submitted successfully!', 'application_id': new_application_id, 'prediction_result': prediction_result})
        
    except Exception as e:
        print(f"Error submitting educational loan application: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# Route to handle Gold Loan form submission
@app.route('/submit_loan/gold', methods=['POST'])
def submit_gold_loan():
    global applications, application_id_counter
    try:
        data = request.json if request.is_json else request.form.to_dict()
        
        print(f"[DEBUG] Submitting gold loan. User in session: {session.get('user')}")
        
        # Construct ml_data dictionary with all expected features for the combined model
        # Based on features defined in model.py -> select_features and clean_data
        ml_data = {
            # --- Features from select_features in model.py --- #
            # Important Features:
            'Credit Score': int(data.get('creditScore', data.get('Family_Credit_Score', 300))), # Map 'creditScore', also check 'Family_Credit_Score'
            'Annual Income': int(data.get('annualIncome', data.get('income', 0))), # Map 'annualIncome', also check 'income'
            'Income': int(data.get('annualIncome', data.get('income', 0))), # Explicitly adding 'Income' as it appeared in the error
            'Age': int(data.get('age', 25)), # Map 'age'
            'Job Stability(years)': int(data.get('jobStability', 0)), # Map 'jobStability'
            'Existing Loans': int(data.get('existingLoans', 0)), # Explicitly adding 'Existing Loans' as it appeared in the error, default to 0
            'Existing EMIs': int(data.get('existingEMIs', data.get('Existing_Family_Loans_EMIs', 0))), # Map 'existingEMIs', also check 'Existing_Family_Loans_EMIs'
            'LoanAmount': int(data.get('loanAmount', 0)), # Map 'loanAmount', duplicate for robustness
            'Loan Amount Requested': int(data.get('loanAmount', 0)), # Map 'loanAmount'
            'Employment Type': data.get('employmentType', data.get('Parent_Guardian_Occupation', 'Salaried')), # Map 'employmentType', also check 'Parent_Guardian_Occupation'
            'Defaults': data.get('loanDefaults', data.get('Previous_Loan_Defaults_in_Family', 'No')), # Map 'loanDefaults', also check 'Previous_Loan_Defaults_in_Family'
            'RepaymentDuration': int(data.get('loanTenure', 0)), # Map 'loanTenure'
            'Gender': data.get('Gender', 'Unknown'), # Default for gold loan
            'Marital Status': data.get('maritalStatus', 'Single'), # Map 'maritalStatus'
            'Number of Dependents': int(data.get('dependents', data.get('Family_Dependents', 0))), # Map 'dependents', also check 'Family_Dependents'
            'ITRAvailable': data.get('ITRAvailable', 'No'), # Default
            'Purpose of Loan': data.get('loanPurpose', 'Other'), # Map 'loanPurpose'

            # Gold loan specific features:
            'Gold Purity (karats)': float(data.get('goldPurity', 0)), # Map 'goldPurity'
            'Min Gold Weight (g)': float(data.get('goldWeight', 0)), # Map 'goldWeight'
            'Gold Valuation Certificate': data.get('valuationCertificate', 'No'), # Map 'valuationCertificate'
            'Gold Insurance': data.get('goldInsurance', 'No'), # Map 'goldInsurance'
            'Gold Storage': data.get('goldStorage', 'Bank'), # Map 'goldStorage'

            # Car loan specific features (provide defaults):
            'Car brand': 'N/A', 
            'Car insurance available': data.get('carInsurance', 'No'), # Default
            'Down Payment Made': int(data.get('downPayment', 0)), # Default

            # Educational loan specific features (provide defaults):
            'CourseType': data.get('Course_Type', 'N/A'), # Explicitly adding missing column from error
            'Institution': data.get('Institution', 'Unknown'), # Explicitly adding missing column from error
            'Degree': data.get('Degree', 'Unknown'), # Explicitly adding missing column from error
            'Marks12': float(data.get('Marks12', 0.0)), # Explicitly adding missing column from error
            'ExamScore': int(data.get('Entrance_Exam_Score', data.get('ExamScore', 0))), # Explicitly adding missing column from error, check 'Entrance_Exam_Score'
            'AdmissionStatus': data.get('AdmissionStatus', 'Unknown'), # Explicitly adding missing column from error
            'TotalCost': int(data.get('Total_Course_Fee', data.get('TotalCost', 0))), # Explicitly adding missing column from error, check 'Total_Course_Fee'
            'TuitionFees': int(data.get('TuitionFees', 0)), # Explicitly adding missing column from error
            'FamilyContribution': int(data.get('Self_Family_Contribution', data.get('FamilyContribution', 0))), # Explicitly adding missing column from error, check 'Self_Family_Contribution'

            # --- Engineered Features from clean_data in model.py --- #
            # These will be recalculated in clean_data, but including here with placeholders for completeness if needed before clean_data
            'Loan_to_Income_Ratio': 0.0, 
            'Credit_Score_to_Loans': 0.0,

            # Additional fields that might be expected based on previous data exploration or other forms, provide defaults:
            'Bank_Account_last_4_digits': data.get('bankAccount', ''), # Map 'bankAccount'
            'Category': data.get('Category', 'Other'),
            'Course_Level': data.get('Course_Level', 'N/A'),
            'Institution_Type': data.get('Institution_Type', 'N/A'),
            'Institution_Ranking': int(data.get('Institution_Ranking', 0)),
            'Previous_Academic_Grade': float(data.get('Previous_Academic_Grade', 0.0)),
            'Study_Location': data.get('Study_Location', 'N/A'),
            'Parent_Guardian_Annual_Income': int(data.get('Parent_Guardian_Annual_Income', 0)),
            'Parent_Guardian_Occupation': data.get('Parent_Guardian_Occupation', 'N/A'),
            'Family_Credit_Score': int(data.get('Family_Credit_Score', 0)),
            'Existing_Family_Loans_EMIs': int(data.get('Existing_Family_Loans_EMIs', 0)),
            'Property_Ownership': data.get('Property_Ownership', 'No'),
            'Family_Dependents': int(data.get('Family_Dependents', 0)),
            'Self_Family_Contribution': int(data.get('Self_Family_Contribution', 0)),
            'Scholarship_Amount': int(data.get('Scholarship_Amount', 0)),
            'Moratorium_Period': data.get('Moratorium_Period', 'None'),
            'Preferred_Repayment_Tenure_Years': int(data.get('Preferred_Repayment_Tenure_Years', 0)),
            'Collateral_Available': data.get('Collateral_Available', 'No'),
            'Previous_Loan_Defaults_in_Family': data.get('Previous_Loan_Defaults_in_Family', 'No'),
            'Work_Experience_Years': int(data.get('Work_Experience_Years', 0)),
            'Future_Career_Plans': data.get('future_career_plan', 'Unknown'), # Ensure this field is mapped
            # Note: 'Student_Full_Name' is not used in prediction, so not included here
            # Note: 'Loan_Type' is handled by the backend route '/submit_loan/education'
        }
        
        # Get ML prediction
        prediction_result = predict_loan_default('gold', ml_data)
        
        # Generate auto-incrementing ID
        application_id_counter += 1
        new_application_id = application_id_counter

        # Create new application entry
        new_application = {
            'id': new_application_id,
            'applicant_name': session.get('user', 'N/A'), # Use logged-in username as applicant name
            'loan_type': 'gold',
            'loan_amount': data.get('loanAmount', 'N/A'),
            'submitted_at': datetime.now().isoformat(),
            'ml_prediction': prediction_result.get('prediction', 'N/A'),
            'prediction_probability': prediction_result.get('approval_probability', 0.0) * 100, # Convert to percentage
            'risk_level': prediction_result.get('risk_level', 'Unknown'),
            'status': 'Pending', # Default status
            'action': 'Pending Review', # Default action
            'data': data # Store original form data
        }

        applications.append(new_application)
        save_applications(applications)

        return jsonify({'success': True, 'message': 'Application submitted successfully!', 'application_id': new_application_id, 'prediction_result': prediction_result})
        
    except Exception as e:
        print(f"Error submitting gold loan application: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# Route to get a specific application by ID
@app.route('/get_application/<int:app_id>')
def get_application(app_id):
    for app in applications:
        if app.get('id') == app_id:
            return jsonify(app)
    return jsonify({'message': 'Application not found'}), 404

# Route to render loan result page - might not be needed if result is shown on form page
@app.route('/loan_result/<int:app_id>')
def loan_result(app_id):
    # Find the application by ID
    application = next((app for app in applications if app.get('id') == app_id), None)
    if application:
        # Render a template or return data for the result page
        # For now, just return the application data as JSON
        return jsonify(application)
    return jsonify({'message': 'Application not found'}), 404

# API route to get a specific application by ID (redundant, use get_application)
@app.route('/api/application/<int:app_id>')
def api_get_application(app_id):
    for app in applications:
        if app.get('id') == app_id:
            return jsonify(app)
    return jsonify({'message': 'Application not found'}), 404

# Route to get all applications
@app.route('/get_applications')
def get_applications():
    # This endpoint is called by the frontend to get all applications
    # Ensure we return the loaded applications list
    updated_applications = load_applications()
    return jsonify(updated_applications)

# Endpoint to update application status (used by total-applications.html)
@app.route('/update_application_status', methods=['POST'])
def update_application_status_json():
    try:
        data = request.json
        app_id_to_update = data.get('application_id')
        new_status = data.get('status')

        if not app_id_to_update or not new_status:
            return jsonify({'success': False, 'message': 'Missing application_id or status'}), 400

        # Find the application by ID (handle both int and string IDs if necessary, but prioritize string IDs)
        # Convert app_id_to_update to string to match frontend
        app_id_str = str(app_id_to_update)
        app_index = -1
        for i, app in enumerate(applications):
            if str(app.get('id')) == app_id_str:
                app_index = i
                break

        if app_index != -1:
            applications[app_index]['status'] = new_status
            # Save the updated applications list to the JSON file
            save_applications(applications)
            return jsonify({'success': True, 'message': 'Application status updated successfully!'})
        else:
            return jsonify({'success': False, 'message': 'Application not found'}), 404

    except Exception as e:
        print(f"Error updating application status: {e}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500

# Route to generate report (placeholder)
@app.route('/generate_report')
def generate_report():
    # This would typically generate a downloadable report (CSV, PDF, etc.)
    return jsonify({'message': 'Report generation not implemented yet'})

# Route to test models (placeholder)
@app.route('/test_models')
def test_models():
    # This could run a script to test model performance
    return jsonify({'message': 'Model testing not implemented yet'})

# Route for serving static files (existing)
@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('loan_mlf', path)

# Error handlers (existing)
@app.errorhandler(404)
def not_found(error):
    return jsonify({'message': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'message': 'Internal server error'}), 500

# Route to list all applications (redundant, use get_applications)
@app.route('/applications', methods=['GET'])
def list_all_applications():
    # This route is similar to /get_applications, let's make it use that logic
    return get_applications()

# Route to update application status by ID (redundant, use update_application_status_json)
@app.route('/update_application_status/<int:app_id>/<string:status>', methods=['POST'])
def update_application_status_by_id(app_id, status):
    # This route can be simplified to use the update_application_status_json logic
    # Convert int app_id to string for consistency with JSON endpoint
    return update_application_status_json(application_id=str(app_id), status=status)

# New route to delete an application by ID
@app.route('/delete_application/<string:app_id>', methods=['DELETE'])
def delete_application(app_id):
    global applications
    try:
        # Find the application by ID (handle both int and string IDs if necessary)
        original_applications_count = len(applications)
        applications = [app for app in applications if str(app.get('id')) != str(app_id)]

        if len(applications) == original_applications_count:
            # No application was removed, meaning ID was not found
            return jsonify({'success': False, 'message': 'Application not found'}), 404

        save_applications(applications) # Save the updated list

        return jsonify({'success': True, 'message': 'Application deleted successfully'})

    except Exception as e:
        print(f"Error deleting application: {e}")
        return jsonify({'success': False, 'message': 'Internal server error'}), 500

if __name__ == '__main__':
    print("="*50)
    print("LOAN APPLICATION SYSTEM")
    print("="*50)
    print(f"Model loaded: {'Using our model' if model_loaded else 'Using mock predictions'}")
    # Load applications again to ensure latest state in startup message
    current_applications = load_applications()
    print(f"Total applications loaded: {len(current_applications)}")
    print("="*50)
    print("Available endpoints:")
    print("- / : Login page")
    print("- /home.html : Main dashboard")
    print("- /car_loan.html : Car loan form")
    print("- /education_loan.html : Education loan form") 
    print("- /gold_loan.html : Gold loan form")
    print("- /loan_result/<id> : Application result page")
    print("- /test_models : Test ML models")
    print("="*50)
    
    app.run(debug=True, port=5000, host='0.0.0.0')