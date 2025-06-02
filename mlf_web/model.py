import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def clean_data(df):
    """
    Clean and preprocess the loan dataset
    """
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Identify target variable (loan status) and convert to binary
    if 'Loan status' in data.columns:
        data['Loan status'] = data['Loan status'].map({'Approved': 1, 'Declined': 0})
        # Ensure no NaN values in target variable
        if data['Loan status'].isna().any():
            print(f"Warning: Found {data['Loan status'].isna().sum()} missing values in target variable.")
            # Fill missing values with most frequent value
            most_frequent = data['Loan status'].mode()[0]
            data['Loan status'] = data['Loan status'].fillna(most_frequent)
            print(f"Filled missing target values with most frequent value: {most_frequent}")
    
    # Convert string percentages to float
    for col in data.columns:
        if data[col].dtype == object:
            # Try to convert percentage strings to float
            try:
                # Check if column contains percentage values
                if data[col].str.contains('%').any():
                    data[col] = data[col].str.rstrip('%').astype(float) / 100
            except:
                pass
    
    # Convert categorical Yes/No columns to binary
    for col in data.columns:
        if data[col].dtype == object:
            try:
                data[col] = data[col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'YES': 1, 'NO': 0})
            except:
                pass

    # Handle date columns if they exist
    date_columns = ['DOB']
    for col in date_columns:
        if col in data.columns:
            try:
                data[col] = pd.to_datetime(data[col])
                # Extract year, month from date
                data[f'{col}_year'] = data[col].dt.year
                data[f'{col}_month'] = data[col].dt.month
                # Drop original date column
                data = data.drop(col, axis=1)
            except:
                pass
    
    # Clean and convert Credit Score column
    if 'Credit Score' in data.columns:
        try:
            data['Credit Score'] = pd.to_numeric(data['Credit Score'], errors='coerce')
        except:
            pass
            
    # Clean and convert Income columns
    income_cols = ['Annual Income', 'Income', 'FamilyIncome']
    for col in income_cols:
        if col in data.columns:
            try:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            except:
                pass
    
    # Clean loan amount columns
    loan_amount_cols = ['LoanAmount', 'Loan Amount Requested']
    for col in loan_amount_cols:
        if col in data.columns:
            try:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            except:
                pass
                
    # Calculate derived features that might be useful
    if 'LoanAmount' in data.columns and 'Annual Income' in data.columns:
        data['Loan_to_Income_Ratio'] = data['LoanAmount'] / data['Annual Income']
        # Handle infinite values from division by zero
        data['Loan_to_Income_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if 'Credit Score' in data.columns and 'Existing Loans' in data.columns:
        data['Credit_Score_to_Loans'] = data['Credit Score'] / (data['Existing Loans'] + 1)
        # Handle infinite values from division by zero
        data['Credit_Score_to_Loans'].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return data

def select_features(data):
    """
    Select the most relevant features for the model
    """
    # Identify numeric and categorical columns
    numeric_features = []
    categorical_features = []
    
    # Common important features across loan types
    important_features = [
        'Credit Score', 
        'Annual Income', 
        'Income',
        'Age', 
        'Job Stability(years)',
        'Existing Loans',
        'Existing EMIs',
        'LoanAmount', 
        'Loan Amount Requested',
        'Employment Type',
        'Defaults',
        'RepaymentDuration',
        'Loan_to_Income_Ratio',
        'Credit_Score_to_Loans',
        'Gender',
        'Marital Status',
        'Number of Dependents',
        'ITRAvailable',
        'Purpose of Loan'
    ]
    
    # Add loan-type specific features
    # Gold loan specific features
    gold_features = [
        'Gold Purity (karats)',
        'Min Gold Weight (g)',
        'Gold Valuation Certificate',
        'Gold Insurance',
        'Gold Storage'
    ]
    
    # Car loan specific features
    car_features = [
        'Car brand',
        'Car insurance available',
        'Down Payment Made'
    ]
    
    # Educational loan specific features
    edu_features = [
        'CourseType',
        'Institution',
        'Degree',
        'Marks12',
        'ExamScore',
        'AdmissionStatus',
        'TotalCost',
        'TuitionFees',
        'FamilyContribution'
    ]
    
    # Combine all features
    all_features = important_features + gold_features + car_features + edu_features
    
    # Filter features that actually exist in the dataframe
    existing_features = [f for f in all_features if f in data.columns]
    
    # Print the number of features found
    print(f"Found {len(existing_features)} features out of {len(all_features)} possible features")
    
    # Separate numeric and categorical features
    for feature in existing_features:
        if pd.api.types.is_numeric_dtype(data[feature]):
            numeric_features.append(feature)
        else:
            categorical_features.append(feature)
    
    # Return features and target if available
    if 'Loan status' in data.columns:
        target = 'Loan status'
        return numeric_features, categorical_features, target
    else:
        return numeric_features, categorical_features, None

def build_preprocessing_pipeline(numeric_features, categorical_features):
    """
    Build a preprocessing pipeline for numeric and categorical features
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Only include transformers for feature types that exist
    transformers = []
    if numeric_features:
        transformers.append(('num', numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    return preprocessor

def build_model(preprocessor):
    """
    Build a model pipeline with preprocessing and random forest classifier
    """
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    
    return model

def data_analysis(df):
    """
    Perform basic data analysis to understand the dataset
    """
    print("\n===== DATA ANALYSIS =====")
    print(f"Dataset shape: {df.shape}")
    
    # Check target variable distribution if it exists
    if 'Loan status' in df.columns:
        print("\nTarget variable distribution:")
        print(df['Loan status'].value_counts())
        print(f"Missing values in target: {df['Loan status'].isna().sum()}")
    
    # Check missing values in features
    print("\nMissing values in features:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0].sort_values(ascending=False).head(10))
    
    # Check numeric feature statistics
    print("\nNumeric feature statistics:")
    numeric_cols = df.select_dtypes(include=['number']).columns
    if not numeric_cols.empty:
        print(df[numeric_cols].describe().T[['count', 'mean', 'min', 'max']].head(5))
    
    # Check categorical feature counts
    print("\nCategorical feature value counts (sample):")
    cat_cols = df.select_dtypes(include=['object']).columns
    if not cat_cols.empty:
        for col in list(cat_cols)[:3]:  # Show first 3 categorical columns
            print(f"\n{col}:")
            print(df[col].value_counts().head(3))
    
    print("\n===== END OF DATA ANALYSIS =====\n")

def train_model(data_path):
    """
    Train the model and save it to disk
    """
    # Read data
    print("Reading data...")
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Perform data analysis
        data_analysis(df)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Clean data
    print("Cleaning data...")
    cleaned_data = clean_data(df)
    
    # Select features
    print("Selecting features...")
    numeric_features, categorical_features, target = select_features(cleaned_data)
    
    if target is None or target not in cleaned_data.columns:
        print("Target variable 'Loan status' not found in the dataset.")
        return
    
    # Check for missing values in target
    if cleaned_data[target].isna().any():
        print(f"Warning: Target variable still has {cleaned_data[target].isna().sum()} missing values after cleaning.")
        print("Dropping rows with missing target values...")
        cleaned_data = cleaned_data.dropna(subset=[target])
        print(f"Dataset shape after dropping missing targets: {cleaned_data.shape}")
    
    # Split data
    print("Splitting data into train and test sets...")
    X = cleaned_data.drop(target, axis=1)
    y = cleaned_data[target]
    
    # Check class balance
    value_counts = y.value_counts()
    print(f"Class distribution: {dict(value_counts)}")
    if len(value_counts) < 2:
        print("Error: Target variable has only one class. Cannot train a classifier.")
        return
        
    # Create feature lists for model based on available columns
    available_numeric = [col for col in numeric_features if col in X.columns]
    available_categorical = [col for col in categorical_features if col in X.columns]
    
    print(f"Using {len(available_numeric)} numeric features and {len(available_categorical)} categorical features")
    
    # Check if features are adequate
    if len(available_numeric) + len(available_categorical) == 0:
        print("Error: No features available for modeling.")
        return
        
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError as e:
        print(f"Error during train/test split: {e}")
        print("Attempting split without stratification...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build preprocessing pipeline
    print("Building preprocessing pipeline...")
    preprocessor = build_preprocessing_pipeline(available_numeric, available_categorical)
    
    # Build and train model
    print("Building and training model...")
    model = build_model(preprocessor)
    
    try:
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"Model training accuracy: {train_score:.4f}")
        print(f"Model testing accuracy: {test_score:.4f}")
        
        # Get feature importances
        if hasattr(model['classifier'], 'feature_importances_'):
            try:
                # Get feature names after one-hot encoding
                feature_names = []
                for name, transformer, features in preprocessor.transformers_:
                    if name == 'num':
                        # For numeric features
                        feature_names.extend(features)
                    elif name == 'cat' and hasattr(transformer.named_steps.get('onehot', None), 'get_feature_names_out'):
                        # For categorical features
                        cat_features = transformer.named_steps['onehot'].get_feature_names_out(features)
                        feature_names.extend(cat_features)
                
                # Get importances
                importances = model['classifier'].feature_importances_
                
                # Match lengths if there's a mismatch
                if len(importances) == len(feature_names):
                    # Sort features by importance
                    indices = np.argsort(importances)[::-1]
                    
                    print("\nTop 10 most important features:")
                    for i in range(min(10, len(indices))):
                        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
                else:
                    print(f"Warning: Feature names length ({len(feature_names)}) doesn't match importances length ({len(importances)})")
            except Exception as e:
                print(f"Error getting feature importances: {e}")
        
        # Save model and preprocessing information
        print("Saving model and preprocessing information...")
        
        # Save feature lists for use during prediction
        feature_info = {
            'numeric_features': available_numeric,
            'categorical_features': available_categorical
        }
        
        # Save model and feature information
        joblib.dump(model, 'loan_default_model.pkl')
        joblib.dump(feature_info, 'feature_info.pkl')
        
        print("Model training complete and saved to 'loan_default_model.pkl'")
        print("Feature information saved to 'feature_info.pkl'")
        
        return model, feature_info
    
    except Exception as e:
        print(f"Error during model training: {e}")
        return None, None

def predict_loan_approval(model_path, feature_info_path, applicant_data):
    """
    Make predictions for new loan applicants
    
    Parameters:
    model_path (str): Path to the saved model file
    feature_info_path (str): Path to the feature info file
    applicant_data (dict or pd.DataFrame): Applicant information
    
    Returns:
    dict: Prediction results with probabilities
    """
    try:
        # Load model and feature info
        model = joblib.load(model_path)
        feature_info = joblib.load(feature_info_path)
        
        # Convert dict to DataFrame if needed
        if isinstance(applicant_data, dict):
            applicant_data = pd.DataFrame([applicant_data])
        
        # Clean data
        cleaned_data = clean_data(applicant_data)
        
        # Make prediction
        prediction_proba = model.predict_proba(cleaned_data)
        prediction = model.predict(cleaned_data)
        
        # Prepare results
        results = {
            'approved': bool(prediction[0] == 1),
            'approval_probability': float(prediction_proba[0][1]),
            'decline_probability': float(prediction_proba[0][0])
        }
        
        return results
    
    except Exception as e:
        print(f"Error making prediction: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    train_model('Loan_Dataset.csv')