import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

def main ():
    X = pd.read_csv("Desktop/FDS-project/train.csv")
    y = X['Crime_Category']
    X = X.drop('Crime_Category', axis=1)

    # remove column with 80% null values
    X.drop('Cross_Street', axis=1)

    # handle missing data
    X['Victim_Sex'] = X['Victim_Sex'].replace(['H', 'X'], 'Unknown')
    X['Victim_Descent'] = X['Victim_Descent'].fillna('Unknown')
    X['Weapon_Description'] = X['Weapon_Description'].fillna('No Weapon') 
    X['Weapon_Used_Code'] = X['Weapon_Used_Code'].fillna(0) # Weapon_Used_Code is in the range [1,3990], 0 is for missing code
    X['Modus_Operandi'] = X['Modus_Operandi'].fillna('Unknown')

    # data handling
    X['Date_Reported'] = pd.to_datetime(X['Date_Reported'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    X['Date_Occurred'] = pd.to_datetime(X['Date_Occurred'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    X['Year_Reported'] = X.Date_Reported.dt.year
    X['Year_Occurred'] = X.Date_Occurred.dt.year
    X['Month_Reported'] = X.Date_Reported.dt.month
    X['Month_Occurred'] = X.Date_Occurred.dt.month
    X['Day_Reported'] = X.Date_Reported.dt.day
    X['Day_Occurred'] = X.Date_Occurred.dt.day
    X.drop(['Date_Reported', 'Date_Occurred'], axis=1, inplace=True)

    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

    numerical_pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )

    categorical_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        )

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_columns),
        ('cat', categorical_pipeline, categorical_columns)
    ])

    # full pipeline
    pipe = make_pipeline(
        preprocessor
    )

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    #################################################
    # UPLOAD HERE X_train, X_test, y_train, y_test #
    #################################################

    y_encoder = LabelEncoder()
    y_train_encoded = y_encoder.fit_transform(y_train)
    y_test_encoded = y_encoder.transform(y_test)

    # apply the transformations to the data
    X_train_transformed = pipe.fit_transform(X_train)
    X_val_transformed = pipe.transform(X_test)

    