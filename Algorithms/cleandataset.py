import pandas as pd

# Load the diabetes dataset
df = pd.read_csv("diabetes.csv")

# Replace 0s with NaN for columns except 'Pregnancies' and 'Outcome'
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
df[cols_to_replace] = df[cols_to_replace].replace(0, pd.NA)

# Remove duplicates
df = df.drop_duplicates()

# Drop rows with missing values
df_cleaned = df.dropna()

# Save the cleaned DataFrame to a new CSV file
df_cleaned.to_csv("newdiabetes.csv", index=False)

print("File 'newdiabetes.csv' has been created with rows containing missing values nad duplicates removed.")
