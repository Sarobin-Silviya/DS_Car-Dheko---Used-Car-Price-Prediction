#!/usr/bin/env python
# coding: utf-8

# In[143]:


import pandas as pd
import ast  # For safely evaluating strings as Python literals
import os

# Define the folder where the CSV/Excel files are located
data_folder = 'Untitled Folder'  # Ensure this folder name is correct and in quotes

# List of file paths (relative to the notebook's location)
file_paths = [
    os.path.join(data_folder, 'kolkata_cars.xlsx'), 
    os.path.join(data_folder, 'jaipur_cars.xlsx'), 
    os.path.join(data_folder, 'delhi_cars.xlsx'), 
    os.path.join(data_folder, 'chennai_cars.xlsx'), 
    os.path.join(data_folder, 'hyderabad_cars.xlsx'),
    os.path.join(data_folder, 'bangalore_cars.xlsx')
]

# Initialize an empty list to store DataFrames
dataframes = []

# Load each Excel file and append it to the list
for file_path in file_paths:
    df_temp = pd.read_excel(file_path)  # Read each Excel file
    dataframes.append(df_temp)  # Append the DataFrame to the list

# Concatenate all DataFrames into a single DataFrame
df = pd.concat(dataframes, ignore_index=True)

def extract_airbags(features_str):
    # Initialize the variables for driver and passenger airbags
    driver_airbag = 0
    passenger_airbag = 0

    # Safely evaluate the string to convert it into a dictionary
    try:
        features = ast.literal_eval(features_str)  # Convert string to dictionary
    except (ValueError, SyntaxError):
        return driver_airbag, passenger_airbag  # Return 0 for both if conversion fails

    # Loop through the data to find the 'Airbags' details
    for item in features.get('data', []):
        if item.get('heading') == 'Safety':
            # If the heading is 'Safety', check the list for Airbag values
            for feature in item.get('list', []):
                value = feature.get('value', '')
                # Check for driver and passenger airbags
                if 'Driver Air Bag' in value:
                    driver_airbag = 1  # Set to 1 if present
                elif 'Passenger Air Bag' in value:
                    passenger_airbag = 1  # Set to 1 if present

    return driver_airbag, passenger_airbag  # Return both values

# Extract Driver and Passenger Airbags
df[['Driver Airbag', 'Passenger Airbag']] = df['new_car_feature'].apply(extract_airbags).apply(pd.Series)

def extract_new_car_features(features_str):
    # Initialize the dictionary for top features
    new_feature_dict = {
        'Top Features': [],
    }

    # Safely evaluate the string to convert it into a dictionary
    try:
        features = ast.literal_eval(features_str)  # Convert string to dictionary
    except (ValueError, SyntaxError):
        return pd.Series(new_feature_dict)  # Return default values if conversion fails

    # Extract top features
    top_features = features.get('top', [])
    if isinstance(top_features, list):
        # Check if the top features contain dictionaries and get the 'value'
        new_feature_dict['Top Features'] = [item['value'] for item in top_features if isinstance(item, dict) and 'value' in item]
    
    return pd.Series(new_feature_dict)

# Extract new features from 'new_car_feature'
new_car_features_df = df['new_car_feature'].apply(extract_new_car_features)

# Combine the new features with the original DataFrame
df = pd.concat([df, new_car_features_df], axis=1)

# Define the specific features we want to create binary columns for
specific_features = [
    "Power Steering",
    "Power Windows Front",
    "Air Conditioner",
    "Heater",
    "Adjustable Head Lights",
    "Manually Adjustable Exterior Rear View Mirror",
    "Centeral Locking",
    "Child Safety Locks"
]

# Create binary columns for each specific feature
for feature in specific_features:
    # Ensure the 'Top Features' column is a list for proper checking
    df['Top Features'] = df['Top Features'].apply(lambda x: x if isinstance(x, list) else [])
    
    # Create a new column for each feature
    df[feature] = df['Top Features'].apply(lambda x: 1 if feature in x else 0)

# Display the updated DataFrame with new binary columns
print("Updated DataFrame with Binary Columns:")
print(df[specific_features + ['Driver Airbag', 'Passenger Airbag']].head(10))  # Display the first 10 rows for verification

# Drop the unwanted columns before saving
columns_to_drop = ['new_car_details', 'new_car_owner', 'new_car_feature', 'new_car_spec', 'car_links', 'Top Features', 'new_car_specs', 'new_car_overview', 'new_car_detail']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')  # Use 'errors="ignore"' to avoid issues if columns don't exist

# Add a new column 'c_id' with sequential integer values starting from 1
df['c_id'] = range(1, len(df) + 1)

# Save the updated DataFrame into a single CSV file
df.to_csv(os.path.join(data_folder, 'feature_car_data.csv'), index=False)

print("Driver Airbag and Passenger Airbag details, along with binary columns for specific features, added to the DataFrame and saved to CSV without unwanted columns.")


# In[145]:


import pandas as pd
import numpy as np
import os
import ast  # For safe evaluation of strings

# Define the folder where the CSV/Excel files are located
data_folder = 'Untitled Folder'  # Ensure this folder name is correct and in quotes

# List of file paths (relative to the notebook's location)
file_paths = [
    os.path.join(data_folder, 'kolkata_cars.xlsx'), 
    os.path.join(data_folder, 'jaipur_cars.xlsx'), 
    os.path.join(data_folder, 'delhi_cars.xlsx'), 
    os.path.join(data_folder, 'chennai_cars.xlsx'), 
    os.path.join(data_folder, 'hyderabad_cars.xlsx'),
    os.path.join(data_folder, 'bangalore_cars.xlsx')
]

# Initialize an empty list to store DataFrames
dataframes = []

# Function to convert price format
def convert_price(price_str):
    if pd.isna(price_str) or price_str in ('Not available', ''):
        return np.nan
    price_str = price_str.replace('₹ ', '').strip()
    if 'Lakh' in price_str:
        return float(price_str.replace(' Lakh', '').replace(' ', '').replace(',', '')) * 100000
    elif 'Crore' in price_str:
        return float(price_str.replace(' Crore', '').replace(' ', '').replace(',', '')) * 10000000
    return np.nan

# Iterate over each file and extract data
for file in file_paths:
    # Load the Excel file
    df = pd.read_excel(file)
    
    # Check if the 'new_car_detail' column exists
    if 'new_car_detail' in df.columns:
        # Convert string representations of dictionaries to actual dictionaries
        new_car_details_df = pd.json_normalize(df['new_car_detail'].apply(ast.literal_eval))  # Safely evaluate string to dictionary
        
        # Combine new car details with original DataFrame
        df = pd.concat([df, new_car_details_df], axis=1)
       
        # Clean and preprocess the extracted features
        df['kilometers_driven'] = df['km'].str.replace(',', '').astype(int)  # Convert kilometers to integer
        df['kilometers_driven'] = df['kilometers_driven'].apply(round)  # Round off kilometers
        
        df['model_year'] = df['modelYear'].astype(int)  # Convert model year to integer
        
        # Extract and convert price column from the new_car_detail dictionary
        df['price'] = new_car_details_df['price'].apply(lambda x: x if isinstance(x, str) else 'Not available')
        df['price'] = df['price'].apply(convert_price).round(0).astype('Int64')  # Convert price format

        # Extract owner number and add it as a separate column
        df['ownerNo'] = new_car_details_df['ownerNo']  # Extract ownerNo as a separate column

        # Drop unnecessary columns
        df.drop(columns=['new_car_detail', 'new_car_overview', 'new_car_feature', 'new_car_specs', 'km', 'it', 'owner', 'priceActual', 'priceSaving', 'priceFixedText'], inplace=True, errors='ignore')
        
        # Add city information (extracting city name from the file name)
        city_name = os.path.splitext(os.path.basename(file))[0].replace('_cars', '')  # Get the base name of the file and remove '_cars'
        df['city'] = city_name  # Add city column

        # Select only the required columns
        df = df[[  # Adjusted the columns to your original requirement
            'car_links', 'ft', 'bt', 'transmission', 'oem', 'model', 'modelYear',
            'centralVariantId', 'variantName', 'price', 'trendingText.imgUrl',
            'trendingText.heading', 'trendingText.desc', 'kilometers_driven',
            'model_year', 'ownerNo',  # Include the newly added ownerNo column
            'city'
        ]]

        # Append the processed DataFrame to the list
        dataframes.append(df)
        
# Combine all processed DataFrames
combined_df = pd.concat(dataframes, ignore_index=True)

# Add c_id column for merging later
combined_df.insert(0, 'c_id', range(1, len(combined_df) + 1))  # Assigning incremental IDs starting from 1

output_csv_file = os.path.join('.', 'Untitled Folder', 'detail_car_data.csv')
combined_df.to_csv(output_csv_file, index=False)

print(f"Combined DataFrame saved to {output_csv_file}")


# In[4]:


import pandas as pd
import ast  # For safe evaluation of strings
import os
import re  # For extracting numbers from 'Seats'

# Define the folder where the CSV/Excel files are located
data_folder = 'Untitled Folder'  # Ensure this folder name is correct and in quotes

# List of file paths (relative to the notebook's location)
file_paths = [
    os.path.join(data_folder, 'kolkata_cars.xlsx'), 
    os.path.join(data_folder, 'jaipur_cars.xlsx'), 
    os.path.join(data_folder, 'delhi_cars.xlsx'), 
    os.path.join(data_folder, 'chennai_cars.xlsx'), 
    os.path.join(data_folder, 'hyderabad_cars.xlsx'),
    os.path.join(data_folder, 'bangalore_cars.xlsx')
]

# Initialize an empty list to store DataFrames
overview_dataframes = []

# Initialize a global counter for c_id
c_id_counter = 1

# Load Excel files and process 'new_car_overview' column
for file_path in file_paths:
    df = pd.read_excel(file_path)
    
    # Check if 'new_car_overview' exists in the current file
    if 'new_car_overview' in df.columns:
        # Safe literal_eval to handle string to dict conversion
        def safe_literal_eval(val):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return None
        
        # Apply safe evaluation to 'new_car_overview'
        df['new_car_overview'] = df['new_car_overview'].apply(safe_literal_eval)
        
        # Function to extract only the new features ('Insurance Validity' and 'Seats')
        def extract_new_overview_details(overview):
            # Initialize dictionary to store only required features
            new_overview_dict = {
                'Insurance Validity': None,
                'Seats': None
            }

            # If the 'top' field is a list of dictionaries, extract relevant data
            if isinstance(overview, dict) and 'top' in overview:
                for item in overview['top']:
                    key = item['key']
                    value = item.get('value', None)

                    if key == 'Insurance Validity':
                        new_overview_dict['Insurance Validity'] = value
                    
                    if key == 'Seats' and value:
                        # Extract the number of seats using regex to get the numeric value
                        seat_number = re.search(r'\d+', value)
                        if seat_number:
                            # Convert to int directly
                            new_overview_dict['Seats'] = int(seat_number.group(0))  # Ensure it's an int

            return pd.Series(new_overview_dict)

        # Extract new features from 'new_car_overview'
        new_overview_details_df = df['new_car_overview'].apply(extract_new_overview_details)
        
        # Concatenate extracted features with the original DataFrame
        df = pd.concat([df, new_overview_details_df], axis=1)
        
        # Drop the 'new_car_overview' column after extraction
        df.drop(columns=['new_car_overview'], inplace=True)

    # Drop 'new_car_detail' and 'new_car_feature' columns if they exist
    if 'new_car_detail' in df.columns:
        df.drop(columns=['new_car_detail'], inplace=True)
    if 'new_car_feature' in df.columns:
        df.drop(columns=['new_car_feature'], inplace=True)

    # Check if DataFrame is not empty before processing c_id
    if not df.empty:
        # Add c_id column for merging later
        df['c_id'] = range(c_id_counter, c_id_counter + len(df))  # Assigning incremental IDs
        c_id_counter += len(df)  # Update the counter

        # Append the processed DataFrame to the list, including only the necessary columns
        overview_dataframes.append(df[['Insurance Validity', 'Seats', 'c_id']])  # Keep only the specified columns

# Combine all processed DataFrames
combined_overview_df = pd.concat(overview_dataframes, ignore_index=True)

# Convert 'Seats' to integer type to ensure rounding is not necessary
combined_overview_df['Seats'] = combined_overview_df['Seats'].astype('Int64')  # Use 'Int64' for nullable integer type

# Define the output file path
output_csv_file_overview = os.path.join(data_folder, 'overview_car_data.csv')
combined_overview_df.to_csv(output_csv_file_overview, index=False)

print(f"Combined Overview DataFrame saved to {output_csv_file_overview}")

# Print the first few rows of combined_overview_df to verify the output
print(combined_overview_df.head())


# In[10]:


import pandas as pd
import numpy as np
import os
import ast
import re  # Import re for regular expression operations

# Define the folder where the CSV/Excel files are located
data_folder = 'Untitled Folder'  # Ensure this folder name is correct and in quotes

# List of file paths (relative to the notebook's location)
file_paths = [
    os.path.join(data_folder, 'kolkata_cars.xlsx'), 
    os.path.join(data_folder, 'jaipur_cars.xlsx'), 
    os.path.join(data_folder, 'delhi_cars.xlsx'), 
    os.path.join(data_folder, 'chennai_cars.xlsx'), 
    os.path.join(data_folder, 'hyderabad_cars.xlsx'),
    os.path.join(data_folder, 'bangalore_cars.xlsx')
]

# Initialize an empty list to store DataFrames
dataframes = []

# Initialize a global counter for c_id
c_id_counter = 1

# Function to convert price format
def convert_price(price_str):
    if pd.isna(price_str) or price_str in ('Not available', ''):
        return np.nan
    price_str = price_str.replace('₹ ', '').strip()
    if 'Lakh' in price_str:
        return float(price_str.replace(' Lakh', '').replace(' ', '').replace(',', '')) * 100000
    elif 'Crore' in price_str:
        return float(price_str.replace(' Crore', '').replace(' ', '').replace(',', '')) * 10000000
    return np.nan

# Function to safely extract features from specs with a default value
def safe_extract(specs, key, default=None):
    for group in specs:
        if 'list' in group:
            for item in group['list']:
                if item.get('key') == key:
                    return item.get('value', default)
    return default

# Function to extract only numeric values from strings
def extract_numeric(value_str):
    if pd.isna(value_str) or value_str == 'Not specified':
        return np.nan
    numbers = re.findall(r'\d+', value_str)
    return int(numbers[0]) if numbers else np.nan

# Function to safely convert length, width, and height to integers
def safe_convert_to_int(value):
    if isinstance(value, str):
        numeric_value = ''.join(filter(str.isdigit, value))
        return int(numeric_value) if numeric_value else 0
    return 0

# Function to safely extract mileage and return as whole number
def extract_mileage_from_top(top_list):
    if isinstance(top_list, list):
        for item in top_list:
            if isinstance(item, dict) and item.get('key') == 'Mileage':  # Look for the 'Mileage' key
                mileage_str = item.get('value', '0 kmpl')  # Default to '0 kmpl' if not found
                mileage_numeric = re.findall(r'\d+\.\d+|\d+', mileage_str)  # Extract both integer and decimal numbers
                if mileage_numeric:
                    return int(round(float(mileage_numeric[0])))  # Convert to float, round, and return as whole number
    return np.nan  # Return NaN if no 'Mileage' found

# Iterate over each file and extract data
for file in file_paths:
    # Load the Excel file
    df = pd.read_excel(file)
    
    if 'new_car_detail' in df.columns:
        new_car_details_df = pd.json_normalize(df['new_car_detail'].apply(ast.literal_eval))
        df = pd.concat([df, new_car_details_df], axis=1)
       
        df['kilometers_driven'] = df['km'].str.replace(',', '').astype(int).round()  # Clean kilometers driven
        df['model_year'] = df['modelYear'].astype(int)  # Clean model year
        
        # Apply the price conversion
        df['price'] = new_car_details_df['price'].apply(lambda x: x if isinstance(x, str) else 'Not available')
        df['price'] = df['price'].apply(convert_price).round(0).astype('Int64')

        if 'new_car_specs' in df.columns:
            specs_df = pd.json_normalize(df['new_car_specs'].apply(ast.literal_eval))
            specs_data = specs_df['data'].tolist()

            # Extract specs like doors, cylinders, etc.
            df['number_of_doors'] = [safe_extract(specs, 'No Door Numbers', default=0) for specs in specs_data]
            df['tyre_type'] = [safe_extract(specs, 'Tyre Type', default='Not specified') for specs in specs_data]
            df['number_of_cylinders'] = [safe_extract(specs, 'No of Cylinder', default=0) for specs in specs_data]
            df['drive_type'] = [safe_extract(specs, 'Drive Type', default='Not specified') for specs in specs_data]
            df['turbo_charger'] = [safe_extract(specs, 'Turbo Charger', default='No') for specs in specs_data]
            df['super_charger'] = [safe_extract(specs, 'Super Charger', default='No') for specs in specs_data]
            df['length(mm)'] = [safe_convert_to_int(safe_extract(specs, 'Length', default='0')) for specs in specs_data]
            df['width(mm)'] = [safe_convert_to_int(safe_extract(specs, 'Width', default='0')) for specs in specs_data]
            df['height(mm)'] = [safe_convert_to_int(safe_extract(specs, 'Height', default='0')) for specs in specs_data]
            df['color'] = [safe_extract(specs, 'Color', default='Not specified') for specs in specs_data]

            # Extract mileage from the 'top' list in the specs
            df['Mileage(Kmpl)'] = [extract_mileage_from_top(specs.get('top', [])) if isinstance(specs, dict) else np.nan for specs in specs_df.to_dict(orient='records')]
            
            # Convert Mileage(Kmpl) to integer type to avoid floating-point representation
            df['Mileage(Kmpl)'] = df['Mileage(Kmpl)'].astype('Int64')  # This will allow nullable integers

        # Drop unnecessary columns except new_car_detail
        df.drop(columns=['new_car_overview', 'new_car_feature', 'new_car_specs', 'km', 'it', 'ownerNo', 'priceActual', 'priceSaving', 'priceFixedText', 'owner'], inplace=True, errors='ignore')
        
        city_name = os.path.splitext(os.path.basename(file))[0].replace('_cars', '')
        df['city'] = city_name
        
        # Add c_id column for merging later
        df['c_id'] = range(c_id_counter, c_id_counter + len(df))  # Assigning incremental IDs
        c_id_counter += len(df)  # Update the counter

        # Select relevant columns
        df = df[[  
            'c_id', 'number_of_doors', 'tyre_type', 'number_of_cylinders', 
            'drive_type', 'turbo_charger', 'super_charger', 'length(mm)', 
            'width(mm)', 'height(mm)', 'color', 'city', 'Mileage(Kmpl)'  # Include Mileage(Kmpl)
        ]]

        dataframes.append(df)

# Combine all processed DataFrames
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined DataFrame to CSV
output_csv_file = os.path.join(data_folder, 'specs_car_data.csv')
combined_df.to_csv(output_csv_file, index=False)

# Display the combined DataFrame
print(combined_df)

# If you want a better visual display in Jupyter Notebook
try:
    from IPython.display import display
    display(combined_df)
except ImportError:
    print("Please run this in a Jupyter Notebook for a better display.")


# In[ ]:


# COMBINING ALL EXTRACTED CSV FILES


# In[195]:


import pandas as pd
import os

# Define the folder where the CSV files are located
data_folder = 'Untitled Folder'

# List of file paths to merge
file_paths = [
    os.path.join(data_folder, 'specs_car_data.csv'),
    os.path.join(data_folder, 'overview_car_data.csv'),
    os.path.join(data_folder, 'detail_car_data.csv'),
    os.path.join(data_folder, 'feature_car_data.csv')
]

# Initialize an empty DataFrame for merging
merged_df = None

# Load and merge each CSV file on 'c_id'
for file in file_paths:
    df = pd.read_csv(file)
    
    # If merged_df is None, initialize it with the first DataFrame
    if merged_df is None:
        merged_df = df
    else:
        # Perform inner merge on 'c_id'
        merged_df = pd.merge(merged_df, df, on='c_id', how='inner')

# Remove duplicate entries based on all columns or specific ones
merged_df = merged_df.drop_duplicates()

# Alternatively, to remove duplicates based on a specific column (e.g., 'c_id'):
# merged_df = merged_df.drop_duplicates(subset=['c_id'])

# Save the merged DataFrame to a new CSV file
output_merged_csv_file = os.path.join(data_folder, 'merged_car_data.csv')
merged_df.to_csv(output_merged_csv_file, index=False)

# Display the merged DataFrame
print(merged_df)

# If you want a better visual display in Jupyter Notebook
try:
    from IPython.display import display
    display(merged_df)
except ImportError:
    print("Please run this in a Jupyter Notebook for a better display.")


# In[196]:


print(merged_df.columns)


# In[197]:


print(merged_df.describe)


# In[198]:


# Display the data types of each column
print(merged_df.dtypes)


# In[ ]:


# Handling Missing Values,Standardising Data Formats,Encoding Categorical Variables,Normalizing Numerical Features,Removing Outliers.


# In[199]:


# Step 1: Convert float columns to int after handling missing values

# Fill missing values in 'Seats' and 'price' columns in merged_df
merged_df['Seats'] = merged_df['Seats'].fillna(0).astype(int)
merged_df['price'] = merged_df['price'].fillna(0).astype(int)
merged_df['Mileage(Kmpl)'] = merged_df['Mileage(Kmpl)'].fillna(0).astype(int)

# Verify the changes
print(merged_df[['Seats', 'price','Mileage(Kmpl)']].dtypes)


# In[200]:


# Verify no missing values remain
print(merged_df.isnull().sum())


# In[201]:


# Fill missing values in 'Insurance Validity' and 'bt' with their mode
merged_df['Insurance Validity'].fillna(merged_df['Insurance Validity'].mode()[0], inplace=True)
merged_df['bt'].fillna(merged_df['bt'].mode()[0], inplace=True)

# Verify no missing values remain
print("\nMissing Values Count After Filling:\n", merged_df.isnull().sum())


# In[202]:


import pandas as pd

# Assuming merged_df is your DataFrame

# Step 1: Remove unwanted columns
columns_to_remove = ['trendingText.imgUrl', 'trendingText.heading', 
                     'trendingText.desc', 'model_year', 'car_links', 
                     'centralVariantId', 'Child Safety Locks','city_y','c_id']
merged_df = merged_df.drop(columns=columns_to_remove)

# Display the updated DataFrame structure
print(merged_df.dtypes)


# In[203]:


# Rename city_x to city
merged_df = merged_df.rename(columns={'city_x': 'city'})
merged_df = merged_df.rename(columns={'Insurance Validity': 'Insurance Type'})
# Display the updated DataFrame structure
print(merged_df.dtypes)


# In[204]:


merged_df = merged_df.rename(columns={'ft': 'Fuel Type'})
merged_df = merged_df.rename(columns={'bt': 'Body Type'})
merged_df = merged_df.rename(columns={'oem': 'original equipment manufacturer'})
print(merged_df.dtypes)


# In[205]:


# Example mapping for drive_type, tyre_type, and oem
drive_type_mapping = {
   'FWD': 'FWD',
    '2WD': '2WD',
    'Not specified': '4WD',
    'AWD': 'AWD',
    '2 WD': '2WD',
    '4WD': '4WD',
    '4X4': '4WD',
    'RWD': 'RWD',
    '4X2': '4WD',
    'Rear Wheel Drive with ESP': 'RWD',
    '4x2': '4WD',
    'FWD ': 'FWD',
    'Front Wheel Drive': 'FWD',
    'Permanent all-wheel drive quattro': 'AWD',
    'RWD(with MTT)': 'RWD',
    '4x4': '4WD',
    'Two Wheel Drive': '2WD',
    '4 WD': '4WD',
    'All Wheel Drive': 'AWD',
    'AWD INTEGRATED MANAGEMENT': 'AWD',
    '2WD ': '2WD'
}

tyre_type_mapping = {
    'Tubeless,Radial': 'Tubeless Radial',
    'Not specified': 'Radial',
    'Tubeless, Radial': 'Tubeless Radial',
    'Tubeless Tyres, Radial': 'Tubeless Radial',
    'Tubeless': 'Tubeless',
    'Radial, Tubless': 'Tubeless Radial',
    'Tubless, Radial': 'Tubeless Radial',
    'Tubeless, Runflat': 'Tubeless Runflat',
    'Tubeless Tyres Mud Terrain': 'Tubeless',
    'Tubeless,Radials': 'Tubeless Radial',
    'Tubeless Tyres': 'Tubeless',
    'tubeless tyre': 'Tubeless',
    'Radial,Tubeless': 'Tubeless Radial',
    'Radial': 'Radial',
    'Runflat Tyres': 'Runflat',
    'Tubeless, Radials': 'Tubeless Radial',
    'Tubeless Radial Tyres': 'Tubeless Radial',
    'Runflat': 'Runflat',
    'Tubeless ': 'Tubeless',
    'Tubeless,Runflat': 'Tubeless Runflat',
    'Tubeless Radial': 'Tubeless Radial',
    'Tubeless,Radial ': 'Tubeless Radial',
    'Run-Flat': 'Runflat',
    'Radial ': 'Radial',
    'Radial with tube': 'Radial Tube',
    'Runflat Tyre': 'Runflat',
    'Runflat,Radial': 'Runflat',
    'Tubeless. Runflat': 'Tubeless Runflat',
    'Radial Tubeless': 'Tubeless Radial',
    'Tubeless Tyres All Terrain': 'Tubeless',
    'Radial Tyres': 'Radial',
    'Tubeless Radials Tyre': 'Tubeless Radial',
    'Tubeless Tyre': 'Tubeless',
    'Tubless,Radial': 'Tubeless Radial',
    'Radial, Tubeless': 'Tubeless Radial'
}

# Apply the mappings
merged_df['drive_type'] = merged_df['drive_type'].replace(drive_type_mapping)
merged_df['tyre_type'] = merged_df['tyre_type'].replace(tyre_type_mapping)


# In[206]:


# Step 1: Identify unique values in 'tyre_type'
print(merged_df['tyre_type'].unique())
# Step 1: Identify unique values in 'drive_type'
print(merged_df['drive_type'].unique())


# In[207]:


print(merged_df['Insurance Type'].unique())
print(merged_df['Fuel Type'].unique())
print(merged_df['Body Type'].unique())


# In[208]:


import pandas as pd

# Mapping dictionary for standardization
color_mapping = {
    'White': 'White',
    'Red': 'Red',
    'Blue': 'Blue',
    'Brown': 'Brown',
    'Silver': 'Silver',
    'Grey': 'Gray',
    'Black': 'Black',
    'Gray': 'Gray',
    'Green': 'Green',
    'Others': 'Silver',
    'Maroon': 'Maroon',
    'Golden': 'Gold',
    'Foliage': 'Green',
    'Sky Blue': 'Blue',
    'Orange': 'Orange',
    'Off White': 'Off White',
    'Bronze': 'Bronze',
    'G Brown': 'Brown',
    'Purple': 'Purple',
    'Golden Brown': 'Brown',
    'Parpel': 'Purple',
    'Yellow': 'Yellow',
    'Outback Bronze': 'Bronze',
    'Cherry Red': 'Red',
    'Sunset Red': 'Red',
    'Silicon Silver': 'Silver',
    'Gold': 'Gold',
    'golden brown': 'Brown',
    'Dark Blue': 'Blue',
    'Technometgrn+Gryroof': 'Green',
    'Light Silver': 'Silver',
    'Out Back Bronze': 'Bronze',
    'Violet': 'Purple',
    'Bright Silver': 'Silver',
    'Porcelain White': 'White',
    'Tafeta White': 'White',
    'Coral White': 'White',
    'Diamond White': 'White',
    'Brick Red': 'Red',
    'Carnelian Red Pearl': 'Red',
    'Urban Titanium Metallic': 'Gray',
    'Silky silver': 'Silver',
    'Mediterranean Blue': 'Blue',
    'Mist Silver': 'Silver',
    'Gravity Gray': 'Gray',
    'Candy White': 'White',
    'Metallic Premium silver': 'Silver',
    'Polar White': 'White',
    'Glistening Grey': 'Gray',
    'Super white': 'White',
    'Deep Black Pearl': 'Black',
    'PLATINUM WHITE PEARL': 'White',
    'Twilight Blue': 'Blue',
    'Caviar Black': 'Black',
    'Pearl Met. Arctic White': 'White',
    'Superior white': 'White',
    'Pearl White': 'White',
    'Sleek Silver': 'Silver',
    'Phantom Black': 'Black',
    'Metallic silky silver': 'Silver',
    'Pearl Arctic White': 'White',
    'Pure white': 'White',
    'Smoke Grey': 'Gray',
    'Fiery Red': 'Red',
    'StarDust': 'Red',
    'Alabaster Silver Metallic - Amaze': 'Silver',
    'Ray blue': 'Blue',
    'Glacier White Pearl': 'White',
    'OUTBACK BRONZE': 'Bronze',
    'Granite Grey': 'Gray',
    'Solid Fire Red': 'Red',
    'Daytona Grey': 'Gray',
    'Metallic Azure Grey': 'Gray',
    'Moonlight Silver': 'Silver',
    'Aurora Black Pearl': 'Black',
    'Fire Brick Red': 'Red',
    'Cashmere': 'Beige',
    'Pearl Snow White': 'White',
    'Minimal Grey': 'Gray',
    'Metallic Glistening Grey': 'Gray',
    'Light Orange': 'Orange',
    'Medium Blue': 'Blue',
    'Alabaster Silver Metallic': 'Silver',
    'Carbon Steel': 'Gray',
    'Cavern Grey': 'Gray',
    'ESPRESO_BRWN': 'Brown',
    'Beige': 'Beige',
    'Magma Grey': 'Gray',
    'Dark Red': 'Red',
    'Falsa Colour': 'Gray',
    'Cherry': 'Red',
    'TAFETA WHITE': 'White',
    'P Black': 'Black',
    'Golden brown': 'Brown',
    'Star Dust': 'Red',
    'METALL': 'Gray',
    'MET ECRU BEIGE': 'Beige',
    'COPPER': 'Brown',
    'TITANIUM': 'Gray',
    'CHILL': 'Gray',
    'TITANIUM GREY': 'Gray',
    'Burgundy': 'Red',
    'Lunar Silver Metallic': 'Silver',
    'SILKY SILVER': 'Silver',
    'MODERN STEEL METALLIC': 'Gray',
    'BERRY RED': 'Red',
    'PREMIUM AMBER METALLIC': 'Gold',
    'R EARTH': 'Brown',
    'PLATINUM SILVER': 'Silver',
    'ORCHID WHITE PEARL': 'White',
    'CARNELIAN RED PEARL': 'Red',
    'POLAR WHITE': 'White',
    'BEIGE': 'Beige',
    'Hip Hop Black': 'Black',
    'Nexa Blue': 'Blue',
    'Passion Red': 'Red',
    'Cirrus White': 'White',
    'Arizona Blue': 'Blue',
    'Galaxy Blue': 'Blue',
    'Silky Silver': 'Silver',
    'Modern Steel Metal': 'Gray',
    'GOLDEN BROWN': 'Brown',
    'Burgundy Red Metallic': 'Red',
    'magma gray': 'Gray',
    'CBeige': 'Beige',
    'Goldan BRWOON': 'Brown',
    'm grey': 'Gray',
    'b red': 'Red',
    'urban titanim': 'Gray',
    'g brown': 'Brown',
    'beige': 'Beige',
    'Rosso Brunello': 'Red',
    'a silver': 'Silver',
    'b grey': 'Gray',
    'Radiant Red M': 'Red',
    'c bronze': 'Brown',
    'Champagne Mica Metallic': 'Gold',
    'Bold Beige Metallic': 'Beige',
    'Starry Black': 'Black',
    'Symphony Silver': 'Silver',
    'Metallic Magma Grey': 'Gray',
    'Not specified': 'Silver',
    'c brown': 'Brown',
    'chill': 'Gray',
    'Modern Steel Metallic': 'Gray',
    'Arctic Silver': 'Silver',
    'O Purple': 'Purple',
    'Other': 'Silver',
    'PLATINUM WHITE': 'White',
    'Flash Red': 'Red',
    'Wine Red': 'Red',
    'Taffeta White': 'White',
    'T Wine': 'Red',
    'Prime Star Gaze': 'Silver'
}

# Mapping and standardizing the 'color' column
merged_df['color'] = merged_df['color'].map(color_mapping)

# Check the DataFrame after mapping and standardization
print("Standardized color values:")
print(merged_df['color'].unique())


# In[209]:


# Define a mapping for standardizing turbo_charger values
turbo_charger_mapping = {
    'no': 'No',
    'NO': 'No',
    'yes': 'Yes',
    'YES': 'Yes',
    'Turbo': 'Yes',
    'twin': 'Yes',
    'Twin': 'Yes'
}

# Standardize turbo_charger values
merged_df['turbo_charger'] = merged_df['turbo_charger'].replace(turbo_charger_mapping)

# Check unique values after mapping
print("Unique values in 'turbo_charger' after mapping:")
print(merged_df['turbo_charger'].unique())


# In[210]:


import pandas as pd

# Define a mapping for standardizing super charger values
super_charger_mapping = {
    'Yes': 'Yes',
    'yes': 'Yes',
    'yEs': 'Yes',
    'NO': 'No',
    'No': 'No',
    'no': 'No'
}

# Standardize super charger values
merged_df['super_charger'] = merged_df['super_charger'].replace(super_charger_mapping)

# Check the unique values after mapping
print("Unique values in 'super_charger' after mapping:")
print(merged_df['super_charger'].unique())


# In[211]:


# Check the unique values in turbo_charger and super_charger
print("Unique values in 'turbo_charger':", merged_df['turbo_charger'].unique())
print("Unique values in 'super_charger':", merged_df['super_charger'].unique())

#  Map the values if necessary (example mapping)
# Adjust the mapping based on the actual values in your columns
turbo_charger_mapping = {
    'Yes': 1,
    'No': 0,
    1: 1,
    0: 0,
    '': 0  # Handle any empty strings if present
}

super_charger_mapping = {
    'Yes': 1,
    'No': 0,
    1: 1,
    0: 0,
    '': 0  # Handle any empty strings if present
}

# Apply the mapping
merged_df['turbo_charger'] = merged_df['turbo_charger'].replace(turbo_charger_mapping).astype(int)
merged_df['super_charger'] = merged_df['super_charger'].replace(super_charger_mapping).astype(int)

#  Print the columns after encoding
print("Columns after encoding:")
print(merged_df.columns)

# Optional: Check the first few rows to confirm changes
print("First few rows after encoding:")
print(merged_df[['turbo_charger', 'super_charger']].head())


# In[212]:


import numpy as np

# Define a mapping for standardizing Insurance Validity values
insurance_validity_mapping = {
    'Zero Dep': 'Comprehensive',       # Change "Zero Dep" to "Comprehensive"
    'Third Party insurance': 'Third Party',  # Change "Third Party insurance" to "Third Party"
    'Not Available': 'Third Party',     # Change "Not Available" to "Third Party"
    '1': 'Third Party',                 # Assuming "1" means "Third Party" insurance
    '2': 'Comprehensive',               # Assuming "2" means "Comprehensive" insurance
}

# First, replace NaN values separately using numpy
merged_df['Insurance Type'].replace(np.nan, 'Third Party', inplace=True)

# Then apply the mapping to standardize the rest of the values
merged_df['Insurance Type'] = merged_df['Insurance Type'].replace(insurance_validity_mapping)

# Check unique values after mapping
print("Unique values in 'Insurance Type' after mapping:")
print(merged_df['Insurance Type'].unique())


# In[213]:


import numpy as np

# Replace NaN values in 'Body Type' column with 'Unknown'
merged_df['Body Type'].replace(np.nan, 'SUV', inplace=True)

# Check unique values after replacement
print("Unique values in 'Body Type' after handling NaN:")
print(merged_df['Body Type'].unique())


# In[214]:


print(merged_df['transmission'].unique())
print(merged_df['city'].unique())
print(merged_df['original equipment manufacturer'].unique())
print(merged_df['model'].unique())
print(merged_df['variantName'].unique())
print(merged_df['Fuel Type'].unique())
print(merged_df['color'].unique())
print(merged_df['tyre_type'].unique())
print(merged_df['drive_type'].unique())
print(merged_df['Body Type'].unique())
print(merged_df['super_charger'].unique())    
print(merged_df['turbo_charger'].unique())
print(merged_df['Insurance Type'].unique())


# In[215]:


merged_df.columns


# In[216]:


print(merged_df['color'].unique())


# In[217]:


print(merged_df.dtypes)


# In[218]:


#APPLYING FREQUENCY ENCODING


# In[ ]:





# In[ ]:


# STORING FREQUENCY ENCODING IN DICTIONARY


# In[220]:


# Frequency encoding function with mapping storage
def frequency_encoding_with_mapping(df, columns):
    frequency_mappings = {}  # Dictionary to store the mappings
    
    for column in columns:
        freq = df[column].value_counts(normalize=True)  # Get frequency of each unique value
        df[f'{column}_freq'] = df[column].map(freq)    # Map the frequencies to a new column
        
        # Store the mapping of original values and their frequencies in a dictionary
        frequency_mappings[column] = freq.reset_index().rename(columns={'index': column, column: f'{column}_freq'})
    
    return df, frequency_mappings

# List of categorical columns to perform frequency encoding
categorical_columns = [
    'transmission', 'city', 'original equipment manufacturer', 'model', 
    'variantName', 'Fuel Type', 'color', 'tyre_type', 'drive_type', 
    'Body Type', 'super_charger', 'turbo_charger', 'Insurance Type'
]

# Perform frequency encoding on all the categorical columns and store the mappings
merged_df, frequency_mappings = frequency_encoding_with_mapping(merged_df, categorical_columns)

# Display the first few rows of the updated DataFrame
print(merged_df[['drive_type', 'drive_type_freq', 'tyre_type', 'tyre_type_freq',
                 'model', 'model_freq', 'variantName', 'variantName_freq',
                 'color', 'color_freq', 'Fuel Type', 'Fuel Type_freq','transmission','transmission_freq','city','city_freq','original equipment manufacturer','original equipment manufacturer_freq','Body Type','Body Type_freq',
                'super_charger','super_charger_freq','turbo_charger','turbo_charger_freq','Insurance Type','Insurance Type_freq']].head(20))

# Example of accessing the mapping for a specific column (e.g., 'model')
print(frequency_mappings['model'])


# In[221]:


frequency_mappings['model']


# In[231]:


frequency_mappings


# In[223]:


print(merged_df.dtypes)


# In[224]:


print(merged_df.head())


# In[225]:


print(merged_df.columns)


# In[226]:


print(merged_df.dtypes)


# In[227]:


print(merged_df.shape)


# In[228]:


# Save the updated DataFrame to a CSV file
merged_df.to_csv('merged_data_before_processing.csv', index=False)

print("Data has been saved ")


# In[229]:


print(merged_df.describe)


# In[230]:


print(merged_df['Seats'])


# In[232]:


import pickle
# Save the dictionary to a pickle file
with open('Frequency_mapping.pkl', 'wb') as f:
    pickle.dump(frequency_mappings, f)

print("Encoding dictionary saved!")


# In[233]:


import pickle

# Load the pickle file
with open('frequency_mapping.pkl', 'rb') as file:
    frequency_mapping = pickle.load(file)

# Inspect the data
print(frequency_mapping)

# Accessing specific data (for example, the transmission frequency proportions)
if 'transmission' in frequency_mapping:
    print("Transmission Frequency Proportions:")
    print(frequency_mapping['transmission'])


# In[264]:


import pickle

# Replace 'your_file.pkl' with the path to your pickle file
with open('frequency_mapping.pkl', 'rb') as file:
    data = pickle.load(file)

# Now you can use 'data' which contains the contents of the pickle file
print(data)


# In[234]:


# List of original categorical columns to remove
columns_to_remove = [
    'transmission', 'city', 'original equipment manufacturer', 'model', 
    'variantName', 'Fuel Type', 'color', 'tyre_type', 'drive_type', 
    'Body Type', 'super_charger', 'turbo_charger', 'Insurance Type'
]

# Remove the original categorical columns from merged_df
merged_df = merged_df.drop(columns=columns_to_remove)

# Display the updated DataFrame to verify removal
print(merged_df.head())

# Save the updated DataFrame to a CSV file (if needed)
merged_df.to_csv('encoded_data_without_original_columns.csv', index=False)

print("Updated data has been saved to 'encoded_data_without_original_columns.csv'")


# In[235]:


print(merged_df.dtypes)


# In[ ]:


#PROCESSING


# In[236]:


get_ipython().system('pip install scikit-learn')


# In[237]:


# Select numerical columns for any zeros
numerical_cols = [ 'length(mm)', 'width(mm)', 'height(mm)', 
                  'price', 'kilometers_driven', 
                  'Mileage(Kmpl)'] 
# Loop over the numerical columns
for col in numerical_cols:
    # Calculate the median of the column, ignoring zeros
    median_value = merged_df[merged_df[col] != 0][col].median()
    
    # Replace zeros with the calculated median
    merged_df[col] = merged_df[col].replace(0, median_value)

# Check the data after replacement
print("Data after replacing zeros with median values:")
print(merged_df[numerical_cols].head())


# In[238]:


from sklearn.preprocessing import MinMaxScaler

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Select numerical columns for normalization
numerical_cols2= [ 'length(mm)', 'width(mm)', 'height(mm)', 
                   'kilometers_driven', 
                  'Mileage(Kmpl)']  

# Apply normalization
merged_df[numerical_cols2] = scaler.fit_transform(merged_df[numerical_cols2])

# Check the updated DataFrame after normalization
print("DataFrame shape after normalization:")
print(merged_df[numerical_cols2].head())


# In[239]:


merged_df.head()


# In[290]:


# Save the MinMaxScaler to a file
scaler_filename = 'minmax_scaler.pkl'
with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)

print(f"MinMaxScaler object saved as {scaler_filename}")


# In[291]:


import pickle

# Replace 'your_file.pkl' with the path to your pickle file
with open('minmax_scaler.pkl', 'rb') as file:
    data = pickle.load(file)

# Now you can use 'data' which contains the contents of the pickle file
print(data)


# In[292]:


from sklearn.preprocessing import MinMaxScaler
import pickle

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Select numerical columns for normalization
numerical_cols2 = ['length(mm)', 'width(mm)', 'height(mm)', 
                   'kilometers_driven', 'Mileage(Kmpl)']  

# Apply normalization
merged_df[numerical_cols2] = scaler.fit_transform(merged_df[numerical_cols2])

# Save the MinMaxScaler to a file
scaler_filename = 'minmax_scaler.pkl'
with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)

print(f"MinMaxScaler object saved as {scaler_filename}")


# In[293]:


# Load the MinMaxScaler from a file
with open(scaler_filename, 'rb') as file:
    loaded_scaler = pickle.load(file)

# Check the attributes of the loaded scaler to see that it has the fitted data
print("Scaler min:", loaded_scaler.data_min_)
print("Scaler scale:", loaded_scaler.scale_)


# In[294]:


sample_data = merged_df[['length(mm)', 'width(mm)', 'height(mm)', 'kilometers_driven', 'Mileage(Kmpl)']].sample(5)  # or any subset
scaled_data = loaded_scaler.transform(sample_data)
print("Scaled sample data:\n", scaled_data)


# In[268]:


print(merged_df[numerical_cols2].describe())


# In[241]:


print(merged_df.dtypes)


# In[242]:


merged_df.head()


# In[244]:


numerical_cols3=['number_of_doors','number_of_cylinders','Seats','modelYear','ownerNo'
,'Driver Airbag','Passenger Airbag','Power Steering','Power Windows Front','Air Conditioner','Heater','Adjustable Head Lights','Manually Adjustable Exterior Rear View Mirror',
'Centeral Locking''length(mm)', 'width(mm)', 'height(mm)', 
                  'price', 'kilometers_driven', 
                  'Mileage(Kmpl)']


# In[246]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn
sns.set(style="whitegrid")

# Plot boxplots for numerical columns to visualize outliers
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(y=merged_df[col])
    plt.title(f'Boxplot for {col}')
    plt.show()

# Plot histograms for numerical columns
n_cols = 3  # Number of columns in the histogram layout
n_rows = (len(numerical_cols) + n_cols - 1) // n_cols  # Calculate number of rows needed

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()  # Flatten the axes array for easy iteration

for i, col in enumerate(numerical_cols):
    axes[i].hist(merged_df[col], bins=20, color='blue', alpha=0.7)
    axes[i].set_title(f'Histogram of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

# Hide any unused subplots
for j in range(i + 1, n_rows * n_cols):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle('Histograms of Numerical Columns', y=1.02)  # Adjust title position
plt.show()


# In[247]:


import pandas as pd


# Select the continuous numerical columns for outlier detection using IQR
continuous_cols = ['length(mm)', 'width(mm)', 'height(mm)', 
                    'kilometers_driven', 'Mileage(Kmpl)']

# Function to remove outliers using IQR for continuous columns
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Determine outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove outliers for continuous variables
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

# Function to handle discrete columns with manual range filtering
def handle_discrete_columns(df):
    # Define ranges for modelYear, ownerNo, number_of_doors, number_of_cylinders, and Seats
    df = df[(df['modelYear'] >= 1980) & (df['modelYear'] <= 2024)]
    df = df[(df['ownerNo'] >= 1) & (df['ownerNo'] <= 5)]
    df = df[(df['number_of_doors'] >= 1) & (df['number_of_doors'] <= 7)]
    df = df[(df['number_of_cylinders'] >= 1) & (df['number_of_cylinders'] <= 12)]
    df = df[(df['Seats'] >= 1) & (df['Seats'] <= 15)]  
    
    return df

# Apply the IQR function for continuous variables
cleaned_df = remove_outliers_iqr(merged_df.copy(), continuous_cols)

# Apply the manual filtering for discrete variables including number_of_cylinders and Seats
cleaned_df = handle_discrete_columns(cleaned_df)

# Check the shape of the DataFrame before and after outlier removal
print(f"Original DataFrame shape: {merged_df.shape}")
print(f"Cleaned DataFrame shape: {cleaned_df.shape}")

# Display the cleaned DataFrame
print(cleaned_df)


# In[248]:


# Save the cleaned DataFrame as a pickle file
cleaned_df.to_pickle("cleaned_data_after_outlier_removal.pkl")


# In[ ]:


# EDA


# In[251]:


import pandas as pd

# Assuming your cleaned DataFrame is named cleaned_df
summary_statistics = cleaned_df.describe(include='all')  # Include all columns for summary statistics
print(summary_statistics)


# In[252]:


# Check the data types of the DataFrame
print(cleaned_df.dtypes)


# In[253]:


print(cleaned_df.tail())


# In[254]:


# Histograms for numerical columns
plt.figure(figsize=(15, 10))
for i, col in enumerate(continuous_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(cleaned_df[col], bins=15, kde=True, color='blue')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


# In[255]:


# Box plots for numerical columns
plt.figure(figsize=(15, 10))
for i, col in enumerate(continuous_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=col, data=cleaned_df)
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()


# In[256]:


# Calculate variance only for numeric columns
numeric_df = cleaned_df.select_dtypes(include=[np.number])

# Identify features with zero variance
low_variance_features = numeric_df.columns[numeric_df.var() == 0]

# Print low variance features
print("Low variance features:", low_variance_features)




# In[257]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Calculate the correlation matrix
correlation_matrix = cleaned_df.corr()

# Step 2: Plot the correlation heatmap (including price, but without log transformation)
plt.figure(figsize=(30, 28))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap including price')
plt.show()


# In[258]:


# Print the correlation matrix
print(correlation_matrix)


# In[259]:


print(cleaned_df.dtypes)


# In[260]:


print(correlation_matrix.isna().sum())  # Check for any NaN values


# In[261]:


from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Define the features (X) and target variable (y)
X = cleaned_df.drop(columns=['price'])  # Dropping the target column
y = cleaned_df['price']  # Target variable

# Check for infinite values only in numeric columns
numeric_X = X.select_dtypes(include=[np.number])  # Select only numeric columns
print("Infinite values in X:", np.isinf(numeric_X).sum().sum())

# Replace infinite values with NaN in numeric columns
numeric_X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaNs with the mean of the column
numeric_X.fillna(numeric_X.mean(), inplace=True)

# If necessary, reassign cleaned numeric values back to X
X[numeric_X.columns] = numeric_X

# Verify again for infinite values
print("Infinite values in X after handling:", np.isinf(X.select_dtypes(include=[np.number])).sum().sum())

# Train a RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Get feature importances
feature_importances = model.feature_importances_

# Create a DataFrame to display the importance scores
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Print the most important features
print(feature_importance_df.head(20)) # Display top 20 important features


# In[262]:


print("X shape:", X.shape) # Number of features in X
print("Importances length:", len(feature_importances))# Number of importances from the model


# In[263]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np



# Get feature importances from the trained model
importances = model.feature_importances_
feature_names = X.columns  

# Check if the length of importances matches the feature names
if len(importances) == len(feature_names):
    print("Lengths match!")
else:
    print(f"Mismatch: X has {len(feature_names)} features, but importances have length {len(importances)}")

# Now create the DataFrame for feature importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort values by importance
importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Optionally, limit to top N features (e.g., top 20)
top_n = 20
importance_df_top = importance_df.head(top_n)

# Plotting the top N features
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df_top)
plt.title('Top Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# If you want to plot all features, you can use this:
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('All Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()


# In[ ]:


#MODEL DEVELOPEMENT 


# In[73]:


cleaned_df.columns


# In[91]:


X_train.isnull().sum()


# In[ ]:


#MODEL COMPARISON 


# In[274]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Ensure that cleaned_df is already defined in your environment
# Define the features (X) and target variable (y)
X = cleaned_df.drop(columns=['price'])  # Dropping the target column
y = cleaned_df['price']  # Target variable

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Train models and evaluate their performance
mse_results = {}
feature_importances = {}

for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    mse_results[model_name] = mse
    print(f"{model_name} MSE: {mse:.4f}")
    
    # Check and store feature importance or coefficients
    if hasattr(model, 'coef_'):  # For Linear Regression
        feature_importances[model_name] = model.coef_
    elif hasattr(model, 'feature_importances_'):  # For Tree-based models
        feature_importances[model_name] = model.feature_importances_

# Print feature importances for each model
for model_name, importances in feature_importances.items():
    print(f"\n{model_name} Feature Importances:")
    for feature, importance in zip(X.columns, importances):
        print(f"{feature}: {importance:.4f}")

# Find the best model based on MSE
best_model_name = min(mse_results, key=mse_results.get)
best_model_mse = mse_results[best_model_name]
print(f"\nBest Model: {best_model_name} with MSE: {best_model_mse:.4f}")


# In[275]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Split the dataset into training and testing sets
X = cleaned_df.drop(columns=['price'])  # Features
y = cleaned_df['price']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
lasso = Lasso()
ridge = Ridge()
rf_model = RandomForestRegressor(random_state=42)

# Cross-validation function
def cross_val_model(model, X, y):
    return cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Perform cross-validation on Random Forest
rf_cv_scores = cross_val_model(rf_model, X_train, y_train)
print(f'Random Forest CV MSE: {-np.mean(rf_cv_scores)}')

# Hyperparameter tuning for Lasso
lasso_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
lasso_grid = GridSearchCV(Lasso(), lasso_params, scoring='neg_mean_squared_error', cv=5)
lasso_grid.fit(X_train, y_train)
print(f'Best Lasso Alpha: {lasso_grid.best_params_["alpha"]}')
print(f'Lasso MSE: {-lasso_grid.best_score_}')

# Hyperparameter tuning for Ridge
ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge_grid = GridSearchCV(Ridge(), ridge_params, scoring='neg_mean_squared_error', cv=5)
ridge_grid.fit(X_train, y_train)
print(f'Best Ridge Alpha: {ridge_grid.best_params_["alpha"]}')
print(f'Ridge MSE: {-ridge_grid.best_score_}')

# Train Random Forest model
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
print(f'Random Forest Test MSE: {rf_mse}')

# Feature importance from Random Forest
feature_importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Random Forest Feature Importances:")
print(feature_importance_df)

# Ensure all 32 features are being used
print(f'Total Features Used: {X.shape[1]}')


# In[276]:


# Display the features used to train the model
features_used = X.columns.tolist()  # Get the list of feature names
print("Features used to train the model:")
print(features_used)


# In[277]:


import matplotlib.pyplot as plt

# MSE values for each model
mse_values = {
    'Linear Regression': 310162466906.2487,
    'Decision Tree': 94348962509.1441,
    'Random Forest': 40246525605.8812,
    'Lasso': 762555604677.6555,
    'Ridge': 763408734812.0348,
    'Random Forest Test': 40246525605.881165,
}

# Plotting the MSE values
plt.figure(figsize=(10, 6))
plt.barh(list(mse_values.keys()), list(mse_values.values()), color='skyblue')
plt.xlabel('Mean Squared Error (MSE)')
plt.title('MSE Comparison of Different Models')
plt.xscale('log')  # Using log scale for better visualization
plt.grid(axis='x')

# Display the plot
plt.show()


# In[279]:


import joblib

# Save the trained Random Forest model
joblib.dump(model, 'random_forest_model.pkl')


# In[280]:


import pickle

# Load the model to check what it is
model_filename = 'random_forest_model.pkl'
with open(model_filename, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

print(type(loaded_model))  # Check the type of the loaded model
print(loaded_model)  # Print the loaded model (if it's not too large)


# In[287]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Assuming cleaned_df is your DataFrame with the features and the target column 'price'
X = cleaned_df.drop(columns=['price'])  # Features
y = cleaned_df['price']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model to a file
model_filename = 'random_forest_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)

print(f"Model saved as {model_filename}")


# In[288]:


# Load the model
with open(model_filename, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Check the type of the loaded model
print(type(loaded_model))  # Should be <class 'sklearn.ensemble._forest.RandomForestRegressor'>


# In[289]:


# Simulate input data for prediction (with the correct features)
# Create a sample input DataFrame with the same feature names
sample_input = pd.DataFrame({
    'number_of_doors': [4],
    'number_of_cylinders': [4],
    'length(mm)': [4000],
    'width(mm)': [1700],
    'height(mm)': [1500],
    'Mileage(Kmpl)': [15],
    'Seats': [5],
    'modelYear': [2020],
    'kilometers_driven': [20000],
    'ownerNo': [1],
    'Driver Airbag': [1],
    'Passenger Airbag': [1],
    'Power Steering': [1],
    'Power Windows Front': [1],
    'Air Conditioner': [1],
    'Heater': [0],
    'Adjustable Head Lights': [1],
    'Manually Adjustable Exterior Rear View Mirror': [0],
    'Centeral Locking': [1],
    'transmission_freq': [0.723145],  # Example frequency values
    'city_freq': [0.177441],
    'original equipment manufacturer_freq': [0.269088],
    'model_freq': [0.044211],
    'variantName_freq': [0.040268],
    'Fuel Type_freq': [0.663640],
    'color_freq': [0.416059],
    'tyre_type_freq': [0.771060],
    'drive_type_freq': [0.667463],
    'Body Type_freq': [0.426694],
    'super_charger_freq': [0],
    'turbo_charger_freq': [1],
    'Insurance Type_freq': [0.587286]
})

# Make a prediction
prediction = loaded_model.predict(sample_input)
print(f"Predicted Price: {prediction[0]}")


# In[284]:


pip install scikit-learn==1.3.0


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




