#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Path to your file
file_path = 'C:\\Users\\jsava\\Jupyter excercise\\end_module_assignment\\LAEI-2019-Emissions-Summary-including-Forecast.xlsx'

# Load the 'Emissions by Grid ID' sheet from the excel file LAEI-2019-Emissions-Summary-including-Forecast.xlsx
emissions_by_grid_id = pd.read_excel(file_path, sheet_name='Emissions by Grid ID')

# Data exploration
print(emissions_by_grid_id.head())
print(emissions_by_grid_id.info())
print(emissions_by_grid_id.describe())
print(emissions_by_grid_id.isnull().sum())


# In[ ]:


# Filter the dataset for the year 2019
data_2019 = emissions_by_grid_id[emissions_by_grid_id['Year'] == 2019]

print(data_2019.head())
print(data_2019['Year'].describe())


# In[ ]:


print(data_2019.info())


# In[ ]:


print(data_2019.isnull().sum())


# In[ ]:


# Save the 2019 data to a new CSV file
data_2019.to_csv('emissions_2019_only.csv', index=False)


# In[ ]:


# Visualisation missing data
plt.figure(figsize=(12, 8))
sns.heatmap(data_2019.isnull(), cbar=False, cmap='viridis')
plt.title('Heatmap of Missing Values')
plt.show()

missing_counts = data_2019.isnull().sum()
missing_counts = missing_counts[missing_counts > 0].sort_values()
missing_counts.plot(kind='barh', figsize=(10, 6))
plt.title('Number of Missing Values per Column')
plt.xlabel('Number of Missing Values')
plt.ylabel('Columns')
plt.show()


# In[ ]:


# deleting column with more than 50% missing data
threshold = len(data_2019) * 0.5  
data_2019 = data_2019.dropna(thresh=threshold, axis=1)

print(data_2019.columns)


# In[ ]:


print(data_2019.describe())


# In[ ]:


print(data_2019.isnull().sum())


# In[ ]:


print(data_2019.info())


# In[ ]:


# Correlation matrix for pullutant
corr_matrix = data_2019[['co2', 'nox', 'pm10', 'pm2.5']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix for Pollutants')
plt.show()


# In[ ]:


# Percentage of missing values
missing_percent = data_2019[['co2', 'nox', 'pm10', 'pm2.5']].isnull().mean() * 100
print("Percentage of Missing Values:\n", missing_percent)


# In[ ]:


# impute missing values using median for pm10 and pm2.5
data_2019['pm10'].fillna(data_2019['pm10'].median(), inplace=True)
data_2019['pm2.5'].fillna(data_2019['pm2.5'].median(), inplace=True)


# In[ ]:


print(data_2019[['co2', 'nox']].isnull().sum())


# In[ ]:


# Investigation of nature of missing data for co2 and nox
missing_analysis = pd.DataFrame({
    'CO2_missing': data_2019['co2'].isna(),
    'NOx_missing': data_2019['nox'].isna()
})

pattern_counts = missing_analysis.value_counts()
print("Missing value patterns:")
print(pattern_counts)


# In[ ]:


# Using Linear Regression for imputation first when we have nox but not co2 then when we have co2 but not nox
data_2019_imputed = data_2019.copy()

mask_co2 = data_2019_imputed['co2'].isna() & data_2019_imputed['nox'].notna()


# Create and train our linear regression model
if mask_co2.any():
    # Create a new linear regression model
    model = LinearRegression()
    
    complete_cases = data_2019_imputed['co2'].notna() & data_2019_imputed['nox'].notna()
    
    
    X_train = data_2019_imputed.loc[complete_cases, ['nox']].values.reshape(-1, 1)
    y_train = data_2019_imputed.loc[complete_cases, 'co2']
    
    # Train and fit the model 
    model.fit(X_train, y_train)
    
    print("\nLinear Regression Parameters:")
    print(f"Slope: {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    
    X_missing = data_2019_imputed.loc[mask_co2, ['nox']].values.reshape(-1, 1)
    predicted_co2 = model.predict(X_missing)
    
    
    data_2019_imputed.loc[mask_co2, 'co2'] = predicted_co2

# Handling cases where both values are missing 
both_missing = data_2019_imputed['co2'].isna() & data_2019_imputed['nox'].isna()
print("\nCases where both values are missing:", both_missing.sum())

if both_missing.any():
    
    co2_mean = data_2019_imputed['co2'].mean()
    nox_mean = data_2019_imputed['nox'].mean()
    
    print("\nImputation values for completely missing cases:")
    print(f"CO2 mean: {co2_mean:.2f}")
    print(f"NOx mean: {nox_mean:.2f}")
    
    
    data_2019_imputed.loc[both_missing, 'co2'] = co2_mean
    data_2019_imputed.loc[both_missing, 'nox'] = nox_mean

# Verification and comparison
remaining_missing = data_2019_imputed[['co2', 'nox']].isna().sum()
print("\nRemaining missing values after imputation:")
print(remaining_missing)


original_correlation = data_2019[['co2', 'nox']].corr().iloc[0,1]
imputed_correlation = data_2019_imputed[['co2', 'nox']].corr().iloc[0,1]

print("\nCorrelation between CO2 and NOx:")
print(f"Original correlation: {original_correlation:.3f}")
print(f"Correlation after imputation: {imputed_correlation:.3f}")

# Visual result

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(data=data_2019, x='co2', y='nox', alpha=0.5)
plt.title('Original Data\n(excluding missing values)')

plt.subplot(1, 2, 2)
sns.scatterplot(data=data_2019_imputed, x='co2', y='nox', alpha=0.5)
plt.title('After Imputation')

plt.tight_layout()
plt.show()


# In[ ]:


data_2019.loc[:, 'co2'] = data_2019_imputed['co2']
data_2019.loc[:, 'nox'] = data_2019_imputed['nox']


# In[ ]:


# Plot the distribution of co2 and nox after imputation
plt.figure(figsize=(10, 6))
sns.histplot(data_2019['co2'], kde=True, bins=30, color='blue')
plt.title('Distribution of CO2 After Imputation')
plt.xlabel('CO2 (tonnes/annum)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data_2019['nox'], kde=True, bins=30, color='blue')
plt.title('Distribution of nox After Imputation')
plt.xlabel('nox (tonnes/annum)')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


# More visual data exploration
pollutants = ['pm10', 'pm2.5']

for pollutant in pollutants:
    plt.figure(figsize=(10, 6))
    sns.histplot(data_2019[pollutant], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {pollutant}')
    plt.xlabel(f'{pollutant} (tonnes/annum)')
    plt.ylabel('Frequency')
    plt.show()


# In[ ]:


# Log transformation to reduce skewiness
data_2019.loc[:, 'log_pm10'] = np.log1p(data_2019['pm10'])
data_2019.loc[:, 'log_pm2.5'] = np.log1p(data_2019['pm2.5'])
data_2019.loc[:, 'log_co2'] = np.log1p(data_2019['co2'])
data_2019.loc[:, 'log_nox'] = np.log1p(data_2019['nox'])


# In[ ]:


# Updated visualisation
pollutants = ['log_pm10', 'log_pm2.5', 'log_co2', 'log_nox']
for pollutant in pollutants:
    plt.figure(figsize=(10, 6))
    sns.histplot(data_2019[pollutant], kde=True, bins=30, color='green')
    plt.title(f'Distribution of {pollutant}')
    plt.xlabel(pollutant)
    plt.ylabel('Frequency')
    plt.show()


# In[ ]:


#More data exploration
top_emitters = data_2019.nlargest(10, 'co2')
print(top_emitters[['Borough', 'Sector', 'Source', 'co2']])


# In[ ]:


sector_analysis = data_2019.groupby('Sector')[['co2', 'nox', 'pm10', 'pm2.5']].mean()
print(sector_analysis)


# In[ ]:


# More data visualisation
sector_co2 = data_2019.groupby('Sector')['co2'].mean().sort_values(ascending=False)
sector_co2.plot(kind='bar', figsize=(12, 6), color='blue', title='Sector-wise Average CO2 Emissions')
plt.xlabel('Sector')
plt.ylabel('CO2 Emissions (tonnes/annum)')
plt.show()

pollutant_by_sector = data_2019.groupby('Sector')[['co2', 'nox', 'pm10', 'pm2.5']].mean()
plt.figure(figsize=(12, 8))
sns.heatmap(pollutant_by_sector, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title('Sector-Wise Emissions for Different Pollutants')
plt.xlabel('Pollutants')
plt.ylabel('Sector')
plt.show()


# In[ ]:


nox_by_zone = data_2019.groupby('Zone')['nox'].sum().sort_values(ascending=False)
print(nox_by_zone)


# In[ ]:


vmin = np.percentile(data_2019['nox'], 10)  
vmax = np.percentile(data_2019['nox'], 90)
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    data_2019['Easting'], data_2019['Northing'], c=data_2019['nox'], cmap='viridis', alpha=0.7, s=10, vmin=vmin, vmax=vmax
)
plt.colorbar(scatter, label='NOx Emissions (tonnes/annum)')
plt.title('Geographic Distribution of NOx Emissions')
plt.xlabel('Easting')
plt.ylabel('Northing')
plt.show()


# In[ ]:


vmin_value = np.percentile(data_2019['co2'], 10) 
vmax_value = np.percentile(data_2019['co2'], 10)  

plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    data_2019['Easting'], data_2019['Northing'],
    c=data_2019['co2'], cmap='coolwarm', alpha=0.7, s=10,
    vmin=vmin_value, vmax=vmax_value 
)
plt.colorbar(scatter, label='CO2 Emissions (tonnes/annum)')
plt.title('Geographic Distribution of CO2 Emissions')
plt.xlabel('Easting')
plt.ylabel('Northing')
plt.show()



# In[ ]:


pm_emissions_by_borough = data_2019.groupby('Borough')[['pm10', 'pm2.5']].sum().sort_values(by='pm10', ascending=False)
print(pm_emissions_by_borough)


# In[ ]:


pm_emissions_by_borough.plot(kind='bar', figsize=(12, 8), color=['blue', 'green'])
plt.title('PM10 and PM2.5 Emissions by Borough')
plt.xlabel('Borough')
plt.ylabel('Emissions (tonnes/annum)')
plt.legend(['PM10', 'PM2.5'])
plt.show()


# In[ ]:


print(data_2019)


# In[ ]:


# Dropping columns not useful for ML model
prepared_data = data_2019.drop(columns=['Grid ID 2019', 'LAEI 1km2 ID', 'Year', 'Emissions Unit', 'co2', 'nox', 'pm10', 'pm2.5'])

print(prepared_data.head())
print(prepared_data.info())


# In[ ]:


#Encoding for categorical value with OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_cats = encoder.fit_transform(prepared_data[['Borough', 'Zone', 'Main Source Category', 'Sector', 'Source']])

encoded_columns = encoder.get_feature_names_out(['Borough', 'Zone', 'Main Source Category', 'Sector', 'Source'])
encoded_df = pd.DataFrame(encoded_cats, columns=encoded_columns, index=prepared_data.index)

final_data = pd.concat([prepared_data.drop(['Borough', 'Zone', 'Main Source Category', 'Sector', 'Source'], axis=1),
                        encoded_df], axis=1)


# In[ ]:


#Confirm result
print(final_data.isnull().sum())
print(final_data.head())          
print(final_data.info())


# In[ ]:


#Scaling numerical features
numeric_features = ['Easting', 'Northing', 'log_co2', 'log_nox', 'log_pm10', 'log_pm2.5']

scaler = StandardScaler()
final_data[numeric_features] = scaler.fit_transform(final_data[numeric_features])


# In[ ]:


#Final confirmation before ML
print(final_data.shape)
print(final_data.dtypes)
print(final_data.isnull().sum())
print(final_data.head())


# In[ ]:


final_data.to_csv('emission_summary_cleaned.csv', index=False)
print('Dataset saved')

