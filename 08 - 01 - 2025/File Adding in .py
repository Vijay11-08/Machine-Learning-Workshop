# In Dataset We Conclude That Activity all code here

df.isnull().sum()



df['balcony']=df['balcony'].fillna(df['balcony'].mean())
df.isna().sum()

df.society.mode()

df.fillna({'society':'GrrvaGr'},inplace=True)
df


df.society.isna().sum()


df.fillna({'society':'GrrvaGr'},inplace=True)


# Split the 'size' column into two columns: 'Size_Value' and 'Size_Unit'
df[['Size_Value', 'Size_Unit']] = df['size'].str.extract(r'(\d+)\s*(\D+)')

# Convert 'Size_Value' to integer
df['Size_Value'] = pd.to_numeric(df['Size_Value'], errors='coerce').astype('Int64')  # Using 'Int64' for integer handling with NaN

# Display the updated DataFrame with the new columns
print(df[['size', 'Size_Value', 'Size_Unit']].head())
df


numeric_columns = df.select_dtypes(include=['number'])
averages = numeric_columns.mean()
print(averages)

