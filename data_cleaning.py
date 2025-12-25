import pandas as pd
from datetime import datetime

def clean_phone_dataset():
    # Load the dataset
    print("📁 Loading dataset...")
    df = pd.read_csv('dataset.handphone.csv')
    
    # Initial inspection
    print("\n🔍 Initial Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    print("\n📊 First 5 rows:")
    print(df.head())
    
    print("\n❓ Missing values:")
    print(df.isnull().sum())
    
    # Create a clean copy
    df_clean = df.copy()
    
    # Step 1: Data Type Conversion
    print("\n🔄 Converting data types...")
    
    # Extract Year from Tahun_rilis
    def extract_year(date_str):
        try:
            return pd.to_datetime(date_str).year
        except:
            return None
    
    df_clean['Year'] = df_clean['Tahun_rilis'].apply(extract_year)
    
    # Ensure numerical columns are numeric
    numerical_columns = ['Harga', 'Ram', 'Memori_internal', 'Ukuran_layar', 
                       'Kapasitas_baterai', 'Rating_pengguna', 'Resolusi_kamera']
    
    for col in numerical_columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Ensure categorical columns are strings
    categorical_columns = ['Brand', 'Os', 'Stok_tersedia']
    
    for col in categorical_columns:
        df_clean[col] = df_clean[col].astype(str)
    
    # Step 2: Feature Selection & Final Clean DataFrame
    print("\n🎯 Selecting features for final dataset...")
    
    # Define which features go to which similarity method
    features_jaccard = ['Brand', 'Os', 'Stok_tersedia']
    features_manhattan = ['Harga', 'Ram', 'Memori_internal', 'Ukuran_layar', 
                         'Kapasitas_baterai', 'Rating_pengguna', 'Year' , 'Resolusi_kamera']
    features_identifier = ['Id_hp', 'Nama_hp']
    
    # Create final cleaned dataset with all needed columns
    final_columns = features_identifier + features_jaccard + features_manhattan
    df_final = df_clean[final_columns].copy()
    
    # Final inspection
    print("\n✅ Final Cleaned Dataset Info:")
    print(f"Shape: {df_final.shape}")
    print(f"Columns: {df_final.columns.tolist()}")
    
    print("\n📋 Data Types:")
    print(df_final.dtypes)
    
    print("\n🔢 Numerical Stats:")
    print(df_final[features_manhattan].describe())
    
    print("\n🏷️ Categorical Values:")
    for col in features_jaccard:
        print(f"{col}: {df_final[col].unique()}")
    
    # Step 3: Export to CSV
    print("\n💾 Exporting cleaned dataset...")
    df_final.to_csv('cleaned_phones.csv', index=False)
    print("✅ cleaned_phones.csv created successfully!")
    
    return df_final

if __name__ == "__main__":
    cleaned_df = clean_phone_dataset()
    
    print("\n" + "="*50)
    print("🎉 DATA CLEANING COMPLETE!")
    print("="*50)
    
    print(f"\n📊 Final dataset shape: {cleaned_df.shape}")
    print(f"📍 First 3 phones:")
    print(cleaned_df[['Id_hp', 'Nama_hp', 'Brand', 'Harga', 'Year']].head(3))