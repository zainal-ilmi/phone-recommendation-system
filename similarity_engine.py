import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pymysql
from dotenv import load_dotenv
import os

load_dotenv()

class PhoneSimilarityEngine:
    def __init__(self):
        self.phones_df = None
        self.scaler = MinMaxScaler()
        self.load_data_from_mysql()
        
    def load_data_from_mysql(self):
        """Load phone data from MySQL database"""
        try:
            connection = pymysql.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                user=os.getenv('DB_USER', 'root'),
                password=os.getenv('DB_PASSWORD', 'root'),
                database=os.getenv('DB_NAME', 'phone_recommendation'),
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM phones")
                phones_data = cursor.fetchall()
                
            self.phones_df = pd.DataFrame(phones_data)
            print(f"Loaded {len(self.phones_df)} phones from database")
            
            # Map Indonesian column names to English
            self.map_column_names()
            
            # Preprocess numerical data
            self.preprocess_data()
            
        except Exception as e:
            print(f"Error loading data from MySQL: {e}")
    
    def map_column_names(self):
        """Map Indonesian column names to standard English names"""
        column_mapping = {
            'Id_hp': 'id',
            'Nama_hp': 'model',
            'Brand': 'brand',
            'Harga': 'price',
            'Ram': 'ram',
            'Memori_internal': 'storage',
            'Ukuran_layar': 'screen_size',
            'Kapasitas_baterai': 'battery_capacity',
            'Resolusi_kamera': 'main_camera',
            'Os': 'os',
            'Rating_pengguna': 'user_rating',
            'Year': 'year',
            'Stok_tersedia': 'stock_available'
        }
        
        # Rename columns
        self.phones_df = self.phones_df.rename(columns=column_mapping)
        print(f"Mapped columns: {column_mapping}")
        
        # Ensure ID is properly converted to integer
        if 'id' in self.phones_df.columns:
            self.phones_df['id'] = pd.to_numeric(self.phones_df['id'], errors='coerce').fillna(0).astype(int)
            print("✅ Converted ID column to integers")
        
        # Extract camera MP from text (e.g., "48MP" -> 48)
        if 'main_camera' in self.phones_df.columns:
            self.phones_df['main_camera'] = self.phones_df['main_camera'].str.extract(r'(\d+)').astype(float)
            print("Extracted camera MP from text")
        
        # Convert stock available to boolean
        if 'stock_available' in self.phones_df.columns:
            self.phones_df['stock_available'] = self.phones_df['stock_available'].astype(str).str.lower().map({'true': True, 'false': False})
    
    def preprocess_data(self):
        """Preprocess the phone data for similarity calculations"""
        if self.phones_df.empty:
            return
            
        print("Available columns after mapping:", self.phones_df.columns.tolist())
        
        # Ensure correct data types for numerical columns
        numerical_columns = ['price', 'ram', 'storage', 'screen_size', 'battery_capacity', 'main_camera', 'user_rating', 'year']
        for col in numerical_columns:
            if col in self.phones_df.columns:
                self.phones_df[col] = pd.to_numeric(self.phones_df[col], errors='coerce')
        
        # Fill missing values with reasonable defaults
        fill_values = {
            'screen_size': 6.0,
            'battery_capacity': 4000,
            'main_camera': 48,
            'user_rating': 3.0,
            'year': 2020,
            'model': '',
            'os': 'Unknown'
        }
        
        for col, default_value in fill_values.items():
            if col in self.phones_df.columns:
                self.phones_df[col] = self.phones_df[col].fillna(default_value)
        
        # Store original values for display (before normalization)
        self.phones_df['price_original'] = self.phones_df['price'].copy()
        self.phones_df['ram_original'] = self.phones_df['ram'].copy()
        self.phones_df['storage_original'] = self.phones_df['storage'].copy()
        self.phones_df['screen_size_original'] = self.phones_df['screen_size'].copy()
        self.phones_df['battery_capacity_original'] = self.phones_df['battery_capacity'].copy()
        self.phones_df['main_camera_original'] = self.phones_df['main_camera'].copy()
    

    
    def calculate_similarity(self, input_phone, top_n=5):
        self.load_data_from_mysql()
        """Calculate similarity between input phone and all phones in database - SIMPLIFIED"""
        if self.phones_df.empty:
            return []
        
        similarities = []
        
        # print(f"DEBUG: Input phone - {input_phone}")
        
        # Get ranges for manual normalization
        price_range = self.phones_df['price_original'].max() - self.phones_df['price_original'].min()
        ram_range = self.phones_df['ram_original'].max() - self.phones_df['ram_original'].min()
        storage_range = self.phones_df['storage_original'].max() - self.phones_df['storage_original'].min()
        screen_range = self.phones_df['screen_size_original'].max() - self.phones_df['screen_size_original'].min()
        battery_range = self.phones_df['battery_capacity_original'].max() - self.phones_df['battery_capacity_original'].min()
        camera_range = self.phones_df['main_camera_original'].max() - self.phones_df['main_camera_original'].min()
        rating_range = self.phones_df['user_rating'].max() - self.phones_df['user_rating'].min()
        year_range = self.phones_df['year'].max() - self.phones_df['year'].min()

        
        ranges = [price_range, ram_range, storage_range, screen_range, battery_range, camera_range, rating_range, year_range]
        
        for idx, phone in self.phones_df.iterrows():
            try:
                # Skip phones with invalid data
                if pd.isna(phone.get('brand')) or pd.isna(phone.get('price_original')):
                    continue
                    
                # Categorical similarity (Jaccard) - using brand and OS
                jaccard_brand = 1 if str(input_phone['brand']).lower() == str(phone['brand']).lower() else 0
                jaccard_os = 1 if str(input_phone.get('os', '')).lower() == str(phone.get('os', '')).lower() else 0
                jaccard_stock = 1 if str(input_phone.get('stock_available', '')).lower() == str(phone.get('stock_available', '')).lower() else 0
                jaccard_sim = (jaccard_brand + jaccard_os + jaccard_stock) / 3.0
                
                # Numerical similarity using simple normalized Manhattan distance
                input_values = [
                    float(input_phone['price']),
                    float(input_phone['ram']),
                    float(input_phone['storage']),
                    float(input_phone.get('screen_size', 6.0)),
                    float(input_phone.get('battery_capacity', 4000)),
                    float(input_phone.get('main_camera', 48)),
                    float(input_phone.get('user_rating', 3.0)),
                    float(input_phone.get('year', 2020))
                ]
                
                phone_values = [
                    float(phone['price_original']),
                    float(phone['ram_original']),
                    float(phone['storage_original']),
                    float(phone.get('screen_size_original', 6.0)),
                    float(phone.get('battery_capacity_original', 4000)),
                    float(phone.get('main_camera_original', 48)),
                    float(phone.get('user_rating', 3.0)),
                    float(phone.get('year', 2020))
                ]
                
                # Calculate normalized Manhattan distance
                total_distance = 0
                for i in range(len(input_values)):
                    if ranges[i] > 0:  # Avoid division by zero
                        normalized_diff = abs(input_values[i] - phone_values[i]) / ranges[i]
                        total_distance += normalized_diff
                
                numerical_sim = max(0, 1 - total_distance / len(input_values))
                
                # Combined similarity (weighted)
                total_similarity = (jaccard_sim * 0.5) + (numerical_sim * 0.5)
                
                # Debug first few phones
                # if idx < 3:
                #     print(f"DEBUG: Phone {idx} - {phone['brand']} {phone['model']}, Jaccard: {jaccard_sim:.3f}, Numerical: {numerical_sim:.3f}, Total: {total_similarity:.3f}")
                
                # Remove duplicate brand from display name
                display_name = f"{phone.get('brand', 'Unknown')} {phone.get('model', '').replace(phone.get('brand', '') + ' ', '')}".strip()
                
                # Ensure ID is valid
                phone_id = phone.get('id', 'N/A')
                if pd.isna(phone_id):
                    phone_id = 'N/A'
                
                similarities.append({
                    'id': phone_id,
                    'brand': phone.get('brand', 'Unknown'),
                    'model': phone.get('model', ''),
                    'display_name': display_name,
                    'price': phone.get('price_original', phone.get('price', 0)),
                    'ram': phone.get('ram_original', phone.get('ram', 0)),
                    'storage': phone.get('storage_original', phone.get('storage', 0)),
                    'screen_size': phone.get('screen_size_original', phone.get('screen_size', 6.0)),
                    'battery_capacity': phone.get('battery_capacity_original', phone.get('battery_capacity', 4000)),
                    'main_camera': phone.get('main_camera_original', phone.get('main_camera', 48)),
                    'os': phone.get('os', 'Unknown'),
                    'user_rating': phone.get('user_rating', 3.0),
                    'year': phone.get('year', 2020),
                    'jaccard_similarity': jaccard_sim,
                    'numerical_similarity': numerical_sim,
                    'total_similarity': total_similarity
                })
                    
            except Exception as e:
                print(f"Error calculating similarity for phone {phone.get('id', 'unknown')}: {e}")
                continue
        
        # print(f"DEBUG: Total phones processed: {len(similarities)}")
        
        # Sort by total similarity and return top N
        similarities.sort(key=lambda x: x['total_similarity'], reverse=True)
        return similarities[:top_n]
    
    def get_random_phones(self, n=4):
        """Get random phones from database"""
        if self.phones_df.empty:
            return []
        
        try:
            random_phones = self.phones_df.sample(min(n, len(self.phones_df))).to_dict('records')
            
            # Convert to proper format for display
            result = []
            for phone in random_phones:
                # Remove duplicate brand from display name
                display_name = f"{phone.get('brand', 'Unknown')} {phone.get('model', '').replace(phone.get('brand', '') + ' ', '')}".strip()
                
                result.append({
                    'id': phone.get('id', 'N/A'),
                    'brand': phone.get('brand', 'Unknown'),
                    'model': phone.get('model', ''),
                    'display_name': display_name,
                    'price': phone.get('price_original', phone.get('price', 0)),
                    'ram': phone.get('ram_original', phone.get('ram', 0)),
                    'storage': phone.get('storage_original', phone.get('storage', 0)),
                    'screen_size': phone.get('screen_size_original', phone.get('screen_size', 6.0)),
                    'battery_capacity': phone.get('battery_capacity_original', phone.get('battery_capacity', 4000)),
                    'main_camera': phone.get('main_camera_original', phone.get('main_camera', 48)),
                    'os': phone.get('os', 'Unknown'),
                    'user_rating': phone.get('user_rating', 3.0),
                    'year': phone.get('year', 2020)
                })
            
            return result
        except Exception as e:
            print(f"Error getting random phones: {e}")
            return []
        
    def get_unique_values(self, column_name):
        """Generic function to get unique values from any column"""
        if self.phones_df.empty or column_name not in self.phones_df.columns:
            return []

        try:
            values = sorted(self.phones_df[column_name].unique().tolist())
            return [v for v in values if v and str(v) != 'nan']
        except Exception as e:
            print(f"Error getting {column_name}: {e}")
            return []

    # Then use it:
    def get_all_brands(self):
        return self.get_unique_values('brand')

    def get_all_os(self):
        return self.get_unique_values('os')

    def refresh_data(self):
            """Reload data from database to include new cases"""
            print("🔄 Refreshing phone data from database...")
            self.load_data_from_mysql()