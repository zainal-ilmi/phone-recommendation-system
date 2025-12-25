# HYBRID_FORMULA.md

Hybrid Similarity Formula Specification

Overview
This document specifies the exact mathematical formula for calculating similarity between two smartphones using a hybrid approach combining Jaccard Similarity (for categorical data) and Manhattan Distance (for numerical data).

1. Categorical Similarity (Jaccard)

Features Used:

- Brand (e.g., "Apple", "Samsung")
- Os (e.g., "iOS", "Android")
- Stok_tersedia (e.g., "TRUE", "FALSE")


# Calculation Method:
Binary Jaccard Similarity per feature, then average.

# For each categorical feature, calculate binary similarity
jaccard_brand = 1 if phoneA.Brand == phoneB.Brand else 0
jaccard_os = 1 if phoneA.Os == phoneB.Os else 0
jaccard_stock = 1 if phoneA.Stok_tersedia == phoneB.Stok_tersedia else 0

# Calculate overall categorical score (0-1)
categorical_score = (jaccard_brand + jaccard_os + jaccard_stock ) / 3

# Example:
- Phone A: {Brand: "Apple", Os: "iOS", Stock: "TRUE"}
- Phone B: {Brand: "Apple", Os: "iOS", Stock: "FALSE}
- Calculation: (1 + 1 + 0 ) / 3 = 0.67

2. Numerical Similarity (Manhattan)

Features Used:

- Harga (Price in IDR)
- Ram (GB)
- Memori_internal (Storage in GB)
- Ukuran_layar (Screen size in inches)
- Kapasitas_baterai (Battery in mAh)
- Rating_pengguna (Rating 1.0-5.0)
- Year (Release year)
- Resolusi_kamera (e.g., "12MP", "48MP", "108MP")

# Calculation Method:
Min-Max Normalization → Manhattan Distance → Convert to Similarity

# Step 1: Normalize each numerical feature (0-1 scale)

def normalize_value(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# Normalize all numerical features for both phones
norm_price_A = normalize_value(phoneA.Harga, min_price, max_price)
norm_ram_A = normalize_value(phoneA.Ram, min_ram, max_ram)
norm_storage_A = normalize_value(phoneA.Memori_internal, min_storage, max_storage)
norm_screen_A = normalize_value(phoneA.Ukuran_layar, min_screen, max_screen)
norm_battery_A = normalize_value(phoneA.Kapasitas_baterai, min_battery, max_battery)
norm_rating_A = normalize_value(phoneA.Rating_pengguna, min_rating, max_rating)
norm_year_A = normalize_value(phoneA.Year, min_year, max_year)
norm_camera_A = normalize_value(phoneA.Resolusi_kamera, min_camera, max_camera)

norm_price_B = normalize_value(phoneB.Harga, min_price, max_price)
norm_ram_B = normalize_value(phoneB.Ram, min_ram, max_ram)
norm_storage_B = normalize_value(phoneB.Memori_internal, min_storage, max_storage)
norm_screen_B = normalize_value(phoneB.Ukuran_layar, min_screen, max_screen)
norm_battery_B = normalize_value(phoneB.Kapasitas_baterai, min_battery, max_battery)
norm_rating_B = normalize_value(phoneB.Rating_pengguna, min_rating, max_rating)
norm_year_B = normalize_value(phoneB.Year, min_year, max_year)
norm_camera_B = normalize_value(phoneB.Resolusi_kamera, min_camera, max_camera)


# Step 2: Calculate Manhattan Distance

manhattan_distance = (
abs(norm_price_A - norm_price_B) +
abs(norm_ram_A - norm_ram_B) +
abs(norm_storage_A - norm_storage_B) +
abs(norm_screen_A - norm_screen_B) +
abs(norm_battery_A - norm_battery_B) +
abs(norm_rating_A - norm_rating_B) +
abs(norm_year_A - norm_year_B)
abs(norm_camera_A - norm_camera_B)
)

# Step 3: Convert Distance to Similarity (0-1)

numerical_similarity = 1 - (manhattan_distance / 8) # 8 numerical features

# Ensure similarity is within 0-1 range
numerical_similarity = max(0, min(1, numerical_similarity))

# Min-Max Ranges (from cleaned dataset):
Calculate these from cleaned_phones.csv:

- min_price, max_price
- min_ram, max_ram
- min_storage, max_storage
- min_screen, max_screen
- min_battery, max_battery
- min_rating, max_rating
- min_year, max_year
- min_camera, max_camera

3. Hybrid Combination

# Weighted Combination:

# Initial weights using equal weights (We gave equal importance to both feature types)
weight_categorical = 0.5
weight_numerical = 0.5

# Final similarity score (0-1)
total_similarity = (
    weight_categorical * categorical_score + 
    weight_numerical * numerical_similarity
)

# Final Output:
- Range: 0.0 to 1.0
- 0.0: Completely dissimilar phones
- 1.0: Identical phones

4. Complete Function Signature

def calculate_similarity(phoneA, phoneB, min_max_ranges):
    """
    Calculate hybrid similarity between two phones.
    
    Args:
        phoneA (dict): First phone's attributes
        phoneB (dict): Second phone's attributes  
        min_max_ranges (dict): Min/max values for normalization
    
    Returns:
        float: Similarity score between 0.0 and 1.0
    """
    # Implementation as specified above
    pass

5. Example Usage

# Example phones
phone1 = {
    'Brand': 'Apple', 'Os': 'iOS', 'Stok_tersedia': 'TRUE', 'Resolusi_kamera': '48MP',
    'Harga': 15000000, 'Ram': 6, 'Memori_internal': 128, 'Ukuran_layar': 6.1,
    'Kapasitas_baterai': 4000, 'Rating_pengguna': 4.5, 'Year': 2023
}

phone2 = {
    'Brand': 'Apple', 'Os': 'iOS', 'Stok_tersedia': 'FALSE', 'Resolusi_kamera': '48MP', 
    'Harga': 12000000, 'Ram': 4, 'Memori_internal': 64, 'Ukuran_layar': 5.5,
    'Kapasitas_baterai': 3500, 'Rating_pengguna': 4.2, 'Year': 2022
}

# Calculate similarity
similarity_score = calculate_similarity(phone1, phone2, min_max_ranges)
print(f"Similarity: {similarity_score:.3f}")