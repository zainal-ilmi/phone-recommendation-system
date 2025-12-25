# Data Specification for Hybrid CBR System

## Features for Similarity Calculation

### Jaccard Similarity (Categorical Features)
- **Brand**: Manufacturer (Apple, Asus, Huawei, Infinix, OnePlus, Oppo, Realme, Samsung, Vivo, Xiaomi)
- **Os**: Operating System (iOS, Android)
- **Stok_tersedia**: Stock availability (TRUE, FALSE) - binary categorical

### Manhattan Distance (Numerical Features)
- **Harga**: Price in Indonesian Rupiah -- numerical continuous type
- **Ram**: RAM in GB (4, 6, 8, 12, 16 ) -- numerical discreet type
- **Memori_internal**: Internal storage in GB (64, 128, 256, 512) -- numerical discreet type
- **Ukuran_layar**: Screen size in inches -- numerical continuous type
- **Kapasitas_baterai**: Battery capacity in mAh (4000, 4500, 5000, 6000) -- numerical discreet type
- **Rating_pengguna**: User rating (1.0-5.0) -- numerical continuous type
- **Year**: Release year (extracted from Tahun_rilis) (2019, 2020, 2021, 2022, 2023, 2024) -- numerical discreet type
- **Resolusi_kamera**: Camera resolution (12MP, 48MP, 50MP, 64MP, 108MP .) - numerical discreet type

### Identifier Columns (Not used for similarity)
- **Id_hp**: Unique phone ID
- **Nama_hp**: Phone name

## Data Processing Decisions
1. **Tahun_rilis**: Converted from date string to Year only
2. **All numerical features**: Verified as numeric types
3. **No missing values**: Dataset is complete