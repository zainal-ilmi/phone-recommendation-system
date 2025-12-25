from similarity_engine import PhoneSimilarityEngine

def test_similarity():
    engine = PhoneSimilarityEngine()
    
    # Test with a phone that should definitely match
    test_phone = {
        'brand': 'Apple',
        'os': 'iOS', 
        'price': 15000000,
        'ram': 8,
        'storage': 256,
        'screen_size': 6.0,
        'battery_capacity': 4000,
        'main_camera': 48,
        'user_rating': 4.0,
        'year': 2023
    }
    
    print("=== TESTING SIMILARITY ENGINE ===")
    print(f"Total phones in database: {len(engine.phones_df)}")
    print(f"Available brands: {engine.phones_df['brand'].unique()[:10]}")  # First 10 brands
    
    # Check if there are any Apple phones
    apple_phones = engine.phones_df[engine.phones_df['brand'] == 'Apple']
    print(f"Number of Apple phones: {len(apple_phones)}")
    
    if len(apple_phones) > 0:
        print("Sample Apple phones:")
        for idx, phone in apple_phones.head(3).iterrows():
            print(f"  - {phone['brand']} {phone['model']} | Price: {phone.get('price_original', 'N/A')} | RAM: {phone.get('ram_original', 'N/A')}")
    
    # Check actual values in the database
    print("\n=== DATA RANGES ===")
    if len(engine.phones_df) > 0:
        print(f"Price range: {engine.phones_df['price_original'].min()} - {engine.phones_df['price_original'].max()}")
        print(f"RAM range: {engine.phones_df['ram_original'].min()} - {engine.phones_df['ram_original'].max()}")
        print(f"Storage range: {engine.phones_df['storage_original'].min()} - {engine.phones_df['storage_original'].max()}")
        print(f"Screen size range: {engine.phones_df['screen_size_original'].min()} - {engine.phones_df['screen_size_original'].max()}")
        print(f"Battery range: {engine.phones_df['battery_capacity_original'].min()} - {engine.phones_df['battery_capacity_original'].max()}")
        print(f"Camera range: {engine.phones_df['main_camera_original'].min()} - {engine.phones_df['main_camera_original'].max()}")
    
    # Test similarity
    print("\n=== SIMILARITY RESULTS ===")
    results = engine.calculate_similarity(test_phone, 10)
    print(f"Similarity results found: {len(results)}")
    
    for i, result in enumerate(results):
        print(f"{i+1}. {result['brand']} {result['model']} - Similarity: {result['total_similarity']:.3f}")

if __name__ == "__main__":
    test_similarity()