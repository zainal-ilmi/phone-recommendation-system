from flask import Flask, render_template, request, jsonify
from similarity_engine import PhoneSimilarityEngine
import pymysql
import os
from dotenv import load_dotenv
import re

load_dotenv()

app = Flask(__name__)
similarity_engine = PhoneSimilarityEngine()

@app.route('/')
def index():
    brands = similarity_engine.get_all_brands()
    os_list = similarity_engine.get_all_os()
    random_phones = similarity_engine.get_random_phones(4)
    return render_template('index.html', brands=brands, os_list=os_list, random_phones=random_phones)

@app.route('/search_similar', methods=['POST'])
def search_similar():
    try:
        input_phone = {
            'brand': request.json.get('brand'),
            'os': request.json.get('os', ''),
            'stock_available': str(request.json.get('stock_available', 'true')).lower(),  # ADD THIS
            'price': float(request.json.get('price', 0)),
            'ram': int(request.json.get('ram', 0)),
            'storage': int(request.json.get('storage', 0)),
            'screen_size': float(request.json.get('screen_size', 6.0)),
            'battery_capacity': int(request.json.get('battery_capacity', 4000)),
            'main_camera': int(request.json.get('main_camera', 48)),
            'user_rating': float(request.json.get('user_rating', 3.0)),
            'year': int(request.json.get('year', 2020))
        }
        similar_phones = similarity_engine.calculate_similarity(input_phone, 5)
        return jsonify(similar_phones)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/random_phones')
def random_phones():
    try:
        random_phones = similarity_engine.get_random_phones(4)
        return jsonify(random_phones)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/brands')
def brands():
    try:
        brands = similarity_engine.get_all_brands()
        return jsonify(brands)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/os_list')
def os_list():
    try:
        os_list = similarity_engine.get_all_os()
        return jsonify(os_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_case', methods=['POST'])
def save_case():
    try:
        print("=== SAVE CASE ENDPOINT HIT ===")
        
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        new_phone = request.get_json()
        print(f"Data received: {new_phone}")
        
        if not new_phone:
            return jsonify({'error': 'No data received'}), 400
        
        # Get database connection
        connection = pymysql.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', 'root'),
            database=os.getenv('DB_NAME', 'phone_recommendation'),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        
        with connection.cursor() as cursor:
            # Create a new phone name
            original_name = new_phone.get('display_name', new_phone.get('model', 'Unknown Phone'))
            new_phone_name = f"{original_name} - New Case"
            
            print(f"Saving: {new_phone_name}")
            
            # Insert into database
            sql = """
            INSERT INTO phones 
            (Nama_hp, Brand, Harga, Ram, Memori_internal, Ukuran_layar, Kapasitas_baterai, Resolusi_kamera, Os, Rating_pengguna, Year, Stok_tersedia)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(sql, (
                new_phone_name,
                new_phone.get('brand', 'Unknown'),
                int(new_phone.get('price', 0)),
                int(new_phone.get('ram', 0)),
                int(new_phone.get('storage', 0)),
                float(new_phone.get('screen_size', 6.0)),
                int(new_phone.get('battery_capacity', 4000)),
                f"{int(new_phone.get('main_camera', 48))}MP",
                new_phone.get('os', 'Unknown'),
                float(new_phone.get('user_rating', 3.0)),
                int(new_phone.get('year', 2023)),
                True
            ))
        
        connection.commit()
        connection.close()
        
        # 🔄 CRITICAL FIX: Refresh the data in the similarity engine
        print("🔄 Refreshing similarity engine data...")
        similarity_engine.refresh_data()
        
        print("✅ Successfully saved to database and refreshed data!")
        return jsonify({'success': True, 'message': 'New case saved successfully! Data refreshed.'})
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
# Add this route to your existing app.py
@app.route('/api/phones')
def get_all_phones():
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
            cursor.execute("""
                SELECT 
                    Id_hp as id,
                    Nama_hp as model,
                    Brand as brand,
                    Os as os,
                    Resolusi_kamera as main_camera,  # Change this to main_camera
                    Stok_tersedia as stock_available,
                    Harga as price,
                    Ram as ram,
                    Memori_internal as storage,
                    Ukuran_layar as screen_size,
                    Kapasitas_baterai as battery_capacity,
                    Rating_pengguna as user_rating,  # Change this to user_rating
                    Year as year
                FROM phones 
                ORDER BY Id_hp
            """)
            
            phones = cursor.fetchall()
            
            # Fix: Remove duplicate brand names from model and process camera data
            for phone in phones:
                if phone['model'] and phone['brand']:
                    # Remove brand name from the beginning of model if it exists
                    brand_lower = phone['brand'].lower()
                    model_lower = phone['model'].lower()
                    
                    if model_lower.startswith(brand_lower):
                        # Remove brand name and any extra spaces
                        cleaned_model = phone['model'][len(phone['brand']):].strip()
                        phone['model'] = cleaned_model
                
                # Extract camera MP from text (e.g., "48MP" -> 48)
                if phone.get('main_camera'):
                    # import re
                    camera_match = re.search(r'(\d+)', str(phone['main_camera']))
                    if camera_match:
                        phone['main_camera'] = int(camera_match.group(1))
                    else:
                        phone['main_camera'] = 48  # Default value
        
        connection.close()
        
        return jsonify({
            'success': True,
            'data': phones,
            'total_records': len(phones)
        })
        
    except Exception as e:
        print(f"Error fetching phones: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)