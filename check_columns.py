import pymysql
from dotenv import load_dotenv
import os

load_dotenv()

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
        cursor.execute("DESCRIBE phones")
        columns = cursor.fetchall()
        print("=== DATABASE COLUMNS ===")
        for col in columns:
            print(f"Column: {col['Field']} | Type: {col['Type']}")
            
        print("\n=== SAMPLE DATA ===")
        cursor.execute("SELECT * FROM phones LIMIT 1")
        sample = cursor.fetchone()
        for key, value in sample.items():
            print(f"{key}: {value}")
            
except Exception as e:
    print(f"Error: {e}")