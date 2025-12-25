-- Create database
CREATE DATABASE phone_recommendation;
USE phone_recommendation;

-- Create phones table
CREATE TABLE phones (
    Id_hp INT PRIMARY KEY,
    Nama_hp VARCHAR(50),
    Brand VARCHAR(100),
    Harga DECIMAL(12,2),
    Ram INT,
    Memori_internal INT,
    Ukuran_layar DECIMAL(3,1),
    Kapasitas_baterai INT,
    Resolusi_kamera INT,
    Memori_internal INT,
    Rating_pengguna INT,
    Year VARCHAR(20),
    weight DECIMAL(5,2),
    Os VARCHAR(50),
    chipset VARCHAR(100),
    Stok_tersedia VARCHAR(50)
);

drop table phones;

select * from phones;
use phone_recommendation;
SELECT * FROM phones WHERE Nama_hp LIKE '%Custom%' ORDER BY Id_hp DESC LIMIT 5;

DELETE FROM phones WHERE id_hp is null and Nama_hp LIKE '%New Case%';

SET SQL_SAFE_UPDATES = 1;
