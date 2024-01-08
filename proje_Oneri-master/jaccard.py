import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer

# CSV dosyasından film verilerini yükle
secim = int(input("Hangi veri setini tercih edersiniz:\n 1-)Film \n 2-)Kitap \n 3-)Kıyafet"))
if secim == 1:
    data = pd.read_csv('data.csv', encoding='ISO-8859-9')
elif secim == 2:
    data = pd.read_csv('book_data.csv', encoding='ISO-8859-9')
elif secim == 3:
    data = pd.read_csv('clouth_data1.csv', sep=';', encoding='ISO-8859-9')

print(data.columns)

# Kullanıcının seçtiği özelliklere göre birleştirilmiş özellikleri oluşturan fonksiyon
def create_combined_features(row, selected_features):
    return ' '.join([str(row[feature]) for feature in selected_features])

# Benzerlik hesaplamada kullanılacak özellikleri kullanıcıdan al
selected_features = input("Benzerlik hesaplamak için özellikleri girin (örneğin, title, genre, actors, director, overview): ").split(',')

# Kullanıcının seçtiği özelliklere göre 'combined_features' sütununu oluştur
data['combined_features'] = data.apply(lambda row: create_combined_features(row, selected_features), axis=1)
combined_features= data['combined_features']

two_letter_list= []

# 'combined_features' sütununun satırlarını tek tek gez
for index, value in combined_features.items():
    # sütunları ikişer harflik bigramlara ayır
    value= value.strip() #boşlukları sil
    two_letter_bigrams = [(value[i:i+2]) for i in range(0, len(value)-1, 1)]
    two_letter_list.append(two_letter_bigrams) #listeye ekle (liste olarak)

#########################################################   
#print("value değeri============> \n" , two_letter_list[9998])

# Kullanıcıdan film başlığı girdisi al
title = input("Aradığınızı Girin: ").lower().strip()

# 'title' sütununu küçük harfe dönüştürerek ve boşlukları temizleyerek kontrol et
matching_row = str(data[data['title'].str.lower().str.strip() == title][selected_features])
matching_row=matching_row.strip()
seçilmis_bigrams = [(matching_row[i:i+2]) for i in range(0, len(matching_row)-1, 1)]

#############################################################
#print("seçilmiş bigrams=============> ", seçilmis_bigrams)
jac_degerleri= []
for index, film_bigrams in enumerate(two_letter_list):
    # Ortak bigramları bul
    common_bigrams = set(seçilmis_bigrams) & set(film_bigrams)
    
    #jaccard benzerlik değerini hesaplama
    toplam= (len(two_letter_list[index]))+ (len(seçilmis_bigrams))
    jaccard_degeri= len(common_bigrams)/ toplam
    #jaccard degerlerini bir listede toplama
    jac_degerleri.append(jaccard_degeri)

#en yüksek jaccard degerlerini veren fonksiyon
def en_buyuk_10_jaccard_degerleri(jac_degerleri):
    # Enumerate ile indeks ve değerleri bir arada al
    indeks_ve_degerler = list(enumerate(jac_degerleri))
    
    # Değerlere göre sırala (büyükten küçüğe)
    sirali_liste = sorted(indeks_ve_degerler, key=lambda x: x[1], reverse=True)
    
    # İlk 10 değeri ve sıralarını al
    en_buyuk_10 = sirali_liste[:10]
    
    # Sadece indeksleri ve değerleri döndür
    indeksler = [indeks for indeks, deger in en_buyuk_10]
    degerler = [deger for indeks, deger in en_buyuk_10]
    
    return indeksler, degerler

# Fonksiyonu kullanarak en büyük 10 değeri ve sıralarını al
en_buyuk_indeksler, en_buyuk_degerler = en_buyuk_10_jaccard_degerleri(jac_degerleri)

# Sonuçları yazdır
print("En büyük 10 Jaccard Değeri İndeksleri:", en_buyuk_indeksler)
print("En büyük 10 Jaccard Değerleri:", en_buyuk_degerler)

print("en benzer 10 veri: ")


# İlgili satır ve sütundaki veriye erişim
veri=data.iloc[int(en_buyuk_indeksler[0])]

print(veri.iloc[0])


