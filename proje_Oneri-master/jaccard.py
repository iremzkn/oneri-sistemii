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

# Birleştirilmiş özellikleri vektörleştir
vectorizer = CountVectorizer(binary=True)
tfidf_matrix = vectorizer.fit_transform(data['combined_features'])

# Jaccard benzerliği hesapla
jaccard_sim = pairwise_distances(tfidf_matrix.toarray(), metric='jaccard')


# Kullanıcının girdiği film başlığına göre film önerilerini almak için fonksiyon
def get_movie_recommendations(title, top_n=10):
    if secim == 1:
        index = data[data['title'] == title].index[0]
    elif secim == 2:
        index = data[data['isim'] == title].index[0]
    elif secim == 3:
        index = data[data['numara'] == int(title)].index[0]

    similarity_scores = list(enumerate(jaccard_sim[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=False) 
    top_recommendations = similarity_scores[1:top_n + 1]

    data1 = 'title' if secim == 1 else 'isim' if secim == 2 else 'isim'
    data2 = 'genre' if secim == 1 else 'tur' if secim == 2 else 'firma'
    data3 = 'director' if secim == 1 else 'yazar' if secim == 2 else 'numara'

    recommended = [[data[data1][top_data[0]], data[data2][top_data[0]], data[data3][top_data[0]]] for top_data in top_recommendations]
    return recommended, [score[1] for score in top_recommendations]

# Kullanıcıdan film başlığı girdisi al
title = input("Aradığınızı Girin: ")

# Kullanıcının girdisine göre film önerilerini al
recommendations, similarity_scores = get_movie_recommendations(title, top_n=10)

# Sonuçları çiz ve göster
plt.figure(figsize=(10, 6))
plt.barh(range(len(recommendations)), similarity_scores)
plt.xlabel('Jaccard Benzerliği')
plt.ylabel('Film Başlığı')
plt.title(f"'{title}' için Jaccard Benzerliğine Göre En İyi 10 Tavsiye Film")
plt.tight_layout()
plt.show()

# Tavsiyeleri bir metin dosyasına yaz
output_file = 'output.txt'
with open(output_file, 'w', encoding='utf-8') as file:
    file.write(f"'{title}' için En İyi 10 Tavsiye Film:\n")
    i = 0
    for sonuc in recommendations:
        file.write(str(sonuc[0]) + ',' + str(sonuc[1]) + ',' + str(sonuc[2]) + ',' + str(similarity_scores[i]) + ',' + '\n')
        i = i + 1
print(f"'{title}' için önerilen filmler '{output_file}' dosyasına yazıldı.")
