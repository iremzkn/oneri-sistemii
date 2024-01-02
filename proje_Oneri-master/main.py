import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# CSV dosyasından film verilerini yükle
secim=int(input("Hangi veri setini tercih edersiniz:\n 1-)Film \n 2-)Kitap \n 3-)Kıyafet"))
if secim==1:
    data = pd.read_csv('data.csv', encoding='ISO-8859-9')
if secim ==2:
    data = pd.read_csv('book_data.csv', encoding='ISO-8859-9')
if secim ==3:
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
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['combined_features'])

# Kosinüs benzerliği hesapla
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Kullanıcının girdiği film başlığına göre film önerilerini almak için fonksiyon
def get_movie_recommendations(title, top_n=10,):
    if secim==1:
        index = data[data['title'] == title].index[0]
        similarity_scores = list(enumerate(cosine_sim[index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        top_recommendations = similarity_scores[1:top_n + 1]
        data1='title'
        data2='genre'
        data3='director'
    elif secim==2:
        index = data[data['isim'] == title].index[0]
        similarity_scores = list(enumerate(cosine_sim[index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        top_recommendations = similarity_scores[1:top_n + 1]
        data1='isim'
        data2='tur'
        data3='yazar'
    elif  secim==3:
        index = data[data['numara'] == int(title)].index[0]
        similarity_scores = list(enumerate(cosine_sim[index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        top_recommendations = similarity_scores[1:top_n + 1]
        data1='isim'
        data2='firma'
        data3='numara'



    recommended =[[data[data1][top_data[0]],data[data2][top_data[0]],data[data3][top_data[0]]] for top_data in top_recommendations]
    return recommended, [score[1] for score in top_recommendations]

# Kullanıcıdan film başlığı girdisi al
title = input("Aradığınızı Girin: ")

# Kullanıcının girdisine göre film önerilerini al
recommendations, similarity_scores = get_movie_recommendations(title, top_n=10)

# Sonuçları çiz ve göster
plt.figure(figsize=(10, 6))
plt.barh(range(len(recommendations)), similarity_scores)

plt.xlabel('Kosinüs Benzerliği')
plt.ylabel('Film Başlığı')
plt.title(f"'{title}' için Kosinüs Benzerliğine Göre En İyi 10 Tavsiye Film")
'''y= [1,2,3,4,5,6,7,8,9,10]
x=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
for sonuc in recommendations:
    i=0
    plt.plot(x,y)
    plt.yticks(y[i], str(similarity_scores[i]))
    i=i+1'''
plt.tight_layout()
plt.show()

# Tavsiyeleri bir metin dosyasına yaz
output_file = 'output.txt'
with open(output_file, 'w', encoding='utf-8') as file:
    file.write(f"'{title}' için En İyi 10 Tavsiye Film:\n")
    i=0
    for sonuc in recommendations:

        file.write(str(sonuc[0]) + ',' + str(sonuc[1]) + ',' + str(sonuc[2]) + ',' + str(similarity_scores[i]) + ',' + '\n')
        i=i + 1
print(f"'{title}' için önerilen filmler '{output_file}' dosyasına yazıldı.")
