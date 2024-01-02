import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# CSV dosyasından film verilerini yükle
secim=int(input("Hangi veri setini tercih edersiniz:\n 1-)Film \n 2-)Kitap \n 3-)Kıyafet"))
if secim==1:
    data = pd.read_csv('data.csv', encoding='ISO-8859-9')
if secim ==2:
    d1ata = pd.read_csv('book_data.csv', encoding='ISO-8859-9')
if secim ==3:
    data = pd.read_csv('clouth_data1.csv', sep=';', encoding='ISO-8859-9')


print("benzerlik özellikleri:" , data.columns)
# Kullanıcının seçtiği özelliklere göre birleştirilmiş özellikleri oluşturan fonksiyon
def create_combined_features(row, selected_features):
    return ' '.join([str(row[feature]) for feature in selected_features])

# Benzerlik hesaplamada kullanılacak özellikleri kullanıcıdan al
selected_features = input("Benzerlik hesaplamak için özellikleri girin:").split(',')

# Kullanıcının seçtiği özelliklere göre 'combined_features' sütununu oluştur
data['combined_features'] = data.apply(lambda row: create_combined_features(row, selected_features), axis=1)

# Birleştirilmiş özellikleri vektörleştir
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['combined_features'])

# Dirsek yöntemi (Elbow method) kullanılarak optimal küme sayısını bulma
from sklearn.cluster import KMeans
wcss = []  # Within-Cluster Sum of Squares (WCSS) değerlerini depolamak için boş bir liste oluşturuluyor
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(tfidf_matrix)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Veri seti üzerinde K-Means modelini eğitme
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(tfidf_matrix)

# Kümeleme sonuçlarını görselleştirme
colors = ["red","blue","green","cyan","magenta"]
for i in range(5):
    plt.scatter(tfidf_matrix[y_kmeans == i, 0], tfidf_matrix[y_kmeans == i, 1], s = 100, c = colors[i], label = f'Cluster {i+1}')
    
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 150, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()