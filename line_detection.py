import pickle
import math
from matplotlib import pyplot as plt
import pandas as  pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import matplotlib
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.cluster import AgglomerativeClustering, OPTICS
from sklearn.cluster import DBSCAN, Birch, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from kneed import DataGenerator, KneeLocator
from sklearn.preprocessing import StandardScaler, normalize
from matplotlib import pyplot
import time
from sklearn import metrics

a_file = open("data.pkl", "rb")
car_dic = pickle.load(a_file)
angle = []

num_clusters = 4
dist = 4

eps_DBSCAN = 0.02
eps_OPTICS = 0.04
Spectral = True

'''
features : 'all', 'angle', 'location'

'''

features = 'all' 

x = []
y = []

x_prime = []
y_prime = []

for i in range(0,900):
	for n in car_dic[i+1]:

		x.append(car_dic[i+1][n]['x'])
		y.append(car_dic[i+1][n]['y'])
		if i > dist :
			for k in car_dic[i-dist]:
				if k == n :


					if (car_dic[i+1][k]['x'] - car_dic[i-dist][k]['x'])!=0 and car_dic[i+1][k]['y'] - car_dic[i-dist][k]['y'] !=0:
						angle_raw = np.arctan2((car_dic[i+1][k]['y'] - car_dic[i-dist][k]['y']), (car_dic[i+1][k]['x'] - car_dic[i-dist][k]['x']))*180/np.pi

						if angle_raw<0:
							angle_raw = angle_raw + 360
						angle.append(angle_raw)
						x_prime.append(car_dic[i+1][k]['x'])
						y_prime.append(car_dic[i+1][k]['y'])





print(f'Total number of detected cars : {len(x)}')
print(f'Total number of cars with angle : {len(x_prime)}')
print(f'Number of clusters : {num_clusters}')


plt.figure(figsize=(10,10))
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
plt.scatter(x,y,color='gray', s=2)
plt.title('Allll cars',fontsize=10)
plt.ylim(max(plt.ylim()), min(plt.ylim()))
# plt.show()



data = {'angle' : angle, 'x' : x_prime, 'y' : y_prime}
df = pd.DataFrame(data)

scaler = StandardScaler().fit(df)
scaler_loc = StandardScaler().fit(df[['x', 'y']])
scaler_angle = StandardScaler().fit(np.array(y_prime).reshape(-1,1))

with open('scaler_loc.pkl', 'wb') as f:
    pickle.dump(scaler_loc, f)

scaled_data = scaler.transform(df)

df = pd.DataFrame(scaled_data)
df = df.rename(columns={0 : 'angle', 1 : 'x', 2 : 'y'})


colors = ['purple','red', 'gray', 'blue', 'pink', 'green', 'yellow']

k_means = KMeans(n_clusters=num_clusters,random_state=42)
Agglomerative = AgglomerativeClustering(n_clusters=num_clusters,affinity='euclidean')
Birch = Birch(n_clusters=num_clusters)
BatchKMeans = MiniBatchKMeans(n_clusters=num_clusters)
SpectralClustering = SpectralClustering(n_clusters=num_clusters)

if features == 'angle':

	start = time.time()
	k_means.fit(df[['angle']])
	end = time.time()
	print(f'KMeans calculation time: {end - start}')

	start = time.time()
	Agglomerative.fit(df[['angle']])
	end = time.time()
	print(f'Agglomerative calculation time: {end - start}')



	start = time.time()
	Birch.fit(df[['angle']])
	end = time.time()
	print(f'Birch calculation time: {end - start}')

	start = time.time()
	BatchKMeans.fit(df[['angle']])
	end = time.time()
	print(f'BatchKMeans calculation time: {end - start}')



	try:

		df_spectral = df.iloc[0:len(x_prime)//2, :]
		start = time.time()
		SpectralClustering.fit(df_spectral[['angle']])
		end = time.time()
		print(f'Spectral calculation time: {end - start}')

	except MemoryError:
		Spectral = False
		print('Cannot calculate Spectral Clustering because of lack of memory.')
		pass

if features == 'location':

	start = time.time()	
	k_means.fit(df[['x', 'y']])
	end = time.time()
	print(f'KMeans calculation time: {end - start}')

	start = time.time()
	Agglomerative.fit(df[['x', 'y']])
	end = time.time()
	print(f'Agglomerative calculation time: {end - start}')



	start = time.time()
	Birch.fit(df[['x', 'y']])
	end = time.time()
	print(f'Birch calculation time: {end - start}')

	start = time.time()
	BatchKMeans.fit(df[['x', 'y']])
	end = time.time()
	print(f'BatchKMeans calculation time: {end - start}')


	try:

		df_spectral = df.iloc[0:len(x_prime)//2, :]
		start = time.time()
		SpectralClustering.fit(df_spectral[['x', 'y']])
		end = time.time()
		print(f'Spectral calculation time: {end - start}')

	except MemoryError:

		Spectral = False
		print('Cannot calculate Spectral Clustering because of lack of memory.')
		pass

if features == 'all':

	start = time.time()
	k_means.fit(df[['angle', 'x', 'y']])
	end = time.time()
	print(f'KMeans calculation time: {end - start}')

	start = time.time()
	Agglomerative.fit(df[['angle', 'x', 'y']])
	end = time.time()
	print(f'Agglomerative calculation time: {end - start}')



	start = time.time()
	Birch.fit(df[['angle', 'x', 'y']])
	end = time.time()
	print(f'‌Birch calculation time: {end - start}')

	start = time.time()
	BatchKMeans.fit(df[['angle', 'x', 'y']])
	end = time.time()
	print(f'BatchKMeans calculation time: {end - start}')



	try:
		df_spectral = df.iloc[0:len(x_prime)//2, :]
		start = time.time()
		SpectralClustering.fit(df_spectral[['angle', 'x', 'y']])
		end = time.time()
		print(f'Spectral calculation time: {end - start}')

	except MemoryError:

		Spectral = False
		print('Cannot calculate Spectral Clustering because of lack of memory.')
		pass

df['KMeans_labels']=k_means.labels_
KMeans_metrics = metrics.silhouette_score(df[['angle','x', 'y']], df['KMeans_labels'], metric='euclidean')
df['HR_labels']=Agglomerative.labels_
HR_metrics = metrics.silhouette_score(df[['angle','x', 'y']], df['HR_labels'], metric='euclidean')

df['Birch_labels']=Birch.labels_ 
Birch_metrics = metrics.silhouette_score(df[['angle','x', 'y']], df['Birch_labels'], metric='euclidean')
df['MiniBatchKMeans_labels']=BatchKMeans.labels_ 
MiniBatchKMeans_metrics = metrics.silhouette_score(df[['angle','x', 'y']], df['MiniBatchKMeans_labels'], metric='euclidean')

if Spectral:
	df_spectral['Spectral_labels']=SpectralClustering.labels_ 
	Spectral_metrics = metrics.silhouette_score(df_spectral[['angle','x', 'y']], df_spectral['Spectral_labels'], metric='euclidean')

df.to_csv(f'{num_clusters}_Clusters_Cars.csv')
df_spectral.to_csv(f'{num_clusters}_Spectral_Clusters_Cars.csv')



data_metrics = {'KMeans': KMeans_metrics,
				'Agglomerative': HR_metrics,

				'Birch': Birch_metrics,
				'MiniBatchKMeans': MiniBatchKMeans_metrics,

				'Spectral': Spectral_metrics}

df_metrics = pd.DataFrame(data_metrics, index =['Euclidean'])
df_metrics.to_csv(f'{num_clusters}_metrics.csv')
print(df_metrics)
# Plotting KMeans clusters
plt.figure(figsize=(10,10))
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
plt.scatter(df['x'],df['y'],c=df['KMeans_labels'],cmap=matplotlib.colors.ListedColormap(colors),s=3)
plt.title('K-Means Clustering',fontsize=10)
plt.xlabel('X axis',fontsize=10)
plt.ylabel('Y axis',fontsize=10)
plt.ylim(max(plt.ylim()), min(plt.ylim()))
# plt.show()










# Plotting Hierarchical clusters
plt.figure(figsize=(10,10))
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
plt.scatter(df['x'],df['y'],c=df['HR_labels'],cmap=matplotlib.colors.ListedColormap(colors),s=3)
plt.title('Hierarchical Clustering',fontsize=10)
plt.xlabel('X axis',fontsize=10)
plt.ylabel('Y axis',fontsize=10)
plt.ylim(max(plt.ylim()), min(plt.ylim()))
# plt.show()














# Plotting Birch clusters
plt.figure(figsize=(10,10))
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
plt.scatter(df['x'],df['y'],c=df['Birch_labels'],cmap=matplotlib.colors.ListedColormap(colors),s=3)
plt.title('Birch Clustering',fontsize=10)
plt.xlabel('X axis',fontsize=10)
plt.ylabel('Y axis',fontsize=10)
plt.ylim(max(plt.ylim()), min(plt.ylim()))
# plt.show()







# Plotting MiniBatchKMeans clusters
plt.figure(figsize=(10,10))
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
plt.scatter(df['x'],df['y'],c=df['MiniBatchKMeans_labels'],cmap=matplotlib.colors.ListedColormap(colors),s=3)
plt.title('MiniBatchKMeans Clustering',fontsize=10)
plt.xlabel('X axis',fontsize=10)
plt.ylabel('Y axis',fontsize=10)
plt.ylim(max(plt.ylim()), min(plt.ylim()))
# plt.show()











if Spectral:
	#Plotting Spectral clusters
	plt.figure(figsize=(10,10))
	plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
	plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
	plt.scatter(df_spectral['x'],df_spectral['y'],c=df_spectral['Spectral_labels'],cmap=matplotlib.colors.ListedColormap(colors),s=3)
	plt.title('Spectral Clustering',fontsize=10)
	plt.xlabel('X axis',fontsize=10)
	plt.ylabel('Y axis',fontsize=10)
	plt.ylim(max(plt.ylim()), min(plt.ylim()))
	plt.show()
else:
	plt.show()