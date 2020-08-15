# -*- coding: utf-8 -*-
import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print('Libraries imported.')

import lxml

df=pd.read_html("https://da.wikipedia.org/wiki/Kommuner_i_Danmark_efter_indbyggertal")[0]

df['Km2'] = df['Km2'].astype('float')
df.drop(['1980', '1990', '2006', '2000', '2014', '2018'], axis=1, inplace=True)
df['Km2']=df['Km2']/100
df['2019']=df['2019']*1000
df['indb/km2']=df['indb/km2']*1000

df.head()
df.shape

from geopy.geocoders import Nominatim

from geopy.extra.rate_limiter import RateLimiter
geolocator = Nominatim(user_agent="dk_explorer")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
df['location'] = df['Navn'].apply(geocode)

df['point'] = df['location'].apply(lambda loc: tuple(loc.point) if loc else None)
df[['latitude', 'longitude', 'altitude']] = pd.DataFrame(df['point'].tolist(), index=df.index)

import json # library to handle JSON files

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

import folium # map rendering library

address = 'Denmark'

geolocator = Nominatim(user_agent="copenhagen_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinates of Denmark are {}, {}.'.format(latitude, longitude))

# create map of Denmark using latitude and longitude values
map_denmark = folium.Map(location=[latitude, longitude], zoom_start=7)

# add markers to map
for lat, lng, commune in zip(df['latitude'], df['longitude'], df['Navn']):
    label = '{}'.format(commune)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_denmark)  
    
#map_denmark
map_denmark.save("map_denmark.html")

address = 'Copenhagen'

geolocator = Nominatim(user_agent="copenhagen_explorer")
location = geolocator.geocode(address)
cph_latitude = location.latitude
cph_longitude = location.longitude

# create map of Denmark using latitude and longitude values
map_cph = folium.Map(location=[cph_latitude, cph_longitude], zoom_start=10)

# add markers to map
for lat, lng, commune in zip(df['latitude'], df['longitude'], df['Navn']):
    label = '{}'.format(commune)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_cph)  
    
map_cph.save("map_cph.html")

import os

CLIENT_ID = os.environ.get('CLIENT_ID') # your Foursquare ID
CLIENT_SECRET = os.environ.get('CLIENT_SECRET') # your Foursquare Secret
VERSION = os.environ.get('VERSION') # Foursquare API version

# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']
    
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)

LIMIT= 100

Denmark_venues = getNearbyVenues(names=df['Navn'],
                                   latitudes=df['latitude'],
                                   longitudes=df['longitude'],
                                  radius = 10000
                                )

Denmark_venues['Neighborhood'].value_counts()

Denmark_venues['Neighborhood'].value_counts().plot.barh(figsize=(20,20))
#plt.barh(Denmark_venues['Neighborhood'].value_counts(), width=100)
#plt.figure(figsize=(100,100))

dfcount=Denmark_venues.groupby('Neighborhood').count()

# Which communes to drop?
dftodrop=dfcount[dfcount['Venue'] < 10] 
dftodrop.reset_index(level=0, inplace=True)
print(dftodrop.shape)
dftodrop.head(16)

# drop the rows with too few venues: less than 10
cond = Denmark_venues['Neighborhood'].isin(dftodrop['Neighborhood'])
Denmark_venues.drop(Denmark_venues[cond].index, inplace = True)

print('There are {} unique categories.'.format(len(Denmark_venues['Venue Category'].unique())))

# Analysis of each commune
# one hot encoding
dk_onehot = pd.get_dummies(Denmark_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
dk_onehot['Neighborhood'] = Denmark_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [dk_onehot.columns[-1]] + list(dk_onehot.columns[:-1])
dk_onehot = dk_onehot[fixed_columns]

dk_onehot.head()

dk_grouped = dk_onehot.groupby('Neighborhood').mean().reset_index()
dk_grouped.head()

dk_grouped.drop('Neighborhood', 1).head()

#############
num_top_venues = 5

for hood in dk_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = dk_grouped[dk_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')
################
    
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]

num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
com_venues_sorted = pd.DataFrame(columns=columns)
com_venues_sorted['Neighborhood'] = dk_grouped['Neighborhood']

for ind in np.arange(dk_grouped.shape[0]):
    com_venues_sorted.iloc[ind, 1:] = return_most_common_venues(dk_grouped.iloc[ind, :], num_top_venues)

com_venues_sorted.head()

# 3. Clustering
from yellowbrick.cluster import KElbowVisualizer
# Instantiate the clustering model and visualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,25))

# set number of clusters
# was 5, then 4
kclusters = 5

dk_grouped_clustering = dk_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(dk_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 

visualizer.fit(dk_grouped_clustering)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure

# add clustering labels
com_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)
dk_cluster = df

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
dk_cluster = dk_cluster.join(com_venues_sorted.set_index('Neighborhood'), on='Navn')
dk_cluster.drop(['Nr.', 'location', 'point','2019', 'indb/km2','2019','altitude', 'Km2'], axis=1, inplace=True)
dk_cluster.head() # check the last columns!

dk_cluster.dropna(subset=["Cluster Labels"], axis=0, inplace=True)
dk_cluster.head()

dk_cluster['Cluster Labels'] = dk_cluster['Cluster Labels'].astype('int')

dk_cluster['1st Most Common Venue'].hist(by=dk_cluster['Cluster Labels'])
dk_cluster.rename(columns={"1st Most Common Venue": "case_status"}, inplace=True)

#dk_cluster['1st Most Common Venue'].barh(by=dk_cluster['Cluster Labels'])
dk_cluster.groupby('Cluster Labels').case_status.value_counts().unstack(0).plot.bar()

group=dk_cluster.groupby('Cluster Labels').case_status.value_counts().to_frame()
group.rename(columns={"case_status": "nr"}, inplace=True)
group.reset_index(inplace = True)

group.head()

group.groupby('case_status').count().plot.bar()

#dfl = dk_cluster.groupby(['Cluster Labels'])['case_status'].value_counts()
dfl = group.groupby(['Cluster Labels'])['case_status'].count()


dfl.plot.bar()
#group.groupby('case_status').count().plot.bar()

import matplotlib.pyplot as plt
# plot data
fig, ax = plt.subplots(figsize=(15,7))
# use unstack()
group.groupby(['Cluster Labels']).count().unstack().plot.bar(ax=ax)

# Number of unique
import matplotlib.pyplot as plt
dk_cluster.groupby('Cluster Labels')['case_status'].nunique().plot(kind='bar')
#dk_cluster.groupby('Cluster Labels')['case_status'].plot(kind='bar')
#plt.show()

plt.title('Number of unique 1st most common venues') # add a title to the histogram
#plt.ylabel('Number of Countries') # add y-label
plt.xlabel('Cluster label') # add x-label

import seaborn as sns

sns.catplot(x = "Cluster Labels",       # x variable name
            y = "nr",       # y variable name
            hue = "case_status",  # elements in each group variable name
            data = group,     # dataframe to plot
            kind = "bar")

group.groupby('Cluster Labels').plot(kind='barh',x='case_status',y='nr')

plt.title('Number of 1st most common venues in cluster') # add a title to the histogram
plt.ylabel('1st most common venue') # add y-label
#plt.xlabel('Cluster label') # add x-label
#plt.legend
plt.legend(frameon=False, title='Cluster')

dk_cluster['Cluster Labels'] = dk_cluster['Cluster Labels'].astype('int')

# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=7)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(dk_cluster['latitude'], dk_cluster['longitude'], dk_cluster['Navn'], dk_cluster['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
map_clusters.save("map_clusters.html")

################3
address = 'Copenhagen'

geolocator = Nominatim(user_agent="copenhagen_explorer")
location = geolocator.geocode(address)
cph_latitude = location.latitude
cph_longitude = location.longitude
#print('The geograpical coordinate are {}, {}.'.format(latitude, longitude))

# create map
map_clusters = folium.Map(location=[cph_latitude, cph_longitude], zoom_start=10)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(dk_cluster['latitude'], dk_cluster['longitude'], dk_cluster['Navn'], dk_cluster['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
map_clusters.save("map_clusters_cph.html")

dk_cluster.loc[dk_cluster['Cluster Labels'] == 0, dk_cluster.columns[[0] + list(range(5, dk_cluster.shape[1]))]]

################3
# Data from Statistikbanken

dfp = pd.read_csv("private-2.csv", header = None)
#dfp.columns = ['Parameter', 'KOMNAVN','Virksomheder']
dfp.columns = ['Parameter', 'Commune','PrivateCompanies']
dfp.drop(['Parameter'], axis=1, inplace=True)
dfp.head()

dfmus = pd.read_csv("museer.csv", header = None)
# dfmus.columns = ['Parameter1', 'Parameter2','Parameter3','KOMNAVN','Museer']
dfmus.columns = ['Parameter1', 'Parameter2','Parameter3','Commune','Museums']
dfmus.drop(['Parameter1','Parameter2','Parameter3'], axis=1, inplace=True)
dfmus.head()

df_indkomst = pd.read_csv("Inkomst.csv", header = None)
#df_indkomst.columns = ['Parameter1', 'Parameter2','Year','KOMNAVN','IndFamilier','IndPar','IndEnlige']
df_indkomst.columns = ['Parameter1', 'Parameter2','Year','Commune','IncomeFamilies','IncomeCouples','IncomeSingles']
df_indkomst.drop(['Parameter1','Parameter2','Year'], axis=1, inplace=True)
df_indkomst.head()

df_skils = pd.read_csv("skilsmisser2.csv", header = None)
#df_skils.columns = ['Parameter1', 'KOMNAVN','Skilsmisser']
df_skils.columns = ['Parameter1', 'Commune','Divorces']
df_skils.drop(['Parameter1'], axis=1, inplace=True)
df_skils.head()

df_lige = pd.read_csv("ligestilling2.csv", header = None, delimiter=';')
#df_lige.columns = ['Parameter1', 'Parameter2', 'Parameter3', 'KOMNAVN','Ligestilling']
df_lige.columns = ['Parameter1', 'Parameter2', 'Parameter3', 'Commune','FemaleWork']
df_lige.drop(['Parameter1','Parameter2','Parameter3'], axis=1, inplace=True)
df_lige.head()

from functools import reduce
df_merged=reduce(lambda x,y: pd.merge(x,y, on='Commune', how='outer'), [dfp, dfmus, df_indkomst, df_skils, df_lige])
df_merged.dropna(inplace = True)
df_merged.head()

df_merged.rename(columns={"Commune": "Navn"}, inplace=True)
df_merged2=reduce(lambda x,y: pd.merge(x,y, on='Navn', how='outer'), [df_merged, df])
df_merged2.shape
df_merged2.dropna(inplace = True)
df_merged2.head()

# dataframe df
#df_merged2.drop
df_merged2.drop(['location', 'point', 'altitude', 'Nr.'], axis=1, inplace=True)
df_merged2.head()
df_merged2['MuseumsNorm'] = df_merged2['Museums'] / df_merged2['2019']
df_merged2['DivorceNorm'] = df_merged2['Divorces'] / df_merged2['2019']
df_merged2['PrivComNorm'] = df_merged2['PrivateCompanies'] / df_merged2['2019']
df_merged2.head()

df_merged3 = df_merged2
df_merged3.drop(['Navn', 'IncomeCouples', 'IncomeSingles', 'Divorces', 'Km2', '2019', 'indb/km2', 'Region','latitude','longitude','MuseumsNorm','PrivComNorm'], axis=1, inplace=True)
df_merged3.head()

featureset = df_merged2[['IncomeFamilies',  'FemaleWork', 'Museums', 'PrivateCompanies', 'DivorceNorm']]
from sklearn.preprocessing import StandardScaler

#Clus_dataSet = np.nan_to_num(Clus_dataSet)
featureset = StandardScaler().fit_transform(featureset)
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(df_merged3)
scaled_df = pd.DataFrame(scaled_df, columns=[ 'PrivateCompanies', 'Museums', 'IncomeFamilies',  'FemaleWork', 'DivorceNorm'])
scaler = preprocessing.RobustScaler()
robust_scaled_df = scaler.fit_transform(df_merged3)
robust_scaled_df = pd.DataFrame(robust_scaled_df, columns=['PrivateCompanies', 'Museums', 'IncomeFamilies',  'FemaleWork', 'DivorceNorm'])
scaled_df.head()
robust_scaled_df.head()

# Cloropleth
dk_geo = r'kommuner2.json' # geojson file
# Remember the encoding
# https://github.com/Neogeografen/dagi/blob/master/geojson/kommuner.geojson

dk_map_p = folium.Map(location=[latitude, longitude], zoom_start=7, tiles='Mapbox Bright')

# create a numpy array of length 6 and has linear spacing from the minium total immigration to the maximum total immigration
threshold_scale = np.linspace(dfp['PrivateCompanies'].min(),
                              dfp['PrivateCompanies'].max(),
                              6, dtype=int)
threshold_scale = threshold_scale.tolist() # change the numpy array to a list
threshold_scale[-1] = threshold_scale[-1] + 1 # make sure that the last value of the list is greater than the maximum immigration

# generate choropleth map 
dk_map_p.choropleth(
    geo_data=dk_geo,
    data=dfp,
    columns=['Commune', 'PrivateCompanies'],
    key_on='feature.properties.KOMNAVN',
    threshold_scale=threshold_scale,
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Private companies in Denmark'
)

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(dk_cluster['latitude'], dk_cluster['longitude'], dk_cluster['Navn'], dk_cluster['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=2,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(dk_map_p)



# display map
dk_map_p
dk_map_p.save("dk_map_p.html")

dk_map_mus = folium.Map(location=[latitude, longitude], zoom_start=7, tiles='Mapbox Bright')

# create a numpy array of length 6 and has linear spacing from the minium total immigration to the maximum total immigration
threshold_scale = np.linspace(dfmus['Museums'].min(),
                              dfmus['Museums'].max(),
                              6, dtype=int)
threshold_scale = threshold_scale.tolist() # change the numpy array to a list
threshold_scale[-1] = threshold_scale[-1] + 1 # make sure that the last value of the list is greater than the maximum immigration

# generate choropleth map 
dk_map_mus.choropleth(
    geo_data=dk_geo,
    data=dfmus,
    columns=['Commune', 'Museums'],
    key_on='feature.properties.KOMNAVN',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Museums in Denmark'
)

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(dk_cluster['latitude'], dk_cluster['longitude'], dk_cluster['Navn'], dk_cluster['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=2,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(dk_map_mus)

# display map
dk_map_mus
dk_map_mus.save("dk_map_mus.html")

dk_map_ind = folium.Map(location=[latitude, longitude], zoom_start=7, tiles='Mapbox Bright')

# create a numpy array of length 6 and has linear spacing from the minium total immigration to the maximum total immigration
threshold_scale = np.linspace(df_indkomst['IncomeFamilies'].min(),
                              df_indkomst['IncomeFamilies'].max(),
                              6, dtype=int)
threshold_scale = threshold_scale.tolist() # change the numpy array to a list
threshold_scale[-1] = threshold_scale[-1] + 1 # make sure that the last value of the list is greater than the maximum immigration

# generate choropleth map 
dk_map_ind.choropleth(
    geo_data=dk_geo,
    data=df_indkomst,
    columns=['Commune', 'IncomeFamilies'],
    key_on='feature.properties.KOMNAVN',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Income, families, 2018'
)

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(dk_cluster['latitude'], dk_cluster['longitude'], dk_cluster['Navn'], dk_cluster['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=2,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(dk_map_ind)

# display map
dk_map_ind
dk_map_ind.save("dk_map_ind.html")

dk_map_skils = folium.Map(location=[latitude, longitude], zoom_start=7, tiles='Mapbox Bright')

# create a numpy array of length 6 and has linear spacing from the minium total immigration to the maximum total immigration
threshold_scale = np.linspace(df_skils['Divorces'].min(),
                              df_skils['Divorces'].max(),
                              6, dtype=int)
threshold_scale = threshold_scale.tolist() # change the numpy array to a list
threshold_scale[-1] = threshold_scale[-1] + 1 # make sure that the last value of the list is greater than the maximum immigration

# generate choropleth map 
dk_map_skils.choropleth(
    geo_data=dk_geo,
    data=df_skils,
    columns=['Commune', 'Divorces'],
    key_on='feature.properties.KOMNAVN',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Number of divorces, 2019'
)

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(dk_cluster['latitude'], dk_cluster['longitude'], dk_cluster['Navn'], dk_cluster['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=2,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(dk_map_skils)

# display map
dk_map_skils
dk_map_skils.save("dk_map_skils.html")

dk_map_lige = folium.Map(location=[latitude, longitude], zoom_start=7, tiles='Mapbox Bright')

# create a numpy array of length 6 and has linear spacing from the minium total immigration to the maximum total immigration
threshold_scale = np.linspace(df_lige['FemaleWork'].min(),
                              df_lige['FemaleWork'].max(),
                              6, dtype=int)
threshold_scale = threshold_scale.tolist() # change the numpy array to a list
threshold_scale[-1] = threshold_scale[-1] + 1 # make sure that the last value of the list is greater than the maximum immigration

# generate choropleth map 
dk_map_lige.choropleth(
    geo_data=dk_geo,
    data=df_lige,
    columns=['Commune', 'FemaleWork'],
    key_on='feature.properties.KOMNAVN',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Percentage of women that have a job'
)

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(dk_cluster['latitude'], dk_cluster['longitude'], dk_cluster['Navn'], dk_cluster['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=2,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(dk_map_lige)

# display map
dk_map_lige
dk_map_lige.save("dk_map_lige.html")

dk_map_lige = folium.Map(location=[cph_latitude, cph_longitude], zoom_start=7, tiles='Mapbox Bright')

# create a numpy array of length 6 and has linear spacing from the minium total immigration to the maximum total immigration
threshold_scale = np.linspace(df_lige['FemaleWork'].min(),
                              df_lige['FemaleWork'].max(),
                              6, dtype=int)
threshold_scale = threshold_scale.tolist() # change the numpy array to a list
threshold_scale[-1] = threshold_scale[-1] + 1 # make sure that the last value of the list is greater than the maximum immigration

# generate choropleth map 
dk_map_lige.choropleth(
    geo_data=dk_geo,
    data=df_lige,
    columns=['Commune', 'FemaleWork'],
    key_on='feature.properties.KOMNAVN',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Percentage of women that have a job'
)

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(dk_cluster['latitude'], dk_cluster['longitude'], dk_cluster['Navn'], dk_cluster['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=2,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(dk_map_lige)

# display map
dk_map_lige
dk_map_lige.save("dk_map_lige_cph.html")

## Regressions
# %%capture
# ! pip install seaborn
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
#%matplotlib inline

import seaborn as sns

sns.regplot(x="PrivateCompanies", y="Museums", data=robust_scaled_df)
#plt.ylim(0,)
sns.regplot(x="PrivateCompanies", y="IncomeFamilies", data=robust_scaled_df)
#plt.ylim(0,)
sns.regplot(x="DivorceNorm", y="FemaleWork", data=robust_scaled_df)
#plt.ylim(0,)