# Communes-in-Denmark
Analysis of municipalities in Denmark using Foursquare API, geodata, and Statistikbanken.

The municipalities of Denmark are clustered using machine learning algorithm k-means clustering, based on Foursquare data.
The clusters are visualized using Folium.
Data from Statistikbanken is used to analyze the communes further and the results are visualized using Cloropleth.

See
1. [Capstone-Denmark-data.py](../Capstone-Denmark-data.py) for the code
2. [Analysis-of-communes-in-Denmark-presentation.pdf](../Analysis-of-communes-in-Denmark-presentation.pdf) for the presentation.

References used:
1.	Wikipedia page 
[https://da.wikipedia.org/wiki/Kommuner_i_Danmark_efter_indbyggertal](https://da.wikipedia.org/wiki/Kommuner_i_Danmark_efter_indbyggertal)
2.	Foursquare API
[https://developer.foursquare.com/](https://developer.foursquare.com/)
3.	Geocoder 
[https://geocoder.readthedocs.io/](https://geocoder.readthedocs.io/)
4.	JSON file
[https://github.com/Neogeografen/dagi/blob/master/geojson/kommuner.geojson](https://github.com/Neogeografen/dagi/blob/master/geojson/kommuner.geojson)
5.	Danmarks statistik
[https://www.dst.dk/da/Statistik/statistikbanken](https://www.dst.dk/da/Statistik/statistikbanken)
6.	Elbow method
[https://www.scikit-yb.org/en/latest/api/cluster/elbow.html](https://www.scikit-yb.org/en/latest/api/cluster/elbow.html)
