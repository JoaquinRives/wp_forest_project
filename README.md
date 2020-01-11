
 #  Water permeability prediction project

### Packages:

- wp_knn_model: k nearest neighbor regression model and for prediction of the 
  water permeability level of forest soil. 

- ml_api: Flask app for deployment of the wp_knn_model as a REST API.


### Description:

This project is intended to be use in the forest industry to guide routing decisions 
in harvesting operations.

The soil water permeability 𝑥𝑤𝑝 can be used as an indicator for the bearing capacity 
of the soil, which is a crucially important factor in harvest operations with heavy machinery.

The predicted value of 𝑥𝑤𝑝 will be used to guide routing decisions in order to minimize
risks  for the forest harvester.

A route consists from a fixed number of spatially distributed points, for all of which 
we need a prediction on the 𝑥𝑤𝑝 level, in order to evaluate the route’s goodness.

The predictions are made using the k Nearest Neighbors Algorithm with Euclidean distance.

The model will be served as an online API allowing the harvesters to make on-site prediction 
requests from the harvester’s current geographical position.

Azure Pipelines is used for continuous integration and development of this project. 
Azure DevOps link ("https://dev.azure.com/joaquinrives01/wp_forest_project/_build?definitionId=7").


### Instructions to install and launch:
	
1- Clone or download the repository from "https://github.com/JoaquinRives/wp_forest_project"

2- Install the requirements
>pip install -r /wp_forest_project_public/requirements.txt

3- Start up the Flask server app 
>python /wp_forest_project_public/packages/ml_api/run.py



To install only the knn_model package from github:
>pip install git+git://github.com/JoaquinRives/wp_knn_model

To install only the knn_model package locally:
>pip install -e /wp_forest_project_public/packages/wp_knn_model


	

