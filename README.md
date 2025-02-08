# Operationalizing Machine Learning

This is second of the three projects required for fulfillment of the Nanodegree Machine Learning Engineer with Microsoft Azure from Udacity. 
In this project, we create, publish and consume a Pipeline. We also explore ML model deployment as an HTTP REST API endpoint, swagger API documentation, Apache benchmarking of the deployed endpoint and consumption of the endpoint using JSON documents as an HTTP POST request.

The data used in this project is related with direct marketing campaigns (phone calls) of a banking institution. 
The classification goal is to predict if the client will subscribe a term deposit (variable y). It consists of 20 input variables (columns) and 32,950 rows with 3,692 positive classes and 29,258 negative classes.

Screencase Video hosted at: https://youtu.be/KQHQRZa5HCg

# Architecture
![image](https://github.com/user-attachments/assets/98977d48-637e-49ff-bc11-0df45549c16a)

Before begining with our experiment, 
- we have to register the dataset and configure a compute cluster that will be used for training. `Step 1 + 2`
- Automated ML experiment is used to find the best classification model (Steps 03,04,05). `Step 3 + 4 + 5`
- The best model is then deployed as an HTTP REST endpoint using Azure Container Instances while keeping authentication enabled. `Step 6 + 7`
- We enable application insights for our deployed model using script logs.py. `Step 8`
- To interact with the deployed model documentation, we use swagger. `Step 9`
- Model is then consumed using endpoint.py. An optional benchmarking is done for the deployed model using Apache benchmarking, benchmark.sh is used here. `Step 10 + 11`

# Key Steps


# Improvement Suggestions

