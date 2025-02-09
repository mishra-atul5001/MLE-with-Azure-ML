# Operationalizing Machine Learning

### Project performed using Udacity's VM
Challenges:
- VM times-out after every 4 hours
- Data never gets stored.
- Same steps are prepeated over-&-over-&-over everytime you start the VM
  
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
Registered Data and Respective AutoML Model
`I created 2 runs with different experimentation on Models, took the one that got completed quickly.`  

![](https://github.com/mishra-atul5001/MLE-with-Azure-ML/blob/main/registered-dataset.PNG)  
![](https://github.com/mishra-atul5001/MLE-with-Azure-ML/blob/main/atuoML-model-completed.PNG)  

`Best-Model & Application Insight is set to True while creating AutoML model in itself`  
![](https://github.com/mishra-atul5001/MLE-with-Azure-ML/blob/main/best-model-screenshot-1.PNG)  
![](https://github.com/mishra-atul5001/MLE-with-Azure-ML/blob/main/best-model-screenshot-2.PNG)  
![](https://github.com/mishra-atul5001/MLE-with-Azure-ML/blob/main/application-insight-true.PNG)  

`Logs & End-points results`  
Ran logs via Azure-ML-Notebook instead from console as VM console was not responding.  
![](https://github.com/mishra-atul5001/MLE-with-Azure-ML/blob/main/logs-py-screenshot.PNG)  
![](https://github.com/mishra-atul5001/MLE-with-Azure-ML/blob/main/endpoint-data.png)  
![](https://github.com/mishra-atul5001/MLE-with-Azure-ML/blob/main/endpoint-details.PNG)  
![](https://github.com/mishra-atul5001/MLE-with-Azure-ML/blob/main/endpoint-result.PNG)  

`Swagger UI and API documentation is up & running`  
![](https://github.com/mishra-atul5001/MLE-with-Azure-ML/blob/main/swaggerUI.png)  
![](https://github.com/mishra-atul5001/MLE-with-Azure-ML/blob/main/swagger-configs.PNG)  
![](https://github.com/mishra-atul5001/MLE-with-Azure-ML/blob/main/swagger-best-model.PNG)  

`Use Jupyter Notebook aml-pipelines-with-automated-machine-learning-step.ipynb to create the pipeline.`
![](https://github.com/mishra-atul5001/MLE-with-Azure-ML/blob/main/Screenshot%202025-02-09%20111528.png)  
![](https://github.com/mishra-atul5001/MLE-with-Azure-ML/blob/main/Screenshot%202025-02-09%20111555.png)  
The below Endpoint Status is from very-latest run, performed on Organizational Azure-ML Subscription
![](https://github.com/mishra-atul5001/MLE-with-Azure-ML/blob/main/Screenshot%202025-02-09%20132812.png)
![image](https://github.com/user-attachments/assets/12ed4baf-5e9c-4992-af2a-298e926c3f11)
![image](https://github.com/user-attachments/assets/6a1b2f30-bf9e-4581-961e-ec5e7dbcab2a)  

# Improvement Suggestions
The current accuracy is already very high 0.91, however, for further improvement we can try following:

- Perform deeper-EDA to identify misleading features.
- Remove class imbalance in the exisiting data. The dataset used is highly imbalanced, hence our model maybe biased to one class.
- Increase AutoML run duration. This would allow testing of a lot more models.
