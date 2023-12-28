!pip3 install numpy
!pip3 install pandas
!pip3 install scikit-learn
!pip3 install mlflow
!git+https://github.com/fastforwardlabs/cmlbootstrap#egg=cmlbootstrap
!pip3 install dill

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_squared_error

import mlflow
import mlflow.sklearn
import pickle
#from your_data_loader import load_data
from categoricalencoding import CategoricalEncoder
import datetime

import time
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import xml.etree.ElementTree as ET
from cmlbootstrap import CMLBootstrap
# Set the setup variables needed by CMLBootstrap
HOST = os.getenv("CDSW_API_URL").split(
    ":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split(
    "/")[6]  # args.username  # "vdibia"
API_KEY = os.getenv("CDSW_API_KEY") 
PROJECT_NAME = os.getenv("CDSW_PROJECT")  

import sys
import mlflow
import mlflow.sklearn
#mlflow.set_experiment('Retrain_exp')
experimentId=mlflow.get_experiment_by_name("expRetrain").experiment_id
dfExperiments=mlflow.search_runs(experiment_ids=experimentId)
maxmetric=dfExperiments["metrics.mse"].max()
runId=dfExperiments[dfExperiments["metrics.mse"]==maxmetric].head(1).run_id


# Instantiate API Wrapper
cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)
runtimes=cml.get_runtimes()
runtimes=runtimes['runtimes']
runtimesdf = pd.DataFrame.from_dict(runtimes, orient='columns')
runtimeid=runtimesdf.loc[(runtimesdf['editor'] == 'Workbench') & (runtimesdf['kernel'] == 'Python 3.9') & (runtimesdf['edition'] == 'Standard')]['id']
id_rt=runtimeid.values[0]


script_descriptor = open("dataprep_stock.py")
a_script = script_descriptor.read()
exec(a_script)

target='quantity'
cols = dftraindays.columns
num_cols = dftraindays._get_numeric_data().columns
objectCols=list(set(cols) - set(num_cols))
data=dftraindays.drop(target,axis=1)

for col in objectCols:
    data[col] = pd.Categorical(data[col])


ce = CategoricalEncoder()
X = ce.fit_transform(data)



y=dftraindays[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


a=mlflow.get_run(runId.item()).data.params
print(a)
n_estimators = int(a["n_estimators"])
rf = RandomForestRegressor(n_estimators=n_estimators)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
 
filename = './models/champion/ce.pkl'
pickle.dump(ce, open(filename, 'wb'))

filename = './models/champion/champion.pkl'
pickle.dump(rf, open(filename, 'wb'))

       
project_id = cml.get_project()['id']
params = {"projectId":project_id,"latestModelDeployment":True,"latestModelBuild":True}


default_engine_details = cml.get_default_engine({})
default_engine_image_id = default_engine_details["id"]


print("creando el modelo")
#data.loc[0].to_json()
example_model_input = {"shipmode":"Second Class","city":"Henderson","state":"Kentucky","postalcode":42420,"region":"South","productid":"FUR-BO-10001798","sales":261.96,"profit":41.9136,"segment":"Consumer","loyaltycard":0,"avgmonthlycharges":2607.945586274,"category":"Furniture","subcategory":"Bookcases","ordermonth":10,"weekday":0,"day":31}

try:


              # Create the YAML file for the model lineage
    yaml_text = \
        """"ModelOpsStocks":
      hive_table_qualified_names:                # this is a predefined key to link to training data
        - "default.transactiondata@cm"              
        - "default.customers@cm"
        - "default.products@cm"               # the qualifiedName of the hive_table object representing                

      metadata:                                  # this is a predefined key for additional metadata
        query: "select * from historical_data"   # suggested use case: query used to extract training data
        training_file: "get_champion_stock.py"       # suggested use case: training file used
    """

    with open('lineage.yml', 'w') as lineage:
        lineage.write(yaml_text)
    model_id = cml.get_models(params)[0]['id']
    latest_model = cml.get_model({"id": model_id, "latestModelDeployment": True, "latestModelBuild": True})

    build_model_params = {
      "modelId": latest_model['latestModelBuild']['modelId'],
      "projectId": latest_model['latestModelBuild']['projectId'],
      "targetFilePath": "model_serve_stock.py",
      "targetFunctionName": "predict",
      "engineImageId": default_engine_image_id,
      "kernel": "python3",
      "examples": latest_model['latestModelBuild']['examples'],
      "cpuMillicores": 1000,
      "memoryMb": 2048,
      "nvidiaGPUs": 0,
      "replicationPolicy": {"type": "fixed", "numReplicas": 1},
      "environment": {},"runtimeId":int(id_rt)}

    cml.rebuild_model(build_model_params)
    sys.argv=[]
    print('rebuilding...')

#            model_id = cml.get_models(params)[1]['id']
#            latest_model = cml.get_model({"id": model_id, "latestModelDeployment": True, "latestModelBuild": True})
#
#            build_model_params = {
#              "modelId": latest_model['latestModelBuild']['modelId'],
#              "projectId": latest_model['latestModelBuild']['projectId'],
#              "targetFilePath": "13_model_viz.py",
#              "targetFunctionName": "predict",
#              "engineImageId": default_engine_image_id,
#              "kernel": "python3",
#              "examples": latest_model['latestModelBuild']['examples'],
#              "cpuMillicores": 1000,
#              "memoryMb": 2048,
#              "nvidiaGPUs": 0,
#              "replicationPolicy": {"type": "fixed", "numReplicas": 1},
#              "environment": {},"runtimeId":int(id_rt)}
#
#            cml.rebuild_model(build_model_params)
#            
#            print('rebuilding...')
    # Wait for the model to deploy.


except:

              # Create the YAML file for the model lineage
    yaml_text = \
        """"ModelOpsStocks":
      hive_table_qualified_names:                # this is a predefined key to link to training data
        - "default.transactiondata@cm"              
        - "default.customers@cm"
        - "default.products@cm"               # the qualifiedName of the hive_table object representing                

      metadata:                                  # this is a predefined key for additional metadata
        query: "select * from historical_data"   # suggested use case: query used to extract training data
        training_file: "get_champion_stock.py"       # suggested use case: training file used
    """

    with open('lineage.yml', 'w') as lineage:
        lineage.write(yaml_text)

    create_model_params = {
        "projectId": project_id,
        "name": "ModelOpsStocks",
        "description": "Explain a given model prediction",
        "visibility": "private",
        "enableAuth": False,
        "targetFilePath": "model_serve_stock.py",
        "targetFunctionName": "predict",
        "engineImageId": default_engine_image_id,
        "kernel": "python3",
                "examples": [
                    {
                        "request": example_model_input,
                        "response": {}
                    }],
        "cpuMillicores": 1000,
        "memoryMb": 2048,
        "nvidiaGPUs": 0,
        "replicationPolicy": {"type": "fixed", "numReplicas": 1},
        "environment": {},"runtimeId":int(id_rt)}
    print("creando nuevo modelo")
    new_model_details = cml.create_model(create_model_params)
    access_key = new_model_details["accessKey"]  # todo check for bad response
    model_id = new_model_details["id"]

    print("New model created with access key", access_key)

    # Disable model_authentication
    cml.set_model_auth({"id": model_id, "enableAuth": False})
    sys.argv=[]

    # Wait for the model to deploy.
    is_deployed = False
    while is_deployed == False:
        model = cml.get_model({"id": str(
            new_model_details["id"]), "latestModelDeployment": True, "latestModelBuild": True})
        if model["latestModelDeployment"]["status"] == 'deployed':
            print("Model is deployed")
            break
        else:
            print("Deploying Model.....")
            time.sleep(10)

#            #example_input_viz={"data": {"colnames": ["StreamingTV", "MonthlyCharges", "PhoneService", "PaperlessBilling","Partner", "OnlineBackup", "gender", "Contract", "TotalCharges","StreamingMovies", "DeviceProtection", "PaymentMethod", "tenure","Dependents", "OnlineSecurity", "MultipleLines", "InternetService","SeniorCitizen", "TechSupport"],"coltypes": ["STRING","FLOAT","STRING","STRING","STRING","STRING","STRING","STRING","FLOAT","STRING","STRING","STRING", "INT","STRING","STRING","STRING","STRING","STRING","STRING"],"rows": [["No", "70.35", "No", "No", "No", "No", "Female", "Month-to-month","1397.475", "No", "No", "Bank transfer (automatic)", "29", "No","No", "No", "DSL", "No", "No"],["No", "70.35", "No", "No", "No", "No", "Female", "Month-to-month","1397.475", "No", "No", "Bank transfer (automatic)", "29", "No","No", "No", "DSL", "No", "No"]]}}        
#            example_input_viz= {"data": {"colnames": ["monthlycharges", "totalcharges","tenure","gender","dependents", "onlinesecurity", "multiplelines", "internetservice","seniorcitizen", "techsupport", "contract","streamingmovies", "deviceprotection", "paymentmethod","streamingtv","phoneservice", "paperlessbilling","partner", "onlinebackup"],"coltypes": ["FLOAT", "FLOAT","INT","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING"],"rows": [["70.35", "70.35","29","Male","No","No", "No", "DSL", "No", "No", "Month-to-month", "No", "No", "Bank transfer (automatic)","No",  "No", "No", "No", "No"],["70.35", "70.35","29","Female","No","No", "No", "DSL", "No", "No", "Month-to-month", "No", "No", "Bank transfer (automatic)","No", "No", "No", "No", "No"]]}}
#
#                    
#            create_model_params = {
#                "projectId": project_id,
#                "name": "ModelViz",
#                "description": "visualization a given model prediction",
#                "visibility": "private",
#                "enableAuth": False,
#                "targetFilePath": "13_model_viz.py",
#                "targetFunctionName": "predict",
#                "engineImageId": default_engine_image_id,
#                "kernel": "python3",
#                "examples": [
#                    {
#                        "request": example_input_viz,
#                        "response": {}
#                    }],
#                "cpuMillicores": 1000,
#                "memoryMb": 2048,
#                "nvidiaGPUs": 0,
#                "replicationPolicy": {"type": "fixed", "numReplicas": 1},
#                "environment": {},"runtimeId":int(id_rt)}
#            print("creando nuevo modelo de visualizacion")
#            new_model_details = cml.create_model(create_model_params)
#            access_key = new_model_details["accessKey"]  # todo check for bad response
#            model_id = new_model_details["id"]
#
#            print("New model created with access key", access_key)
#
#            # Disable model_authentication
#            cml.set_model_auth({"id": model_id, "enableAuth": False})
#            sys.argv=[]
#
#            # Wait for the model to deploy.
#            is_deployed = False
#            while is_deployed == False:
#                model = cml.get_model({"id": str(
#                    new_model_details["id"]), "latestModelDeployment": True, "latestModelBuild": True})
#                if model["latestModelDeployment"]["status"] == 'deployed':
#                    print("Model is deployed")
#                    break
#                else:
#                    print("Deploying Model.....")
#                    time.sleep(10)


