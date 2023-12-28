!pip install pandas
import pandas as pd
!pip install -r requirements.txt
#df=pd.read_csv('Superstore.csv',header=0,encoding = "latin1")
#
#dfProduct=df[['Product ID','Category','Sub-Category','Product Name']]
#
#unique_rows = dfProduct.drop_duplicates()
#dfProduct=unique_rows.groupby(['Product ID','Category','Sub-Category']).first().reset_index()
#
#dfProduct.to_csv('products.csv',index=False)
#
#dfCustomers=df[['Customer ID','Customer Name','Segment','Country']]
#unique_rows = dfCustomers.drop_duplicates()
#
#unique_rows.to_csv('customers.csv',index=False)
#
#dfTransactions=df.drop(['Category','Sub-Category','Product Name','Customer Name','Segment','Country'],axis=1)
#dfTransactions.to_csv('transactions.csv',index=False)
#
#import numpy as np
#dfCustomers['loyalCard'] = np.random.randint(0, 2, dfCustomers.shape[0])
#
#import random
#dfCustomers['avgMonthlyCharges'] = np.random.random(dfCustomers.shape[0])*3456
#dfCustomers.to_csv('customers.csv',index=False)
#
#dfTransactions['OrderDate']=pd.to_datetime(dfTransactions['Order Date'],format="%d-%m-%Y")+pd.DateOffset(years=8)+pd.DateOffset(months=11)
#dfTransactions['ShipDate']=pd.to_datetime(dfTransactions['Ship Date'],format="%d-%m-%Y")+pd.DateOffset(years=8)+pd.DateOffset(months=11)
#dfTransactions=dfTransactions.drop(['Order Date','Ship Date'],axis=1)
#
#dfTransactions.to_csv('transactions.csv',index=False)


import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *



spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .master("local[*]")\
    .config("spark.sql.extensions","org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")\
    .config("spark.sql.extensions","org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.spark_catalog","org.apache.iceberg.spark.SparkSessionCatalog") \
    .config("spark.sql.catalog.local","org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local.type","hadoop") \
    .config("spark.sql.catalog.spark_catalog.type","hive") \
    .getOrCreate()


script_descriptor = open("dataprep.py")
a_script = script_descriptor.read()

exec(a_script)

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

# Instantiate API Wrapper
cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)
runtimes=cml.get_runtimes()
runtimes=runtimes['runtimes']
runtimesdf = pd.DataFrame.from_dict(runtimes, orient='columns')
runtimeid=runtimesdf.loc[(runtimesdf['editor'] == 'Workbench') & (runtimesdf['kernel'] == 'Python 3.9') & (runtimesdf['edition'] == 'Standard')]['id']
id_rt=runtimeid.values[0]




target='daysordertoship'
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


try:
  experimentId=mlflow.get_experiment_by_name("expRetrain").experiment_id
  mlflow.delete_experiment(experimentId)

  time.sleep(20)
except:
  print("es la primera vez que se ejecuta")
      
      
run_time_suffix = datetime.datetime.now()
#run_time_suffix = run_time_suffix.strftime("%d%m%Y%H%M%S")
run_time_suffix = run_time_suffix.strftime("%M%S")
#mlflow.set_tracking_uri('http://your.mlflow.url:5000')
mlflow.set_experiment('expRetrain')
valuesParam=[9,11,15]
for i in range(len(valuesParam)):
  with mlflow.start_run(run_name="run_"+run_time_suffix+'_'+str(i)) as run: 

      #with mlflow.start_run() as run: 
          # tracking run parameters
      mlflow.log_param("compute", 'local')
      mlflow.log_param("dataset", 'telco-churn')
      mlflow.log_param("dataset_version", '2.0')
      mlflow.log_param("algo", 'random forest')

          # tracking any additional hyperparameters for reproducibility
      n_estimators = valuesParam[i]
      mlflow.log_param("n_estimators", n_estimators)

          # train the model
      rf = RandomForestRegressor(n_estimators=n_estimators)
      rf.fit(X_train, y_train)
      y_pred = rf.predict(X_test)

          # automatically save the model artifact to the S3 bucket for later deployment
      mlflow.sklearn.log_model(rf, "rf-baseline-model")

          # log model performance using any metric
      msqe=mean_squared_error(y_test.values.reshape(-1,1), y_pred.reshape(-1,1))
      #mse = mean_squared_error(y_test, y_pred)
      mlflow.log_metric("mse", msqe)

      mlflow.end_run()
