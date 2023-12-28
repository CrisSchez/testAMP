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

dfTransactions = spark.read.table("default.transactiondata").toPandas()
dfCustomers= spark.read.table("default.customers").toPandas()
dfProducts= spark.read.table("default.products").toPandas()

df=dfTransactions.merge(dfCustomers,on='customerid')
dftrain=df.merge(dfProducts,on='productid')
dftrain['orderdate']=pd.to_datetime(dftrain['orderdate'])
dftrain['shipdate']=pd.to_datetime(dftrain['shipdate'])

dftrain['ordermonth']=dftrain.orderdate.dt.month
dftrain['weekday']=dftrain.orderdate.dt.weekday
dftrain['day']=dftrain.orderdate.dt.day
daysO2S=pd.to_datetime(dftrain.shipdate)-pd.to_datetime(dftrain.orderdate)
dftrain['daysordertoship']=daysO2S.dt.days
dftraindays=dftrain.drop(['rowid','orderid','customerid','discount','name_x','country','gender','name_y','orderdate','shipdate'],axis=1)
