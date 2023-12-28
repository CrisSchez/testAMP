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

dfProducts=pd.read_csv('products.csv',header=0)

dfProducts.columns=['productid','category','subcategory','name']
schemaProd = StructType(
    [
        StructField("productid", StringType(), True),
        StructField("category", StringType(), True),
        StructField("subcategory", StringType(), True),
        StructField("name", StringType(), True)
    ]
)

prod_data = spark.read.csv(
    "products.csv",
    header=True,
    schema=schemaProd,
    sep=',',
    nullValue='NA'
)


dfCustomers=pd.read_csv('customers.csv',header=0)
dfCustomers.columns=['customerid','name','segment','country','loyaltycard','avgmonthlycharges','gender']
schemaCust = StructType(
    [
        StructField("customerid", StringType(), True),
        StructField("name", StringType(), True),
        StructField("segment", StringType(), True),
        StructField("country", StringType(), True),
        StructField("loyaltycard", IntegerType(), True),
        StructField("avgmonthlycharges", DoubleType(), True),
        StructField("gender", StringType(), True)
    ]
)

customer_data = spark.read.csv(
    "customers.csv",
    header=True,
    schema=schemaCust,
    sep=',',
    nullValue='NA'
)

dfTransactions=pd.read_csv('transactions.csv',header=0)
dfTransactions.columns=['rowid', 'orderid', 'shipmode', 'customerid', 'city', 'state',
       'postalcode', 'region', 'productid', 'sales', 'quantity', 'discount',
       'profit', 'orderdate', 'shipdate']
schemaTran = StructType(
    [
        StructField("rowid", IntegerType(), True),
        StructField("orderid", StringType(), True),
        StructField("shipmode", StringType(), True),
        StructField("customerid", StringType(), True),
        StructField("city", StringType(), True),
        StructField("state", StringType(), True),
        StructField("postalcode", StringType(), True),
        StructField("region", StringType(), True),
        StructField("productid", StringType(), True),
        StructField("sales", DoubleType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("discount", DoubleType(), True),
        StructField("profit", DoubleType(), True),
        StructField("orderdate", DateType(), True),
        StructField("shipdate", DateType(), True),

    ]
)
transaction_data = spark.read.csv(
    "transactions.csv",
    header=True,
    schema=schemaTran,
    sep=',',
    nullValue='NA'
)


transaction_data.write.format("iceberg").mode("overwrite").saveAsTable('default.transactions')
customer_data.write.format("iceberg").mode("overwrite").saveAsTable('default.customers')
prod_data.write.format("iceberg").mode("overwrite").saveAsTable('default.products')

spark.sql("show tables in default").show()