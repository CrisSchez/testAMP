# # RFM Model
# 
# When it comes to finding out who your best customers are, 
# the old RFM matrix principle is the best. 
# RFM stands for Recency, Frequency and Monetary. 
# It is a customer segmentation technique that uses past 
# purchase behavior to divide customers into groups

# RFM Score Calculations 
# RECENCY (R): Days since last purchase
# FREQUENCY (F): Total number of purchases
# MONETARY VALUE (M): Total money this customer spent

import urllib

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.types import *
import datetime
import pandas as pd
import numpy as np

spark = SparkSession.builder.appName('Model_RFM').getOrCreate()

def rfm_score(r,f,m):
  score_value = "{0}{1}{2}".format(r,f,m)
  return score_value
def rfm_score_mod(r,f,m):
  rfm=1/(int(r)*int(f)*int(m))
  score_value = "{0}".format(rfm)
  return score_value

return_rfm_score_udf = F.udf(rfm_score_mod)

## Query historical purchase data

df = spark.read.table("default.transactiondata")
  
max_date = df.select(F.from_unixtime(F.unix_timestamp(F.max('orderdate'), 'YYYY-mm-dd'))).rdd.map(lambda x:(str(x[0]))).collect()
max_date = max_date[0][:10]

now = datetime.datetime.today()

df2 = df.withColumn('date', F.to_date(F.from_unixtime(F.unix_timestamp('orderdate', 'YYYY-mm-dd')))).sort("rowid", ascending=False)

df3 = df2.withColumn("amount_total",  F.col("sales"))

## Calculate last time each client purchase

recency_df = df3.groupBy('customerid').agg(F.max("date").alias("LastPurshaceDate"))
recency_df = recency_df.withColumn("date", F.lit(now))
recency_df = recency_df.withColumn("recency", F.datediff(F.to_date(recency_df["date"]),F.to_date(recency_df["LastPurshaceDate"]))).sort("recency")
recency_df = recency_df.select("customerid", "recency")


frequency_df = df3.groupBy('customerid').agg(F.count("*").alias("frequency"))

monetary_df = df3.groupBy('customerid').agg(F.sum(df3.sales).alias("monetary"))

rfm_table = recency_df.join(frequency_df, "customerid").sort("customerid")
rfm_table = rfm_table.join(monetary_df, "customerid").sort("customerid")

quantile = rfm_table.approxQuantile(["recency", "frequency", "monetary"], [0.25, 0.5, 0.75], 0)

## Calculate Recency

rfm_table2 = rfm_table.withColumn('r_quantile', F.when((F.col("recency") >= 0) & (F.col("recency") <= quantile[0][0]), "4")\
 .when((F.col("recency") > quantile[0][0]) & (F.col("recency") <= quantile[0][1]), "3")\
 .when((F.col("recency") > quantile[0][1]) & (F.col("recency") <= quantile[0][2]), "2").otherwise("1"))

## Calculate Frequency

rfm_table2 = rfm_table2.withColumn('f_quantile', F.when((F.col("frequency") >= 0) & (F.col("frequency") <= quantile[1][0]), "1")\
 .when((F.col("frequency") > quantile[1][0]) & (F.col("frequency") <= quantile[1][1]), "2")\
 .when((F.col("frequency") > quantile[1][1]) & (F.col("frequency") <= quantile[1][2]), "3").otherwise("4"))

## Calculate Monetary

rfm_table3 = rfm_table2.withColumn('m_quantile', F.when((F.col("monetary") >= 0) & (F.col("monetary") <= quantile[2][0]), "1")\
 .when((F.col("monetary") > quantile[2][0]) & (F.col("monetary") <= quantile[2][1]), "2")\
 .when((F.col("monetary") > quantile[2][1]) & (F.col("monetary") <= quantile[2][2]), "3").otherwise("4"))


## Adding RFMScore column

rfm_table4 = rfm_table3.withColumn("RFMScore", return_rfm_score_udf(F.col("r_quantile"),F.col("f_quantile"),F.col("m_quantile")))


rfm_table4.sort("RFMScore", ascending=True).toPandas()
