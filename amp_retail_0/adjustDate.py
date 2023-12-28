#daysO2S=pd.to_datetime(dfTransactions.shipdate)-pd.to_datetime(dfTransactions.orderdate)
#dfTransactions['daysordertoship']=daysO2S.dt.days
#dfTransactions.hist('daysordertoship', by='region')

dfTransactions2=dfTransactions
todayTrans=pd.DataFrame()
nDays=5

for i in range(nDays):
  days2today=pd.to_datetime('today')-pd.to_datetime(dfTransactions2['orderdate'])
  dfTransactions2['orderdate']=pd.to_datetime(dfTransactions2['orderdate'])+pd.DateOffset(days=(days2today).dt.days.min())
  dfTransactions2['shipdate']=pd.to_datetime(dfTransactions2['shipdate'])+pd.DateOffset(days=(days2today).dt.days.min())

  dfTransactions_new=dfTransactions2[dfTransactions2['orderdate']==pd.to_datetime(dfTransactions2['orderdate']).max()]
  dfTransactions2=dfTransactions2[dfTransactions2['orderdate']<pd.to_datetime(dfTransactions2['orderdate']).max()]
  
  todayTrans=pd.concat([todayTrans.reset_index(drop=True),dfTransactions_new.reset_index(drop=True)],axis=0).reset_index(drop=True)
  

todayTrans.to_csv('todayData.csv',index=False)

schemaTran = StructType(    [
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
  
todayTransSD=spark.createDataFrame(todayTrans,schema=schemaTran) 
todayTransSD.write.format("iceberg").mode("overwrite").saveAsTable('default.transactionsNew')

tranHist=spark.createDataFrame(dfTransactions2,schema=schemaTran) 
tranHist.write.format("iceberg").mode("overwrite").saveAsTable('default.transactiondata')
