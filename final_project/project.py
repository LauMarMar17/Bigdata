from pyspark.sql import SparkSession
import bz2

spark = SparkSession.builder.appName("App").getOrCreate()
file_path = "../BigData/data/project_data//1987.csv.bz2"  # Path to the file

with bz2.open(file_path, "rb") as f:
	file_content = f.read()

with open("../BigData/data/project_data/1987.csv", "wb") as f:
	f.write(file_content)

csv_file_path = "../BigData/data/project_data/1987.csv"
df_pyspark = spark.read.csv(csv_file_path, header=True, inferSchema=True)

# Delete variables with arrival information
my_df = df_pyspark.select("Year", "Month", "DayofMonth", "DayOfWeek", "DepTime", "CRSDepTime", "CRSArrTime", "UniqueCarrier", "FlightNum", "TailNum", "CRSElapsedTime", "ArrDelay", "DepDelay", "Origin", "Dest", "Distance", "TaxiOut", "Cancelled", "CancellationCode")
