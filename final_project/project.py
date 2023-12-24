from pyspark.sql import *
from pyspark import SparkContext
from functools import reduce
from pyspark.ml.feature import StringIndexer, VectorAssembler, LinearRegression


import bz2
import os

## Functions ######################################

# Initialize SparkSession
def init_spark():
	sc = SparkContext("local", "ComercialFlights")
	spark = SparkSession.builder \
				.appName("First Session") \
				.master("local[*]") \
				.getOrCreate()

	sc.setLogLevel("ERROR")

# Load data into pyspark dataframe
def load_data():
    # read all csv files in the directory to pyspark dataframe
	csv_files = os.listdir("../BigData/data/project_data/")
	df_pyspark =[]
	for file in csv_files:
		file_path = "../BigData/data/project_data/" + file
		df = spark.read.csv(file_path, header=True, inferSchema=True)
		df_pyspark.append(df)
	
	# merge all dataframes into one
	df_pyspark = reduce(DataFrame.unionAll, df_pyspark)
	return df_pyspark

# Rename columns
def edit_column_names(df):
    df =  df.withColumnRenamed('DayofMonth','day_of_month').\
                withColumnRenamed('DayOfWeek','day_of_week').\
                withColumnRenamed('DepTime','actual_departure_time').\
                withColumnRenamed('CRSDepTime','scheduled_departure_time').\
                withColumnRenamed('ArrTime','actual_arrival_time').\
                withColumnRenamed('CRSArrTime','scheduled_arrival_time').\
                withColumnRenamed('UniqueCarrier','airline_code').\
                withColumnRenamed('FlightNum','flight_number').\
                withColumnRenamed('TailNum','plane_number').\
                withColumnRenamed('ActualElapsedTime','actual_flight_time').\
                withColumnRenamed('CRSElapsedTime','scheduled_flight_time').\
                withColumnRenamed('AirTime','air_time').\
                withColumnRenamed('ArrDelay','arrival_delay').\
                withColumnRenamed('DepDelay','departure_delay').\
                withColumnRenamed('TaxiIn','taxi_in').\
                withColumnRenamed('TaxiOut','taxi_out').\
                withColumnRenamed('CancellationCode','cancellation_code').\
                withColumnRenamed('CarrierDelay','carrier_delay').\
                withColumnRenamed('WeatherDelay','weather_delay').\
                withColumnRenamed('NASDelay','nas_delay').\
                withColumnRenamed('SecurityDelay','security_delay').\
                withColumnRenamed('LateAircraftDelay','late_aircraft_delay')
    for col in df.columns:
        df = df.withColumnRenamed(col, col.lower())
    return df

# select columns:
def my_columns (df):
    df = df.select('year','month','day_of_month', 'day_of_week', 'actual_departure_time',
                   'scheduled_departure_time', 'scheduled_arrival_time', 'airline_code',
                   'flight_number', 'plane_number', 'scheduled_flight_time', 'arrival_delay',
                   'departure_delay', 'origin', 'dest', 'distance', 'taxi_out', 'cancelled',
                   'cancellation_code')
    return df

# combine to create dates:
def add_date_column(df):
    df = df.withColumn('date', to_date(concat(col('day_of_month'), lit(' '),
                                              col('month'), lit(' '), col('year')), 'd M yyyy'))
    return df

# some strings to float:
def string_to_float(df):
    df = df.withColumn('arrival_delay', col('arrival_delay').cast('float'))
    df = df.withColumn('departure_delay', col('departure_delay').cast('float'))
    df = df.withColumn('taxi_out', col('taxi_out').cast('float'))
    df = df.withColumn('distance', col('distance').cast('float'))
    return df

# encode categorical features:
def encode_categorical_features(df):
    indexer = StringIndexer(inputCols=['airline_code', 'origin', 'dest', 'cancellation_code', 'plane_number'],
                            outputCols=['airline_index', 'origin_index', 'dest_index', 'cancellation_index', 'plane_index'])
    
    df = indexer.fit(df).transform(df)
    return df
    

# convert time to minutes:
def convert_time_to_minutes(df):
    # for actual_departure_time, scheduled_departure_time, scheduled_arrival_time, scheduled_flight_time transform to minutes
    # take the first two digits and multiply by 60 and add the last two digits
    df = df.withColumn('actual_departure_hour', (col('actual_departure_time') / 100).cast('int'))
    df = df.withColumn('scheduled_departure_hour', (col('scheduled_departure_time') / 100).cast('int'))
    df = df.withColumn('scheduled_arrival_hour', (col('scheduled_arrival_time') / 100).cast('int'))
    df = df.withColumn('scheduled_flight_hour', (col('scheduled_flight_time') / 100).cast('int'))
    
    df = df.withColumn('actual_departure_time_mins', (col('actual_departure_hour') * 60) + (col('actual_departure_time') % 100))
    df = df.withColumn('scheduled_departure_time_mins', (col('scheduled_departure_hour') * 60) + (col('scheduled_departure_time') % 100))
    df = df.withColumn('scheduled_arrival_time_mins', (col('scheduled_arrival_hour') * 60) + (col('scheduled_arrival_time') % 100))
    df = df.withColumn('scheduled_flight_time_mins', (col('scheduled_flight_hour') * 60) + (col('scheduled_flight_time') % 100))
    
    # drop actual_departure_hour, scheduled_departure_hour, scheduled_arrival_hour, scheduled_flight_hour
    
    df = df.drop('actual_departure_hour', 'scheduled_departure_hour', 'scheduled_arrival_hour', 'scheduled_flight_hour')
    
    return df

# handle missing values:
def handle_missing_values(df):
	# eliminar filas donde actual_departure_time es null
	df = df.filter(df.actual_departure_time.isNotNull())
	# eliminar filas donde scheduled_flight_time es null
	df = df.filter(df.scheduled_flight_time.isNotNull())

	return df

def my_df(df):
	# select columns
	df = df.select('date','year','month','day_of_month', 'day_of_week', 'actual_departure_time_mins',
					'scheduled_departure_time_mins', 'scheduled_arrival_time_mins', 'airline_index',
					'flight_number', 'scheduled_flight_time_mins', 
					'origin_index', 'dest_index', 'distance', 'taxi_out', 'cancelled', 'departure_delay')

	return df
    

# standarize df
def standarize_dataframe(df):
    temp = edit_column_names(df)
    temp = my_columns(temp)
    temp = string_to_float(temp)
    temp = add_date_column(temp)
    temp = encode_categorical_features(temp)
    temp = convert_time_to_minutes(temp)
    temp = handle_missing_values(temp)
    temp = my_df(temp)

    return temp

# vectorize features
def vectorize_features(df, features):
	featureassemble = VectorAssembler(inputCols=[features], outputCol='ind_features')
	df = featureassemble.transform(df)
	return df

#############################################

def main():
	# Initialize SparkSession
	init_spark()
	# load data into pyspark dataframe
	df_pyspark = load_data()
	
	print('Number of rows: ', df_pyspark.count())
	print('Number of columns: ', len(df_pyspark.columns))
	
	# Process data
	df = standarize_dataframe(df_pyspark)
 
	# Vectorize features
	# independent variable = arrival_delay (exclude it from features)
	vectorized = vectorize_features(df, df.columns[:-1]) ####ESTO ESTÁ MAL, NO ESTA QUITANDO LA ARRIVAL_DELAY --> REVISAR
	final_data = vectorized.select('ind_features', 'arrival_delay') # select variables we want to use
	
	# Split data into train and test
	train_data, test_data = final_data.randomSplit([0.7, 0.3])
	regressor = LinearRegression(featuresCol='ind_features', labelCol='arrival_delay')
	model = regressor.fit(train_data)
	print('Coefficients: ', model.coefficients)
	print('Intercept: ', model.intercept)
 
	# Evaluate model
	pred_results = model.evaluate(test_data)
	print('RMSE: ', pred_results.rootMeanSquaredError)
	print('MSE: ', pred_results.meanSquaredError)
	print('R2: ', pred_results.r2)
	print('MAE: ', pred_results.meanAbsoluteError)
 
	# 
 
	
 
    




if __name__ == "__main__":
	main()