from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext
from functools import reduce
from pyspark.ml.feature import Normalizer, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import click 

import bz2
import os

## Functions ######################################

# Load data into pyspark dataframe
def load_data(path):
    # Initialize SparkSession
    sc = SparkContext("local", "ComercialFlights")
    spark = SparkSession.builder \
                .appName("First Session") \
                .master("local[*]") \
                .getOrCreate()

    sc.setLogLevel("ERROR")

    # read all csv files in the directory to pyspark dataframe
    csv_files = os.listdir(path)
    df_pyspark =[]
    for file in csv_files:
        file_path = path + file
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

# String values to float:
def string_to_float(df):
    df = df.withColumn('arrival_delay', col('arrival_delay').cast('float'))
    df = df.withColumn('departure_delay', col('departure_delay').cast('float'))
    df = df.withColumn('taxi_out', col('taxi_out').cast('float'))
    df = df.withColumn('distance', col('distance').cast('float'))
    return df

# Encode categorical features:
def encode_categorical_features(df):
    indexer = StringIndexer(inputCols=['airline_code', 'origin', 'dest', 'cancellation_code', 'plane_number'],
                            outputCols=['airline_index', 'origin_index', 'dest_index', 'cancellation_index', 'plane_index'])
    
    df = indexer.fit(df).transform(df)
    return df

# Convert time to minutes:
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

# Keep only relevant columns:
def my_df(df):
    # select columns
    df = df.select('year', 'month', 'day_of_month', 'day_of_week', 'actual_departure_time_mins',
 					'scheduled_departure_time_mins', 'scheduled_arrival_time_mins', 'airline_index',
 					'flight_number', 'scheduled_flight_time_mins', 'departure_delay',
 					'origin_index', 'dest_index', 'distance', 'cancelled',
 					'arrival_delay')
    return df

# Handle null values:
def drop_nulls(df):
	# remove rows in arrival_delay where arrival_delay is null
	df = df.filter(df.arrival_delay.isNotNull())
	# remove rows in scheduled_flight_time_mins where departure_delay is null
	df = df.filter(df.scheduled_flight_time_mins.isNotNull())
	# remove rows in distance where distance is null
	df = df.filter(df.distance.isNotNull())
	return df

# Drop cancelled flights:
def drop_cancelled(df):
    df = df.filter(df.cancelled == 0)
    return df


# standarize df
def standarize_dataframe(df):
    temp = edit_column_names(df)
    temp = string_to_float(temp)
    temp = encode_categorical_features(temp)
    temp = convert_time_to_minutes(temp)
    temp = my_df(temp)
    temp = drop_nulls(temp)
    temp = drop_cancelled(temp)

    return temp

# Create model
def create_model(df, my_features):
    # Train-test split
    train, test = df.randomSplit([0.7, 0.3], seed=42)
    # Vectorize features
    featureassmebler = VectorAssembler(inputCols=my_features, outputCol='features')
	# Normlizer
    normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)
    # Linear Regression
    lr = LinearRegression(featuresCol='features_norm', labelCol='arrival_delay', maxIter=10, regParam=0.3, elasticNetParam=0.8)
    # Pipeline
    pipeline = Pipeline(stages=[featureassmebler, normalizer, lr])
    # Fit pipeline
    model = pipeline.fit(train)
    
    return train, test, model

# Validate model
def validate_model(model, test_data):
    predictions = model.transform(test_data)
    evaluator = RegressionEvaluator(labelCol="arrival_delay", predictionCol="prediction", metricName="mae")
    mae = evaluator.evaluate(predictions)
    print("MAE = %g" % mae)

    evaluator = RegressionEvaluator(labelCol="arrival_delay", predictionCol="prediction", metricName="mse")
    mse = evaluator.evaluate(predictions)
    print("MSE = %g" % mse)

    evaluator = RegressionEvaluator(labelCol="arrival_delay", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("RMSE = %g" % rmse)

    evaluator = RegressionEvaluator(labelCol="arrival_delay", predictionCol="prediction", metricName="r2")
    r2 = evaluator.evaluate(predictions)
    print("R2 = %g" % r2)
    
#############################################

@click.command()
@click.option('--data_path', 
              default='BigData/data/project_data/', 
              help='Path to the data folder')

def main(data_path):
    click.echo('Loading data from {}'.format(data_path))
    
    # Load data into pyspark dataframe
    df_pyspark = load_data(data_path)

    print('Initial number of rows: ', df_pyspark.count())
    print('initial number of columns: ', len(df_pyspark.columns))

    # Process data
    df = standarize_dataframe(df_pyspark)
    print('Final number of rows: ', df.count())
    print('Final number of columns: ', len(df.columns))

    # Create model
    my_features = ['year', 'month', 'day_of_month', 'day_of_week', 'actual_departure_time_mins',
                    'scheduled_departure_time_mins', 'scheduled_arrival_time_mins', 'airline_index',
                    'flight_number', 'scheduled_flight_time_mins', 'departure_delay',
                    'origin_index', 'dest_index', 'distance']

    print('Number of features: ', len(my_features))
    train, test, model = create_model(df, my_features)

    # Evaluate model
    validate_model(model, test)

if __name__ == "__main__":
	main()