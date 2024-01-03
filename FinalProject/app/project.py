"""
This script loads the data from the data folder, processes it and creates a linear regression model using pyspark.
The model is evaluated using MAE, MSE, RMSE and R2.

Usage:
    python project.py -d <path_to_data_folder>          | default: app/data

Requirements:
    specified in requirements.txt
"""
import os
from functools import reduce

import click
from pyspark.conf import SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import Normalizer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import *
from pyspark.sql.functions import *

## Functions ######################################


# Load data into pyspark dataframe
def load_data(path):
    # Initialize SparkSession
    conf = (
        SparkConf()
        .set("spark.driver.memory", "4g")
        .set("spark.executor.memory", "4g")
        .set("loglevel", "ERROR")
    )
    spark = (
        SparkSession.builder.appName("ComercialFlights")
        .master("local[*]")
        .config(conf=conf)
        .getOrCreate()
    )

    # If the dataframe is already in the local path, load it
    if os.path.exists(path + "/my_df"):
        # try to read it, but it is empty execute the else statement
        try:
            print("###\nLoading data from local path: {}".format(path + "/my_df\n"))
            df = spark.read.csv(path + "/my_df", header=True, sep=",")
            return df
        except:
            print(
                "###\nLocal path exists but the file is empty. Loading data from bz2 files\n"
            )
            pass

    else:
        # Handle FileNotFoundError: if this happens ask the user to introduce another path and try again
        try:
            # Read data
            files = os.listdir(path)
            files = [
                file for file in files if file.endswith(".bz2")
            ]  # filter bz2 files as this is how we download the data
            dfs = []
            for file in files:
                # read them into df
                df = spark.read.csv(path + "/" + files[0], header=True, sep=",")
                dfs.append(df)

            # Merge all dataframes into one
            one_df = reduce(DataFrame.unionAll, dfs)
            print("###\nLoading csv files from {}\n".format(path))

            # Place the dataframe in a local path
            one_df.write.csv(path + "/my_df", header=True)
            return one_df

        except:
            print("###\nFile not found. Please introduce another path\n")
            return False


# Rename columns
def edit_column_names(df):
    """
    Edit column names to lowercase and replace spaces with underscores.

    Param:
    - df: spark dataframe

    Return:
    - df: spark dataframe with edited column names
    """
    df = (
        df.withColumnRenamed("DayofMonth", "day_of_month")
        .withColumnRenamed("DayOfWeek", "day_of_week")
        .withColumnRenamed("DepTime", "actual_departure_time")
        .withColumnRenamed("CRSDepTime", "scheduled_departure_time")
        .withColumnRenamed("ArrTime", "actual_arrival_time")
        .withColumnRenamed("CRSArrTime", "scheduled_arrival_time")
        .withColumnRenamed("UniqueCarrier", "airline_code")
        .withColumnRenamed("FlightNum", "flight_number")
        .withColumnRenamed("TailNum", "plane_number")
        .withColumnRenamed("ActualElapsedTime", "actual_flight_time")
        .withColumnRenamed("CRSElapsedTime", "scheduled_flight_time")
        .withColumnRenamed("AirTime", "air_time")
        .withColumnRenamed("ArrDelay", "arrival_delay")
        .withColumnRenamed("DepDelay", "departure_delay")
        .withColumnRenamed("TaxiIn", "taxi_in")
        .withColumnRenamed("TaxiOut", "taxi_out")
        .withColumnRenamed("CancellationCode", "cancellation_code")
        .withColumnRenamed("CarrierDelay", "carrier_delay")
        .withColumnRenamed("WeatherDelay", "weather_delay")
        .withColumnRenamed("NASDelay", "nas_delay")
        .withColumnRenamed("SecurityDelay", "security_delay")
        .withColumnRenamed("LateAircraftDelay", "late_aircraft_delay")
    )
    for col in df.columns:
        df = df.withColumnRenamed(col, col.lower())
    return df


# String values to float:
def string_to_float(df):
    """
    Convert some columns from string to float.

    Param:
    - df: spark dataframe

    Return:
    - df: spark dataframe with some columns converted to float
    """
    df = df.withColumn("year", col("year").cast("float"))
    df = df.withColumn("month", col("month").cast("float"))
    df = df.withColumn("day_of_month", col("day_of_month").cast("float"))
    df = df.withColumn("day_of_week", col("day_of_week").cast("float"))
    df = df.withColumn("arrival_delay", col("arrival_delay").cast("float"))
    df = df.withColumn("departure_delay", col("departure_delay").cast("float"))
    df = df.withColumn("taxi_out", col("taxi_out").cast("float"))
    df = df.withColumn("distance", col("distance").cast("float"))
    df = df.withColumn("cancelled", col("cancelled").cast("float"))
    df = df.withColumn("flight_number", col("flight_number").cast("float"))
    return df


# Encode categorical features:
def encode_categorical_features(df):
    """
    Encode categorical features using StringIndexer and OneHotEncoder.
    The output is a new DataFrame with the specified columns encoded.

    Params:
    - df: Spark DataFrame

    Returns:
    - df_encoded: Spark DataFrame with categorical features encoded
    """
    # StringIndexer
    indexer = StringIndexer(
        inputCols=["airline_code", "origin", "dest", "plane_number"],
        outputCols=["airline_index", "origin_index", "dest_index", "plane_index"],
        handleInvalid="keep",
    )

    # OneHotEncoder
    encoder = OneHotEncoder(
        inputCols=["airline_index", "origin_index", "dest_index", "plane_index"],
        outputCols=[
            "airline_encoded",
            "origin_encoded",
            "dest_encoded",
            "plane_encoded",
        ],
    )

    # Pipeline
    pipeline = Pipeline(stages=[indexer, encoder])
    df_encoded = pipeline.fit(df).transform(df)

    return df_encoded


# Convert time to minutes:
def convert_time_to_minutes(df):
    """
    Convert time to minutes. Creates new columns with the time in minutes and drop the original ones.
    To compute it we take the first two digits and multiply by 60 and add the last two digits.
    This is applied to: actual_departure_time, scheduled_departure_time, scheduled_arrival_time, scheduled_flight_time
    Param:
    - df: spark dataframe

    Return:
    - df: spark dataframe with new columns with time in minutes
    """
    df = df.withColumn(
        "actual_departure_hour", (col("actual_departure_time") / 100).cast("int")
    )
    df = df.withColumn(
        "scheduled_departure_hour", (col("scheduled_departure_time") / 100).cast("int")
    )
    df = df.withColumn(
        "scheduled_arrival_hour", (col("scheduled_arrival_time") / 100).cast("int")
    )
    df = df.withColumn(
        "scheduled_flight_hour", (col("scheduled_flight_time") / 100).cast("int")
    )

    df = df.withColumn(
        "actual_departure_time_mins",
        (col("actual_departure_hour") * 60) + (col("actual_departure_time") % 100),
    )
    df = df.withColumn(
        "scheduled_departure_time_mins",
        (col("scheduled_departure_hour") * 60)
        + (col("scheduled_departure_time") % 100),
    )
    df = df.withColumn(
        "scheduled_arrival_time_mins",
        (col("scheduled_arrival_hour") * 60) + (col("scheduled_arrival_time") % 100),
    )
    df = df.withColumn(
        "scheduled_flight_time_mins",
        (col("scheduled_flight_hour") * 60) + (col("scheduled_flight_time") % 100),
    )

    df = df.drop(
        "actual_departure_hour",
        "scheduled_departure_hour",
        "scheduled_arrival_hour",
        "scheduled_flight_hour",
    )

    return df


# Keep only relevant columns:
def my_df(df):
    """
    Select columns to keep in the dataframe.
    Some columns are dropped as asked in the project instructions.
    Others are dropped because they are not useful for the model since they are derived from the original ones.

    Params:
    - df: spark dataframe

    Return:
    - df: spark dataframe with selected columns
    """
    df = df.select(
        "year",
        "month",
        "day_of_month",
        "day_of_week",
        "actual_departure_time_mins",
        "scheduled_departure_time_mins",
        "scheduled_arrival_time_mins",
        "airline_encoded",
        "flight_number",
        "scheduled_flight_time_mins",
        "departure_delay",
        "origin_encoded",
        "dest_encoded",
        "distance",
        "cancelled",
        "arrival_delay",
    )
    return df


# Handle null values:
def drop_nulls(df):
    """
    Drop rows with null values in the following columns: arrival_delay, scheduled_flight_time_mins, distance.

    Param:
    - df: spark dataframe

    Return:
    - df: spark dataframe with rows with null values dropped
    """
    # remove rows in arrival_delay where arrival_delay is null
    df = df.filter(df.arrival_delay.isNotNull())
    # remove rows in scheduled_flight_time_mins where departure_delay is null
    df = df.filter(df.scheduled_flight_time_mins.isNotNull())
    # remove rows in distance where distance is null
    df = df.filter(df.distance.isNotNull())
    return df


# Drop cancelled flights:
def drop_cancelled(df):
    """
    Drop rows with cancelled flights.

    Param:
    - df: spark dataframe

    Return:
    - df: spark dataframe with cancelled flights dropped
    """
    df = df.filter(df.cancelled == 0)
    return df


# standarize df
def standarize_dataframe(df):
    """
    Apply all the functions defined above to the dataframe.

    Param:
    - df: spark dataframe

    Return:
    - df: spark dataframe with all the functions applied
    """
    temp = edit_column_names(df)
    temp = string_to_float(temp)
    temp = encode_categorical_features(temp)
    temp = convert_time_to_minutes(temp)
    # Next 2 functions were defined after evaluating null values and whether
    #   to drop the entire variable or drop the rows with null values.
    temp = my_df(temp)
    temp = drop_nulls(temp)

    temp = drop_cancelled(temp)

    return temp


# Create model
def create_model(df, my_features):
    """
    Create a linear regression model.
        First, the data is split into train and test.
        Then, the features are vectorized and normalized.
        Finally, the model is created and fitted.

    Params:
    - df: spark dataframe
    - my_features: list of features to use in the model

    Return:
    - train: train: train data, test: test data, model: lr model
    """

    # Train-test split
    train, test = df.randomSplit([0.7, 0.3], seed=42)
    # Vectorize features
    featureassmebler = VectorAssembler(inputCols=my_features, outputCol="features")
    # Normlizer
    normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)
    # Linear Regression
    lr = LinearRegression(
        featuresCol="features_norm",
        labelCol="arrival_delay",
        maxIter=10,
        regParam=0.3,
        elasticNetParam=0.8,
    )
    # Pipeline
    pipeline = Pipeline(stages=[featureassmebler, normalizer, lr])
    # Fit pipeline
    model = pipeline.fit(train)

    return train, test, model


# Validate model
def validate_model(model, test_data):
    predictions = model.transform(test_data)
    evaluator = RegressionEvaluator(
        labelCol="arrival_delay", predictionCol="prediction", metricName="mae"
    )
    mae = evaluator.evaluate(predictions)
    print("MAE = %g" % mae)

    evaluator = RegressionEvaluator(
        labelCol="arrival_delay", predictionCol="prediction", metricName="mse"
    )
    mse = evaluator.evaluate(predictions)
    print("MSE = %g" % mse)

    evaluator = RegressionEvaluator(
        labelCol="arrival_delay", predictionCol="prediction", metricName="rmse"
    )
    rmse = evaluator.evaluate(predictions)
    print("RMSE = %g" % rmse)

    evaluator = RegressionEvaluator(
        labelCol="arrival_delay", predictionCol="prediction", metricName="r2"
    )
    r2 = evaluator.evaluate(predictions)
    print("R2 = %g" % r2)


#############################################


@click.command()
@click.option("--data_path", "-d", default=None, help="Path to the data folder")
def main(data_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if data_path is None:
        data_folder = os.path.join(script_dir, "data")
    else:
        data_folder = data_path
        click.echo("\n###\nData path selected: {}".format(data_folder))

    # Load data into pyspark dataframe
    df_pyspark = load_data(data_folder)
    while df_pyspark is False:
        path = input("Path: ")
        df_pyspark = load_data(path)

    print("\n###")
    print("Initial number of rows: ", df_pyspark.count())
    print("initial number of columns: ", len(df_pyspark.columns))

    # Process data
    df = standarize_dataframe(df_pyspark)
    print("\n###")
    print("Final number of rows: ", df.count())
    print("Final number of columns: ", len(df.columns))

    # Create model
    my_features = [
        "year",
        "month",
        "day_of_month",
        "day_of_week",
        "actual_departure_time_mins",
        "scheduled_departure_time_mins",
        "scheduled_arrival_time_mins",
        "airline_encoded",
        "flight_number",
        "scheduled_flight_time_mins",
        "departure_delay",
        "origin_encoded",
        "dest_encoded",
        "distance",
    ]

    print("\n###")
    print("Number of features selected: ", len(my_features))
    train, test, model = create_model(df, my_features)

    print("\n###")
    print("Coefficients: " + str(model.stages[-1].coefficients))
    print("Intercept: " + str(model.stages[-1].intercept))

    # Evaluate model
    validate_model(model, test)


if __name__ == "__main__":
    main()
