from kafka import KafkaConsumer
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import udf
from pyspark.ml.clustering import KMeansModel
from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors

#start spark session
spark = SparkSession.builder.getOrCreate()

#load trained model
model = KMeansModel.load('kmeans_model')

# Define a function to convert incoming data to a PySpark DataFrame
def to_dataframe(row):
    lat, lon, *features_str = row[0].split(",")
    feature1 = features_str[1].strip('[')
    feature2 = features_str[2].strip(']')
    features = Vectors.dense(feature1, feature2)
    return [(features)]

# Create a UDF to apply the to_dataframe function to each row of data
to_dataframe_udf = udf(to_dataframe)

# Create a PySpark DataFrame from the incoming data using the defined schema and UDF
def create_dataframe(data):
    rows = to_dataframe(data)
    return spark.createDataFrame([rows] , ['features'])

# Create a Kafka consumer instance
consumer = KafkaConsumer(
    'uber-data',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: m.decode('utf-8')
)

# Continuously consume messages from the Kafka topic
for message in consumer:
    # Extract the row data from the Kafka message
    row_data = message.value
    
    # Convert the row data to a PySpark DataFrame
    row_df = create_dataframe([row_data])
    
    #trasform trained model on received data
    predictions = model.transform(row_df)    

    #show predicted col
    predictions.select(col('prediction')).show()

    # Extract the predicted cluster index from the DataFrame
    prediction = predictions.select('prediction').collect()[0][0]

    # Print the predicted cluster index to the console
    print(f"Predicted cluster for row {row_data} -> prediced cluset: {prediction}")
