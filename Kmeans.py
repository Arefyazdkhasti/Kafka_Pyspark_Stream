from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import ClusteringEvaluator

# Create a SparkSession object
spark = SparkSession.builder.appName("UberAnalysis").getOrCreate()

# Read the data from CSV file into a DataFrame
df = spark.read.csv("uber.csv", header=True, inferSchema=True)

# Combine the 'Lat' and 'Lon' columns into a single feature vector column
assembler = VectorAssembler(inputCols=['Lat', 'Lon'], outputCol='features')
data = assembler.transform(df.select('Lat', 'Lon'))


# Split the data into training and testing sets with a 80:20 ratio
(trainingData, testData) = data.randomSplit([0.8, 0.2], seed=10)

# Create a list to store the cost values for different values of k
cost = []

# Perform K-means clustering for k values ranging from 2 to 10
for i in range(2, 11):
    kmeans = KMeans(k=i, seed=0)
    model = kmeans.fit(data.select('Lat', 'Lon' , 'features'))
    predictions = model.transform(data)
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    print(f"ComputeCost in {i} = " + str(silhouette))
    cost.append(silhouette)

# Plot the cost values for different values of k
plt.plot(range(2, 11), cost)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Cost')
plt.show()


# Train the K-means clustering model on the training set
kmeans = KMeans(k=5, seed=0)
model = kmeans.fit(trainingData)

# Predict the cluster labels for the testing set
predictions = model.transform(testData)

# Save the trained model to a file
# model.save("kmeans_model")

predictions.show(20)

predictions.select(col('prediction')).show()