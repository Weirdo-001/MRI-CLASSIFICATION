import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Fix for Windows PySpark "Cannot run program 'python3'" error
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = SparkSession.builder \
    .appName("BrainTumorMRIProcessing") \
    .getOrCreate()

# Example metadata dataset
image_data = [
    ("glioma_1.jpg", "glioma"),
    ("meningioma_1.jpg", "meningioma"),
    ("pituitary_1.jpg", "pituitary"),
    ("notumor_1.jpg", "notumor")
]
columns = ["image_name", "label"]

df = spark.createDataFrame(image_data, columns)

print("Dataset Overview")
df.show()

# Count per class
print("Class Distribution")
df.groupBy("label").count().show()

# Filtering Example
print("Glioma Samples")
glioma_df = df.filter(col("label") == "glioma")
glioma_df.show()

spark.stop()
