# Create Spark Context with SparkConf
from pyspark import SparkConf, SparkContext, SparkFiles
import time

conf = SparkConf()
sc = SparkContext.getOrCreate(conf)
conf.setAppName("App")
sc.setLogLevel("ERROR")

inputFilePath = "pagecounts"
rdd = sc.textFile(inputFilePath)

enPages = rdd.filter(lambda line: line.startswith('en '))
numEnLines = enPages.count()
print('Number of EN pages:', numEnLines)

enPagesTuples = enPages.flatMap(lambda line: [(pieces[0], pieces[1], int(pieces[2]), int(pieces[3]))
                                              for pieces in [line.split(" ")] if len(pieces) == 4])
for line in enPagesTuples.take(10):
    print(line)

topSortedEnPages = enPagesTuples.sortBy(lambda x: x[2], ascending=False) \
    .take(10) \

for page in topSortedEnPages:
    print(page)

time.sleep(300)

sc.stop()
