from kafka import KafkaProducer
import csv
import time

producer = KafkaProducer(bootstrap_servers='localhost:9092')

with open('uber_testData.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        message = ','.join(row)
        print(f"message -> {message.encode('utf-8')}")
        producer.send('uber-data', message.encode('utf-8'))
        time.sleep(10)

producer.flush()
