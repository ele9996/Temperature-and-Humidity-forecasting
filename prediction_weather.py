
from board import D4
from datetime import datetime
import adafruit_dht
import csv
import time
import tensorflow as tf
import numpy as np
import tensorflow.lite as tflite
import pandas as pd
import argparse
import tensorflow_datasets as tfds
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='can be mlp_model, cnn_1d_model or lstm_model')
args = parser.parse_args()

MODEL= args.model
DELAY_TIME=1
NUMBERS_OF_CYCLE=7
input_width=6
ROOT_DIR= "./tflite_weather_forecasting_models/"
count=0
dht_device = adafruit_dht.DHT11(D4)

while True:
	temperature = dht_device.temperature
	humidity = dht_device.humidity
	day=datetime.today().strftime("%d/%m/%Y")
	ti=datetime.now().strftime("%H:%M:%S")

	print("inserted = {}, {}".format(temperature, humidity))

	with open('temp_humidity.csv', mode='a') as data:
	    data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	    data_writer.writerow([temperature, humidity])

	time.sleep(DELAY_TIME)
	count=count+DELAY_TIME
	if count==NUMBERS_OF_CYCLE:
		count=0
		break




#create the dataset
df = pd.read_csv('temp_humidity.csv')

dataset=df.to_numpy(dtype=np.float32)


#I define mean and standard deviation (for normalization)
#mean = #.mean(axis=0)
#std = #dataset.std(axis=0)

mean= np.array([ 9.107597,75.904076],dtype=np.float32)
std=np.array([ 8.654227,16.557089],dtype=np.float32)

#features: temperature, humidity (x6 values)
#one temperature value (the one corresponding to the next time interval)



#load the model

print ("load the model...")

# Load the TFLite model and allocate tensors.

if MODEL=="mlp_model":
	interpreter = tflite.Interpreter(model_path=ROOT_DIR + "mlp_model")

if MODEL=="cnn_1d_model":
	interpreter = tflite.Interpreter(model_path=ROOT_DIR + "cnn_1d_model")


if MODEL=="lstm_model":
	interpreter = tflite.Interpreter(model_path=ROOT_DIR + "lstm_model")


interpreter.allocate_tensors()


# Get input and output tensors.
input_details = interpreter.get_input_details()

print(input_details)
output_details = interpreter.get_output_details()

print(output_details)

def normalize(features):
	features = (features - mean) / (std + 1.e-6)

	return features

my_in = np.expand_dims(dataset,axis=0)
my_input=normalize(my_in)
print("Input:", my_input)


interpreter.set_tensor(input_details[0]['index'], my_input)

interpreter.invoke()

my_output = interpreter.get_tensor(output_details[0]['index'])

print("Output:", my_output)  


