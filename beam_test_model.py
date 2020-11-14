import tensorflow as tf
import numpy as np
from tensorflow.keras.models import model_from_json,Model
import argparse

##############################################################################

# Load Model

##############################################################################


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into the model
loaded_model.load_weights("model.h5")


###############################################################################

#//User input//#

###############################################################################

parser = argparse.ArgumentParser(description='Configure the files before training the net.')

parser.add_argument('data_folder', help='Location of the data directory', type=str)

args = parser.parse_args('s010/baseline_data'.split())

data_dir = args.data_folder+'/'

###############################################################################

# Coordinate configuration

#test

coord_test_input_file = data_dir+'coord_input/coord_test.npz'

print("Reading dataset... ",coord_test_input_file)

coord_test_cache_file = np.load(coord_test_input_file)

X_coord_test = coord_test_cache_file['coordinates']  

###############################################################################

# LIDAR configuration

#Test

lidar_test_input_file = data_dir+'lidar_input/lidar_test.npz'

print("Reading dataset... ",lidar_test_input_file)

lidar_test_cache_file = np.load(lidar_test_input_file)

X_lidar_test = lidar_test_cache_file['input']

##############################################################################

# Model Prediction

##############################################################################
	
Prediction = loaded_model.predict([X_coord_test, X_lidar_test])

np.savetxt('beam_test_pred.csv', Prediction, delimiter=',')
