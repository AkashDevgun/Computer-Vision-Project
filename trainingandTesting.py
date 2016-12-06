from __future__ import absolute_import
from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import CallbackList, ModelCheckpoint
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization

from sklearn.utils import compute_class_weight
import theano
import theano.tensor as T
from keras.constraints import Constraint

from callbacks import INIBaseLogger, INILearningRateScheduler, INILearningRateReducer, INIHistory
from DataAndFiles import functionsLogic
from batching import training_batch, testing_batch, predicting_batch, calculate_acc
from schedules import TriangularLearningRate
from optimizers import INISGD

import json
import os
import numpy as np

from preprocessing.image import load_img



#This one
def load_samples(fpaths, nb_samples):
	# determine height / width
	img = load_img(fpaths[0])
	(width, height) = img.size
	
	print (nb_samples)

	splitteddata = np.zeros((nb_samples, 3, height, width), dtype="uint8")

	counter = 0
	for i in range(nb_samples):
		img = load_img(fpaths[i])
	
		r, g, b = img.split()
	
		splitteddata[counter, 0, :, :] = np.array(r)
		splitteddata[counter, 1, :, :] = np.array(g)
		splitteddata[counter, 2, :, :] = np.array(b)
		counter += 1

	return splitteddata


def shuffle_data(Xdata, y_val=None, seedval=None):
	# shuffle data
	shuffling_index = np.arange(Xdata.shape[0])

	if seedval:
		np.random.seed(seedval)

	np.random.shuffle(shuffling_index)
	Xdata = Xdata[shuffling_index]
	if y_val is not None:
		y_val = y_val[shuffling_index]
		return Xdata, y_val
	return Xdata




def loadpathfromfiles(datapath, fname_x, fname_y, full_path=True):
	X_datapath = os.path.abspath(os.path.join(datapath, '..', fname_x))
	y_datapath = os.path.abspath(os.path.join(datapath, '..', fname_y))

	print (X_datapath)
	print (y_datapath)

	X_datapath = "/home/akash/Downloads/ComputerVision_Akash/caltech101/datasets/X_test.txt"
	y_datapath = "/home/akash/Downloads/ComputerVision_Akash/caltech101/datasets/y_test.txt"

	if os.path.isfile(X_datapath) and os.path.isfile(y_datapath):
		X = np.loadtxt(X_datapath, dtype=np.str_)
		if full_path:
			X = np.array([os.path.join(datapath, p) for p in X])
		y = np.loadtxt(y_datapath, dtype=np.int)
		print (len(X))
		print (len(y))
		#print (A.shape)

		return X, y
	else:
		raise Exception



def load_crossval_splitpaths(datapath, cv_fold, full_path=True):
		return loadpathfromfiles(datapath,
						  'cv{}_X_train.txt'.format(cv_fold),
						  'cv{}_y_train.txt'.format(cv_fold),
									 full_path=full_path), \
			   loadpathfromfiles(datapath,
						  'cv{}_X_test.txt'.format(cv_fold),
						  'cv{}_y_test.txt'.format(cv_fold),
									 full_path=full_path)




#this one
def loadcrossvalanws(datapath, cv_fold):

	data_path = "home/akash/Downloads/ComputerVision_Akash/caltech101/datasets/101_ObjectCategories"
	X_crossval_mean_path = os.path.abspath(os.path.join(datapath, '..', 'cv{}_X_mean.npy'.format(cv_fold)))
	X_crossval_std_path = os.path.abspath(os.path.join(datapath, '..', 'cv{}_X_std.npy'.format(cv_fold)))

	X_mean = np.load(X_crossval_mean_path)
	X_std = np.load(X_crossval_std_path)

	return X_mean, X_std




class Zero(Constraint):
	def __call__(self, p):
		p = T.zeros_like(p)
		return p


if __name__ == '__main__':
	

	# PARAMETERS

	batchsize = 32
	num_epoch = 20
	num_classes = 102
	shuffling_data = True
	normalizing_data = True
	batchnormalization = True

 
	bias_constraint = Zero() 

	# shape of the image 
	img_width, img_height = 180, 240

	dimensions = 3

	# DATA LOADING

	print("Loading paths")


	data_path = "home/akash/Downloads/ComputerVision_Akash/caltech101/datasets/101_ObjectCategories"


	(X_testing, y_testing) = loadpathfromfiles(data_path, 'X_test.txt', 'y_test.txt')

	for cross_val_folds in [0]: 
		print("fold {}".format(cross_val_folds))

		results = '_bn_triangular_cv{}_e{}'.format(cross_val_folds, num_epoch)

		(X_training, y_training), (X_validation, y_validation) = load_crossval_splitpaths(data_path, cross_val_folds)


		classweights = compute_class_weight('auto', range(num_classes), y_training)
			
		if normalizing_data:
			print("Load mean n STD")
			X_datamean, X_datastd = loadcrossvalanws(data_path, cross_val_folds)
		
			
			normalizing_data = (X_datamean, X_datastd)

		train_samples = X_training.shape[0]
		validation_samples = X_validation.shape[0]
		testing_samples = X_testing.shape[0]

		print('X_training shape:', X_training.shape)
		print(train_samples, 'train samples')
		if X_validation is not None:
			print(validation_samples, 'validation samples')
		print(testing_samples, 'test samples')

		if shuffling_data:
			(X_training, y_training) = shuffle_data(X_training, y_training, seed=None)
			(X_validation, y_validation) = shuffle_data(X_validation, y_validation, seed=None)
			(X_testing, y_testing) = shuffle_data(X_testing, y_testing, seed=None)

		
		# MODEL BUILDING
		

		print("Building The model")

		# Initializing the Parameters
		if batchnormalization:
			regularization_weight = 0.0005 
			dropout = False
			dropout_layer = False
			learningrate = 0.005
			learningrate_decay = 0.0005
		else:
			regularization_weight = 0.0005 
			dropout = True
			learningrate = 0.005
			learningrate_decay = 0.0005

		#Initializing the Model
		model = Sequential()

		#Initializing  and Adding the Convolutional Layer
		conv_layer1 = Convolution2D(128, 5, 5,
							  subsample=(2, 2), 
							  b_constraint=bias_constraint,
							  init='he_normal',
							  dim_ordering = "th",
							  W_regularizer=l2(regularization_weight),
							  input_shape=(dimensions, img_width, img_height))
		model.add(conv_layer1)

		if batchnormalization:
			model.add(BatchNormalization(mode = 1))
		
		#Adding The Activation Layer
		model.add(Activation('relu'))
		
		#Adding The Pooling Layer

		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		
		# Condition for adding dropout Layer
		if dropout:
			model.add(Dropout(0.5))

		#Initializing  and Adding the Convolutional Layer	
		conv_layer2 = Convolution2D(256, 3, 3, 
								b_constraint=bias_constraint, 
								init='he_normal', 
								W_regularizer=l2(regularization_weight))
		
		model.add(conv_layer2)
		
		if batchnormalization:
			model.add(BatchNormalization(mode=1))
		
		#Adding The Activation Layer
		model.add(Activation('relu'))
		
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		
		# Condition for adding dropout Layer
		if dropout:
			model.add(Dropout(0.5))

		model.add(ZeroPadding2D(padding=(1, 1)))
		
		#Initializing  and Adding the Convolutional Layer
		conv_layer3 = Convolution2D(512, 3, 3, 
								b_constraint=bias_constraint, 
								init='he_normal', 
								W_regularizer=l2(regularization_weight))
		
		model.add(conv_layer3)
		
		if batchnormalization:
			model.add(BatchNormalization(mode=1))
		
		#Adding The Activation Layer
		model.add(Activation('relu'))
		
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		
		# Condition for adding dropout Layer
		if dropout:
			model.add(Dropout(0.5))

		model.add(Flatten())

		model.add(Dense(1024, 
						b_constraint=bias_constraint, 
						init='he_normal', 
						W_regularizer=l2(regularization_weight)))

		if batchnormalization:
			model.add(BatchNormalization(mode=1))
		
		#Adding The Activation Layer
		model.add(Activation('relu'))

		# Condition for adding dropout Layer
		if dropout or dropout_layer:
			model.add(Dropout(0.5))

		model.add(Dense(num_classes, 
						b_constraint=bias_constraint, 
						init='he_normal', 
						W_regularizer=l2(regularization_weight)))
		
		model.add(Activation('softmax'))

		print('Compilation of model')
		sgd = INISGD(lr=learningrate, decay=learningrate_decay, momentum=0.9, nesterov=True)
		model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])


		# TRAINING
		
		callbacks = []
		previousLogs = INIHistory()
		callbacks += [previousLogs]

		logger = INIBaseLogger()
		callbacks += [logger]

		stepsize = 8 * (train_samples / batchsize) 
		schedule = TriangularLearningRate(lr=0.001, step_size=stepsize, max_lr=0.02, max_to_min=True)
		learningrates = INILearningRateScheduler(schedule, mode='batch', logger=logger)
		callbacks += [learningrates]


		callbacks = CallbackList(callbacks)

		shuffle_epoch = True
		metricsused = ['loss', 'acc', 'validation_loss', 'validation_accuracy', 'validationclass_accuracy'] 
		validation = True

		callbacks._set_model(model)
		callbacks._set_params({
			'batch_size': batchsize,
			'nb_epoch': num_epoch,
			'nb_sample': train_samples,
			'verbose': 1,
			'do_validation': validation,
			'metrics': metricsused,
		})

		
		# TRAINING START
		
		callbacks.on_train_begin()

		model.stop_training = False
		for epoch in range(num_epoch):
			callbacks.on_epoch_begin(epoch)



			if shuffle_epoch:
				X_training, y_training = shuffle_data(X_training, y_training)

			# train
			training_batch(model, X_training, y_training, num_classes,
								callbacks=callbacks,
								normalize=normalizing_data,
								batch_size=batchsize,
								class_weight=classweights,
								shuffle=False)

			print (epoch)
			print ("EPOCH")

			epoch_savingdata = {}

			
			# VALIDATION
			
			if validation:
				# calculating the overall loss and accuracy
				validation_loss, validation_accuracy, val_size = testing_batch(model, X_validation, y_validation, num_classes,
																 normalize=normalizing_data,
																 batch_size=batchsize,
																 shuffle=False)
				epoch_savingdata['validation_loss'] = validation_loss
				epoch_savingdata['validation_accuracy'] = validation_accuracy
				epoch_savingdata['val_size'] = val_size

				# calculating the accuracy per class
				class_acc = calculate_acc(model, X_validation, y_validation, num_classes,
												normalize=normalizing_data,
												batch_size=batchsize,
												keys=['acc'])
				epoch_savingdata['validationclass_accuracy'] = class_acc['acc']

			callbacks.on_epoch_end(epoch, epoch_savingdata)
			if model.stop_training:
				break

		trainingresults = {}

		
		# TESTING
		
		test_loss, test_acc, test_size = testing_batch(model, X_testing, y_testing, num_classes,
															normalize=normalizing_data,
															batch_size=batchsize,
															shuffle=False)

		trainingresults['test_loss'] = test_loss
		trainingresults['test_acc'] = test_acc
		trainingresults['test_size'] = test_size

		class_acc = calculate_acc(model, X_testing, y_testing, num_classes,
										normalize=normalizing_data,
										batch_size=batchsize,
										keys=['acc'])
		trainingresults['test_class_acc'] = class_acc['acc']

		callbacks.on_train_end(logs=trainingresults)

		
		# MODEL SAVING
		
		with open('results/{{}_architecture.json'.format(dt, results), 'w') as f:
			f.write(model.to_json())
		with open('results/{}_previousLogs.json'.format(dt, results), 'w') as f:
			f.write(json.dumps(previousLogs.previousLogs))
		model.save_weights('results/{}_weights.hdf5'.format(results))
