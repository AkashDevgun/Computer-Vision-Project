from __future__ import absolute_import
from __future__ import print_function

import numpy as np
#from keras.models import make_batches
from keras.engine.training import make_batches

#from .misc import shuffle_data, to_categorical
#from loading import load_samples



def load_samples(fpaths, nb_samples):
    # determine height / width
    img = load_img(fpaths[0])
    (width, height) = img.size
    
    #print (width)
    #print ("HI")
    #print (height)
    #print (width)
    print (nb_samples)

    # allocate memory
    sample_data = np.zeros((nb_samples, 3, height, width), dtype="uint8")

    counter = 0
    #print (sample_data.shape)
    for i in range(nb_samples):
        img = load_img(fpaths[i])
        #print (fpaths[i])
        #print (img)
        #imshow(img)
        #print (img)
        #print (img.shape)

        r, g, b = img.split()
        # print (r)
        # print (r.getdata())
        #print (np.array(r))
        #print (np.array(r).shape)
        # print (np.array(r).shape)
        # print (g)
        # print (A.shape)
        # #print (g.getdata())
        #print (np.array(g).T.shape)
        #print (b.shape)
        sample_data[counter, 0, :, :] = np.array(r)
        sample_data[counter, 1, :, :] = np.array(g)
        sample_data[counter, 2, :, :] = np.array(b)
        counter += 1
        #imshow(sample_data)
        #print (A.shape)

    return sample_data

#This one

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


def to_categorical(y_val, num_classes=None):
    '''Convert class vector (integers from 0 to num_classes)
        to binary class matrix, for use with categorical_crossentropy
    '''
    if len(y_val.shape) is not 2:
        y_val = np.reshape(y_val, (len(y_val), 1))
    y_val = np.asarray(y_val, dtype='int32')
    if not num_classes:
        num_classes = np.max(y_val)+1
    Y = np.zeros((len(y_val), num_classes))
    for i in range(len(y_val)):
        Y[i, y_val[i]] = 1.
    return Y


def training_batch(model, Xdata, y_val, num_classes,
                   callbacks=None, normalize=None, batch_size=32, class_weight=None, class_acc=True, shuffle=False):
    loss = []
    acc = []
    size = []



    num_samples = Xdata.shape[0]
    output_labels = ['loss', 'acc']

    if shuffle:
        Xdata, y_val = shuffle_data(Xdata, y_val)

    # batch train
    num_batches = make_batches(num_samples, batch_size)
    #print (batch_size)
    for batchindex, (batchstart, batchend) in enumerate(num_batches):
        batchresults = {}
        batchresults['batch'] = batchindex
        batchresults['size'] = batchend - batchstart

        if callbacks:
            callbacks.on_batch_begin(batchindex, batchresults)

        # load the actual images; Xdata only contains paths
        X_batchdata = load_samples(Xdata[batchstart:batchend], batchend - batchstart)
        X_batchdata = X_batchdata.astype("float32") / 255

        #print (X_batchdata.shape)
        #print ("Hi")


        y_batchdata = y_val[batchstart:batchend]
        y_batchdata = to_categorical(y_batchdata, num_classes)

        # if normalize:
        #     #for x in xrange(0,len(X_batchdata)):
        #         #print (X_batchdata[x].shape)
        #         #print (normalize[0].shape)
        #     X_batchdata = X_batchdata - normalize[0] # mean
        #     X_batchdata /= normalize[1] # std


        # calculates the overall loss and accuracy
        outputs = model.train_on_batch(X_batchdata, y_batchdata, accuracy=True, class_weight=class_weight)

        
        if type(outputs) != list:
            outputs = [outputs]
        for l, o in zip(output_labels, outputs):
            batchresults[l] = o

        # calculates the accuracy per class
        if class_acc:
            result = calculate_acc(model, Xdata[batchstart:batchend], y_val[batchstart:batchend], num_classes,
                                    normalize=normalize,
                                    batch_size=batch_size,
                                    keys=['acc'])
            batchresults['class_acc'] = result['acc']

        if callbacks:
            callbacks.on_batch_end(batchindex, batchresults)

    return loss, acc, size


#This one
def testing_batch(model, Xdata, y_val, num_classes, normalize=None, batch_size=32, shuffle=False):
    loss = []
    acc = []
    size = []

    num_samples = Xdata.shape[0]

    if shuffle:
        Xdata, y_val = shuffle_data(Xdata, y_val)

    # batch test
    num_batches = make_batches(num_samples, batch_size)
    for batchindex, (batchstart, batchend) in enumerate(num_batches):
        batchresults = {}
        batchresults['batch'] = batchindex
        batchresults['size'] = batchend - batchstart

        # load the actual images; Xdata only contains paths
        X_batchdata = load_samples(Xdata[batchstart:batchend], batchend - batchstart)
        X_batchdata = X_batchdata.astype("float32") / 255

        y_batchdata = y_val[batchstart:batchend]
        y_batchdata = to_categorical(y_batchdata, num_classes)

        if normalize:
            X_batchdata = X_batchdata - normalize[0] # mean
            X_batchdata /= normalize[1] # std

        outputs = model.test_on_batch(X_batchdata, y_batchdata, accuracy=True)

        # logging of the loss, acc and batch_size

        loss += [float(outputs[0])]
        acc += [float(outputs[1])]
        size += [batchend - batchstart]


    return loss, acc, size


#This one
def predicting_batch(model, Xdata, normalize=None, batch_size=32, shuffle=False, verbose=0):
    predictions = []

    #print (Xdata)
    num_samples = Xdata.shape[0]

    if shuffle:
        Xdata = shuffle_data(Xdata)

    # predict
    num_batches = make_batches(num_samples, batch_size)
    for batchindex, (batchstart, batchend) in enumerate(num_batches):
        batchresults = {}
        batchresults['batch'] = batchindex
        batchresults['size'] = batchend - batchstart

        # load the actual images; Xdata only contains paths
        X_batchdata = load_samples(Xdata[batchstart:batchend], batchend - batchstart)
        X_batchdata = X_batchdata.astype("float32") / 255
        if normalize:
            X_batchdata = X_batchdata - normalize[0] # mean
            X_batchdata /= normalize[1] # std

        predictions += [model.predict_classes(X_batchdata, verbose=verbose).tolist()]

    predictions = np.hstack(predictions).tolist()

    return predictions


#This one
def calculate_acc(model, X_test, y_test, num_classes, normalize=None, batch_size=32, keys=['acc', 'avg_acc']):
    loggingresults = {'match': np.zeros((num_classes,)), 'count': np.zeros((num_classes,))}

    predictions = predicting_batch(model, X_test, normalize=normalize, batch_size=batch_size)

    for gt, p in zip(y_test, predictions):
        loggingresults['count'][gt] += 1
        if gt == p:
            loggingresults['match'][gt] += 1

    loggingresults['acc'] = np.array(loggingresults['match'] / loggingresults['count']).tolist()
    loggingresults['avg_acc'] = np.mean(loggingresults['acc']).tolist()

    loggingresults['match'] = loggingresults['match'].tolist()
    loggingresults['count'] = loggingresults['count'].tolist()

    result_log = {key: loggingresults[key] for key in keys}

    return result_log
