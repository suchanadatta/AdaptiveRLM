import sys, os, random
import numpy as np
import keras
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Lambda
from keras.layers.merge import concatenate
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam


if len(sys.argv) < 4:
    print('Needs 3 arguments - \n'
          '1. Batch size during training\n'
          '2. Batch size during testing\n'
          '3. No. of epochs\n')
    exit(0)

seed_value = 12321
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
np.random.seed(seed_value)

# command line both for train and test
DATADIR = '/store/causalIR/model-aware-qpp/input_data_t10/'   # (1)
DATADIR_idf = '/store/causalIR/model-aware-qpp/exp_hist_with_idf/input_10/'
DATADIR_idf_countidf = '/store/causalIR/model-aware-qpp/exp_hist_with_idf/input_idfcount_10/'


NUMCHANNELS = 1
# Num top docs (Default: 10)
K = 10   # (5)
# M: bin-size (Default: 30)
M = 120  # (6)
BATCH_SIZE_TRAIN = int(sys.argv[1])   # (7 - depends on the total no. of ret docs)
BATCH_SIZE_TEST = int(sys.argv[2])
EPOCHS = int(sys.argv[3])  # (8)
LR = 0.0001


class InteractionData:
    # Interaction data of query qid with K top docs -
    # each row vector is a histogram of interaction data for a document

    def __init__(self, qid, dataPathBase=DATADIR_idf):
        self.qid = qid
        histFile = "{}/{}.hist".format(dataPathBase, self.qid)
        # df = pd.read_csv(histFile, delim_whitespace=True, header=None)
        # self.matrix = df.to_numpy()
        histogram = np.genfromtxt(histFile, delimiter=" ")
        self.matrix = histogram[:, 4:]


class PairedInstance:
    def __init__(self, line):
        l = line.strip().split('\t')
        if len(l) > 2:
            self.qid_a = l[0]
            self.qid_b = l[1]
            self.class_label = int(l[2])
        else:
            self.qid_a = l[0]
            # self.qid_b = l[1]

    def __str__(self):
        return "({}, {})".format(self.qid_a, self.qid_b)

    def getKey(self):
        return "{}-{}".format(self.qid_a, self.qid_b)


class PointInstance:
    def __init__(self, line):
        l = line.strip().split('\t')
        if len(l) > 2:
            self.qid_a = l[0]
            self.qid_b = l[1]
            self.class_label = int(l[2])
        else:
            self.qid_a = l[0]
            # self.qid_b = l[1]

    def __str__(self):
        return "({})".format(self.qid_a)

    def getKey(self):
        return "{}".format(self.qid_a)


# Separate instances for training/test sets etc. Load only the id pairs.
# Data is loaded later in batches with a subclass of Keras generator
class PairedInstanceIds:
    '''
    Each line in this file should comprise three tab separated fields
    <id1> <id2> <label (1/0)>
    '''

    def __init__(self, idpairLabelsFile):
        self.data = {}

        with open(idpairLabelsFile) as f:
            content = f.readlines()

        # remove whitespace characters like `\n` at the end of each line
        for x in content:
            instance = PairedInstance(x)
            self.data[instance.getKey()] = instance


class PointInstanceIds:
    '''
    Each line in this file should comprise three tab separated fields
    <id1> <id2> <label (1/0)>
    '''

    def __init__(self, idpointLabelsFile):
        self.data = {}

        with open(idpointLabelsFile) as f:
            content = f.readlines()

        # remove whitespace characters like `\n` at the end of each line
        for x in content:
            instance = PointInstance(x)
            self.data[instance.getKey()] = instance

allPairs_train = PairedInstanceIds(DATADIR + 'train_input/qid_ap.pairs')   # (3)
allPairsList_train = list(allPairs_train.data.values())

allPoint_test = PointInstanceIds(DATADIR + 'test_input/qid.single')    # (4)
allPointsList_test = list(allPoint_test.data.values())

print ('{}/{} pairs for training'.format(len(allPairsList_train), len(allPairsList_train)))
print ('{}/{} pairs for testing'.format(len(allPointsList_test), len(allPointsList_test)))

'''
The files need to be residing in the folder data/
Each file is a matrix of values that's read using 
'''

class PairCmpDataGeneratorTrain(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, paired_instances_ids, dataFolder=DATADIR_idf, batch_size=BATCH_SIZE_TRAIN, dim_interaction=(K, M, NUMCHANNELS), dim_label=(1, 1, 1)):
        'Initialization'
        self.paired_instances_ids = paired_instances_ids
        self.dim = [dim_interaction, dim_interaction, dim_label]
        self.batch_size = batch_size
        self.dataDir = dataFolder
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paired_instances_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs = [self.paired_instances_ids[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs)

        return X

    def on_epoch_end(self):
        'Update indexes after each epoch'
        self.indexes = np.arange(len(self.paired_instances_ids))

    def __data_generation(self, list_IDs):
        'Generates data pairs containing batch_size samples'
        # Initialization
        X = [np.empty((self.batch_size, *self.dim[i])) for i in range(3)]
        # [print('X-shape : ', np.shape(j)) for j in X]
        Y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, paired_instance in enumerate(list_IDs):
            a_id = paired_instance.qid_a
            b_id = paired_instance.qid_b

            # read from the data file and construct the instances
            a_data = InteractionData(a_id, self.dataDir)
            b_data = InteractionData(b_id, self.dataDir)

            w, h = a_data.matrix.shape
            a_data.matrix = a_data.matrix.reshape(w, h, 1)
            b_data.matrix = b_data.matrix.reshape(w, h, 1)

            X[0][i,] = a_data.matrix
            X[1][i,] = b_data.matrix
            X[2][i,] = paired_instance.class_label
            Y[i] = paired_instance.class_label

        return X, Y


class PointCmpDataGeneratorTest(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, paired_instances_ids, dataFolder=DATADIR_idf, batch_size=BATCH_SIZE_TEST, dim_interaction=(K, M, NUMCHANNELS)):
        'Initialization'
        self.paired_instances_ids = paired_instances_ids
        self.dim = dim_interaction
        self.batch_size = batch_size
        self.dataDir = dataFolder
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paired_instances_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs = [self.paired_instances_ids[k] for k in indexes]
        # Generate data
        X = self.__data_generation(list_IDs)

        return X

    def on_epoch_end(self):
        'Update indexes after each epoch'
        self.indexes = np.arange(len(self.paired_instances_ids))

    def __data_generation(self, list_IDs):
        'Generates data pairs containing batch_size samples'
        # Initialization
        X = [np.empty((self.batch_size, *self.dim)) for i in range(1)]

        # Generate data
        for i, paired_instance in enumerate(list_IDs):
            a_id = paired_instance.qid_a
            # b_id = paired_instance.qid_b

            # read from the data file and construct the instances
            a_data = InteractionData(a_id, self.dataDir)
            # b_data = InteractionData(b_id, self.dataDir)

            w, h = a_data.matrix.shape
            a_data.matrix = a_data.matrix.reshape(w, h, 1)
            # b_data.matrix = b_data.matrix.reshape(w, h, 1)

            X[0][i,] = a_data.matrix
            # X[1][i,] = np.zeros((10, 120)).reshape(w, h, 1)
            # X[2][i,] = 0

        return X


def pair_loss(x):
    # Pair Loss function.
    query1, query2, label = x
    hinge_margin = 1
    keras.backend.print_tensor(query1)
    max_margin_hinge = hinge_margin - label * (query1 - query2)
    loss = keras.backend.maximum(0.0, max_margin_hinge)
    return loss


def identity_loss(y_true, y_pred):
    return keras.backend.mean(y_pred)


def base_model(input_shape):
    matrix_encoder = Sequential(name='sequence')
    matrix_encoder.add(Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))
    # matrix_encoder.add(Dense(500))
    matrix_encoder.add(MaxPooling2D(padding='same'))

    matrix_encoder.add(Conv2D(64, (3, 3), activation='relu'))
    # matrix_encoder.add(Dense(500))
    matrix_encoder.add(MaxPooling2D(padding='same'))

    matrix_encoder.add(Flatten())
    matrix_encoder.add(Dropout(0.2))
    matrix_encoder.add(Dense(128, activation='relu'))
    # matrix_encoder.add(Dense(16, activation='relu')) # eta kore kharap hoye geche
    matrix_encoder.add(Dense(1, activation='sigmoid'))

    return  matrix_encoder


def build_siamese_custom_loss(input_shape, input_label_shape, base_model):
    input_a = Input(shape=input_shape, dtype='float32')
    input_b = Input(shape=input_shape, dtype='float32')
    input_c = Input(shape=input_label_shape, dtype='float32')

    matrix_encoder = Sequential(name='sequence')
    matrix_encoder.add(Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))
    # matrix_encoder.add(Dense(500))
    matrix_encoder.add(MaxPooling2D(padding='same'))

    matrix_encoder.add(Conv2D(64, (3, 3), activation='relu'))
    # matrix_encoder.add(Dense(500))
    matrix_encoder.add(MaxPooling2D(padding='same'))

    matrix_encoder.add(Flatten())
    matrix_encoder.add(Dropout(0.2))
    matrix_encoder.add(Dense(128, activation='relu'))
    matrix_encoder.add(Dense(1, activation='linear'))

    encoded_a = base_model(input_a)
    encoded_b = base_model(input_b)

    pair_indicator = Lambda(pair_loss)([encoded_a, encoded_b, input_c])
    # prediction = Dense(1, activation='sigmoid')(pair_indicator)
    siamese_net_custom = Model(inputs=[input_a, input_b, input_c], outputs=pair_indicator)
    return siamese_net_custom

base = base_model((K, M, 1))
siamese_model_custom = build_siamese_custom_loss((K, M, 1), (1, 1, 1), base)
siamese_model_custom.compile(loss=identity_loss,
                             optimizer=Adam(LR),
                             metrics=['accuracy'])

siamese_model_custom.summary()

training_generator = PairCmpDataGeneratorTrain(allPairsList_train, dataFolder=DATADIR_idf + 'train_input/')
siamese_model_custom.fit_generator(generator=training_generator,
                            use_multiprocessing=True,
                            epochs=EPOCHS,
                            workers=4)
#                             # validation_split=0.2,
#                             # verbose=1)
#
# siamese_model.save_weights('/store/causalIR/model-aware-qpp/foo.weights')
test_generator = PointCmpDataGeneratorTest(allPointsList_test, dataFolder=DATADIR_idf + 'test_input/')
predictions = base.predict(test_generator)
print('predict ::: ', predictions)
print('predict shape ::: ', predictions.shape)

with open(DATADIR_idf + "foo.res", 'w') as outFile:     # (9)
    i = 0
    for entry in test_generator.paired_instances_ids:
        outFile.write(entry.qid_a + '\t' + str(round(predictions[i][0], 4)) + '\n')
        i += 1
outFile.close()

# measure pearson's 'r' and Kendall's 'tau'




