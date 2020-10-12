import numpy as np
import h5py
import pickle
from copy import deepcopy
from sklearn.metrics import confusion_matrix
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.preprocessing import sequence
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Convolution1D, MaxPooling1D

from keras.utils import np_utils
from MyNormalizer import token
import csv

################# GLOBAL VARIABLES #####################
# Filenames
# TODO: Add to coding conventions that directories are to always end with '/'
Masterdir = '../'
Datadir = 'Dataset/'
Modeldir = 'Models/'
Featuredir = 'Features/'
inputdatasetfilename = 'tamil_train_dev1.tsv'
inputdatasetfilename1 = 'test+lable.txt'
exp_details = 'new_experiment'

# Data I/O formatting
SEPERATOR = '\t'
DATA_COLUMN = 1
LABEL_COLUMN = 2
LABELS = ['0', '1','2','3','4']  # 0 -> Offensive, 1->Not-offensive
mapping_char2num = {}
mapping_num2char = {}
MAXLEN = 200

# LSTM Model Parameters
# Update your `MaxPooling1D` call to the Keras 2 API: `MaxPooling1D(pool_size=3)`
# model.add(MaxPooling1D(pool_length=pool_length))
# Embedding
MAX_FEATURES = 0
embedding_size = 128
# Convolution
filter_length = 3
nb_filter = 128
pool_length = 3
# LSTM
lstm_output_size = 128
# Training
batch_size = 128
number_of_epochs = 50
numclasses = 4
test_size = 0.1


########################################################

def parse(Masterdir, filename, seperator, datacol, labelcol, labels):
    """
    Purpose -> Data I/O
    Input   -> Data file containing sentences and labels along with the global variables
    Output  -> Sentences cleaned up in list of lists format along with the labels as a numpy array
    """
    # Reads the files and splits data into individual lines
    #f = open(Masterdir + Datadir + filename,'r',encoding='utf-8')
    f = open('E:/复现/BAKSA_IITK-master/HASOC_Off/Dataset/tamil_train_dev1.tsv', 'r', encoding='utf-8')
    lines = f.read().lower()
    # print(lines)

    lines = lines.lower().split('\n')[:-1]
    # print(lines)

    X_train = []
    Y_train = []
    D = []

    # Processes individual lines
    for line in lines:
        # Seperator for the current dataset. Currently '\t'.
        line = line.split(seperator)
        # print(line)

        # for l in line:
        #     re = line[0] + '\n'
        # # print(re)
        # fd = open('/home/lab1510/Desktop/Sub-word-LSTM-master/Data/digt.csv', 'w')
        # fd.write(re)

        # Token is the function which implements basic preprocessing as mentioned in our paper
        tokenized_lines = token(line[datacol])

        # Creates character lists
        char_list = []
        for words in tokenized_lines:
            for char in words:
                char_list.append(char)
            char_list.append(' ')
        # print(char_list) #- Debugs the character list created
        X_train.append(char_list)
        # print(X_train)

        # Appends labels
        if line[labelcol] == labels[0]:
            Y_train.append(0)
        if line[labelcol] == labels[1]:
            Y_train.append(1)
        if line[labelcol] == labels[2]:
            Y_train.append(2)
        if line[labelcol] == labels[3]:
            Y_train.append(3)
        if line[labelcol] == labels[4]:
            Y_train.append(4)
    # Converts Y_train to a numpy array
    Y_train = np.asarray(Y_train)
    assert (len(X_train) == Y_train.shape[0])

    return [X_train, Y_train]


def convert_char2num(mapping_n2c, mapping_c2n, trainwords, maxlen):
    """
    Purpose -> Convert characters to integers, a unique value for every character
    Input   -> Training data (In list of lists format) along with global variables
    Output  -> Converted training data along with global variables
    """
    allchars = []
    errors = 0

    # Creates a list of all characters present in the dataset
    for line in trainwords:
        try:
            allchars = set(allchars + line)
            allchars = list(allchars)
        except:
            errors += 1

    # print(errors) #Debugging
    # print(allchars) #Debugging

    # Creates character dictionaries for the characters
    charno = 0
    for char in allchars:
        mapping_char2num[char] = charno
        mapping_num2char[charno] = char
        charno += 1

    assert (len(allchars) == charno)  # Checks

    # Converts the data from characters to numbers using dictionaries
    X_train = []
    for line in trainwords:
        char_list = []
        for letter in line:
            char_list.append(mapping_char2num[letter])
        # print(no) -- Debugs the number mappings
        X_train.append(char_list)

    print(mapping_char2num)
    print(mapping_num2char)
    # Pads the X_train to get a uniform vector
    # TODO: Automate the selection instead of manual input
    X_train = sequence.pad_sequences(X_train[:], maxlen=maxlen)

    return [X_train, mapping_num2char, mapping_char2num, charno]



# Filenames
# TODO: Add to coding conventions that directories are to always end with '/'
from tensorflow.python.keras.models import load_model

Masterdir1 = '../'
Datadir1 = 'Dataset/'
Modeldir1 = 'Models/'
Featuredir1 = 'Features/'
inputdatasetfilename1 = 'tamil_test+lable.txt'
exp_details1 = 'new_experiment'

# Data I/O formatting
SEPERATOR1 = '\t'
DATA_COLUMN1 = 1
LABEL_COLUMN1 = 2
LABELS1 = ['0', '1', '2','3','4']  # 0 -> Negative, 1-> Neutral, 2-> Positive
mapping_char2num1 = {}
mapping_num2char1 = {}
MAXLEN1 = 200

# LSTM Model Parameters
# Update your `MaxPooling1D` call to the Keras 2 API: `MaxPooling1D(pool_size=3)`
# model.add(MaxPooling1D(pool_length=pool_length))
# Embedding
MAX_FEATURES1 = 0
embedding_size1 = 128
# Convolution
filter_length1 = 3
nb_filter1 = 128
pool_length1 = 3
# LSTM
lstm_output_size1 = 128
# Training
batch_size1 = 128
number_of_epochs1 = 50
numclasses1 = 5
test_size1 = 0.2

def parse1(Masterdir1, filename1, seperator1, datacol1, labelcol1, labels1):
    """
    #datacol表示句子
    Purpose -> Data I/O
    Input   -> Data file containing sentences and labels along with the global variables
    Output  -> Sentences cleaned up in list of lists format along with the labels as a numpy array
    """
    # Reads the files and splits data into individual lines
    #f = open(Masterdir1 + Datadir1 + filename1, 'r',encoding='utf-8')
    #f = open('E:/复现/BAKSA_IITK-master/HASOC_Off/Data/task1/翻译test+lable.txt', 'r', encoding='utf-8')
    f = open('E:/复现/BAKSA_IITK-master/HASOC_Off/Dataset/tamil_test+lable.txt', 'r', encoding='utf-8')

    lines = f.read().lower()
    #print(lines)

    lines = lines.lower().split('\n')[:-1]
    #print(lines)

    X_test = []
    Y_test = []


    # Processes individual lines
    for line in lines:
        # Seperator for the current dataset. Currently '\t'.
        line = line.split(seperator1)
        # Token is the function which implements basic preprocessing as mentioned in our paper
        for l in line:
            re = line[0] + '\n'

        # fd = open('E:/复现/BAKSA_IITK-master/Baseline/Sub-word-LSTM/pred/digt.csv', 'a')
        # fd.write(re)##145条test
        tokenized_lines = token(line[datacol1])
        #print('xxxxxx',line[datacol])#105条
        #print(tokenized_lines)#105条

        # Creates character lists
        char_list = []
        for words in tokenized_lines:
            for char in words:
                char_list.append(char)
            char_list.append(' ')
        #print(char_list) #- Debugs the character list created
        X_test.append(char_list)

        #Appends labels
        if line[labelcol1] == labels1[0]:
            Y_test.append(0)
        if line[labelcol1] == labels1[1]:
            Y_test.append(1)
        if line[labelcol1] == labels1[2]:
            Y_test.append(2)
        if line[labelcol1] == labels1[3]:
            Y_test.append(3)
        if line[labelcol1] == labels1[4]:
            Y_test.append(4)

        # for line in lines:
        #     i=0
        # Y_test.append(i)
        #print(Y_test)
        # Converts Y_train to a numpy array
    Y_test = np.asarray(Y_test)

    assert (len(X_test) == Y_test.shape[0])

    return [X_test, Y_test]


def convert_char2num1(mapping_n2c, mapping_c2n, trainwords, maxlen):
    """
    Purpose -> Convert characters to integers, a unique value for every character
    Input   -> Training data (In list of lists format) along with global variables
    Output  -> Converted training data along with global variables
    """
    allchars = []
    errors = 0

    # Creates a list of all characters present in the dataset
    for line in trainwords:
        try:
            allchars = set(allchars + line)
            allchars = list(allchars)
        except:
            errors += 1

    # print(errors) #Debugging
    # print(allchars) #Debugging

    # Creates character dictionaries for the characters
    charno = 0
    for char in allchars:
        mapping_char2num1[char] = charno
        mapping_num2char1[charno] = char
        charno += 1

    assert (len(allchars) == charno)  # Checks

    # Converts the data from characters to numbers using dictionaries
    X_test = []
    for line in trainwords:
        char_list = []
        for letter in line:
            char_list.append(mapping_char2num1[letter])
        # print(no) -- Debugs the number mappings
        X_test.append(char_list)

    print(mapping_char2num1)
    print(mapping_num2char1)
    # Pads the X_train to get a uniform vector
    # TODO: Automate the selection instead of manual input
    X_test = sequence.pad_sequences(X_test[:], maxlen=maxlen)
    #print('hhhhh',X_test)

    return [X_test, mapping_num2char1, mapping_char2num1, charno]






def RNN(X_train, y_train, args):
    """
    Purpose -> Define and train the proposed LSTM network
    Input   -> Data, Labels and model hyperparameters
    Output  -> Trained LSTM network
    """
    # Sets the model hyperparameters
    # Embedding hyperparameters
    max_features = args[0]
    maxlen = args[1]
    embedding_size = args[2]
    # Convolution hyperparameters
    filter_length = args[3]
    nb_filter = args[4]
    pool_length = args[5]
    # LSTM hyperparameters
    lstm_output_size = args[6]
    # Training hyperparameters
    batch_size = args[7]
    nb_epoch = args[8]
    numclasses = args[9]
    test_size = args[10]

    # Format conversion for y_train for compatibility with Keras
    y_train = np_utils.to_categorical(y_train, numclasses)

    # Train & Validation data splitting
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=test_size, random_state=42)

    # Build the sequential model
    # Model Architecture is:
    # Input -> Embedding -> Conv1D+Maxpool1D -> LSTM -> LSTM -> FC-1 -> Softmaxloss
    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(LSTM(lstm_output_size, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
    model.add(LSTM(lstm_output_size, dropout_W=0.2, dropout_U=0.2, return_sequences=False))
    model.add(Dense(numclasses))
    model.add(Activation('softmax'))

    # Optimizer is Adamax along with categorical crossentropy loss
    model.compile(loss='categorical_crossentropy',
                  optimizer='adamax',
                  metrics=['accuracy'])

    print('Train...')
    # Trains model for 50 epochs with shuffling after every epoch for training data and validates on validation data
    model.fit(X_train, y_train,
              batch_size=batch_size,
              shuffle=True,
              nb_epoch=nb_epoch,
              validation_data=(X_valid, y_valid))
    return model


def save_model(Masterdir, filename, model):
    """
    Purpose -> Saves Keras model files to the given directory
    Input   -> Directory and experiment details to be saved and trained model file
    Output  -> Nil
    """
    # Referred from:- http://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
    model.save_weights(Masterdir + 'Models/LSTM_' + filename + '_weights.h5')
    json_string = model.to_json()
    f = open(Masterdir + 'Models/' + 'LSTM_' + filename + '_architecture.h5', 'w')
    f.write(json_string)
    f.close()


def get_activations(model, layer, X_batch):
    """
    Purpose -> Obtains outputs from any layer in Keras
    Input   -> Trained model, layer from which output needs to be extracted & files to be given as input
    Output  -> Features from that layer
    """
    # Referred from:- TODO: Enter the forum link from where I got this
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output, ])
    activations = get_activations([X_batch, 0])
    return activations


def evaluate_model(X_test, y_test, model, batch_size, numclasses):
    """
    Purpose -> Evaluate any model on the testing data
    Input   -> Testing data and labels, trained model and global variables
    Output  -> Nil
    """
    # Convert y_test to one-hot encoding
    y_test = np_utils.to_categorical(y_test, numclasses)
    # Evaluate the accuracies
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)


def save_data(Masterdir, filename, X_train, X_test, y_train, y_test, features_train, features_test):
    """
    Purpose -> Saves train, test data along with labels and features in the respective directories in the folder
    Input   -> Train and test data, labels and features along with the directory and experiment details to be mentioned
    Output  -> Nil
    """
    h5f = h5py.File(Masterdir + Datadir + 'Xtrain_' + filename + '.h5', 'w')
    h5f.create_dataset('dataset', data=X_train)
    h5f.close()

    # h5f = h5py.File(Masterdir + Datadir + 'Xtest_' + filename + '.h5', 'w')
    # h5f.create_dataset('dataset', data=X_test)
    # h5f.close()

    output = open(Masterdir + Datadir + 'Ytrain_' + filename + '.pkl', 'wb')
    pickle.dump([y_train], output)
    output.close()

    # output = open(Masterdir + Datadir + 'Ytest_' + filename + '.pkl', 'wb')
    # pickle.dump([y_test], output)
    # output.close()

    h5f = h5py.File(Masterdir + Featuredir + 'features_train_' + filename + '.h5', 'w')
    h5f.create_dataset('dataset', data=features_train)
    h5f.close()

    h5f = h5py.File(Masterdir1 + Featuredir1 + 'features_test_' + filename1 + '.h5', 'w')
    h5f.create_dataset('dataset', data=features_test)
    h5f.close()


from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


def accuracy(original, predicted):
    print("F1 score is: " + str(f1_score(original, predicted, average='macro')))
    scores = confusion_matrix(original, predicted)
    print(scores)
    print(np.trace(scores) / float(np.sum(scores)))



if __name__ == '__main__':
    """
    Master function
    """
    print('Starting RNN Engine...\nModel: Char-level LSTM.\nParsing data files...')
    out = parse(Masterdir, inputdatasetfilename, SEPERATOR, DATA_COLUMN, LABEL_COLUMN, LABELS)
    X_train = out[0]
    y_train = out[1]
    out = parse1(Masterdir1, inputdatasetfilename1, SEPERATOR1, DATA_COLUMN1,LABEL_COLUMN1, LABELS1)
    X_test = out[0]
    y_test =out[1]
    # print(DATA_COLUMN)
    # print('X_test',X_test)
    # print('y_test',y_test)
    print('Parsing complete!')

    print('Creating character dictionaries and format conversion in progess...')
    out = convert_char2num(mapping_num2char, mapping_char2num, X_train, MAXLEN)
    mapping_num2char = out[1]
    mapping_char2num = out[2]
    MAX_FEATURES = out[3]
    X_train = np.asarray(out[0])
    y_train = np.asarray(y_train).flatten()

    out = convert_char2num1(mapping_num2char1, mapping_char2num1, X_test, MAXLEN1)
    mapping_num2char = out[1]
    mapping_char2num = out[2]
    MAX_FEATURES = out[3]
    X_test = np.asarray(out[0])
    y_test = np.asarray(y_test).flatten()
    print('Complete!')

    print('Splitting data into train and test...')
    #X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    # print(X_test)

    print('Creating LSTM Network...')
    model = RNN(deepcopy(X_train), deepcopy(y_train), [MAX_FEATURES, MAXLEN, embedding_size, \
                                                       filter_length, nb_filter, pool_length, lstm_output_size,
                                                       batch_size, \
                                                       number_of_epochs, numclasses, test_size])

    # print('Evaluating model...')
    # evaluate_model(X_test, deepcopy(y_test), model, batch_size, numclasses)

    print('Feature extraction pipeline running...')
    activations = get_activations(model, 4, X_train)
    features_train = np.asarray(activations)
    activations = get_activations(model, 4, X_test)
    features_test = np.asarray(activations)
    print('Features extracted!')

    # print('Saving experiment...')
    # save_model(Masterdir,exp_details,model)
    # save_data(Masterdir,exp_details,X_train,X_test,y_train,y_test,features_train,features_test)
    # #save_data(Masterdir, exp_details, X_train,  y_train,  features_train)
    # print('Saved! Experiment finished!')




Featuredir1 = 'Features/'
experiment_details1 = 'lstm128_subword'

#y_test = np.asarray(y_test).flatten()
y_test = np.asarray(y_test).flatten()
y_test2 = np_utils.to_categorical(y_test, numclasses)

print('Evaluating model...')
print(y_test.shape)
y_pred = model.predict_classes(X_test, batch_size=batch_size)
evaluate_model(X_test, deepcopy(y_test), model, batch_size, numclasses1)
accuracy(y_test, y_pred)
print(y_pred)
for i in y_pred:
    p=str(i)+'\n'

    with open('E:/复现/BAKSA_IITK-master/HASOC_Off/Data/task1/pre/pred.csv', 'a', encoding='UTF-8') as f1:
        f1.write(str(p))

sentiment = ' '
str1 = []
k = 1
with open('E:/复现/BAKSA_IITK-master/HASOC_Off/Data/task1/pre/pred.csv', 'r') as out:
    #    out.write('Uid,Sentiment')
    for line in out:
        #print("1:",line)
        #print("12:",type(line))
        # line = line.strip().split(' ')
        if int(line) == 0:
            #print("=============")
            sentiment = 'Off'
        elif int(line) == 1:
            sentiment = 'Not'
        str1 = str(k) + ',' + str(sentiment) + '\n'
        k += 1
        # print(str1)
        fd = open('E:/复现/BAKSA_IITK-master/HASOC_Off/Dataset/pred3.csv', 'a', encoding='utf-8')
        fd.write(str1)