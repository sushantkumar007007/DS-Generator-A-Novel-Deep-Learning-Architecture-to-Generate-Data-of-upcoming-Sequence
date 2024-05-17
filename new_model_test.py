import tensorflow as tf

import os
import pandas as pd
import numpy as np
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential, Model
from keras.layers import Embedding, Input
from keras.layers import Dense, LSTM, Dropout
from keras.layers import TimeDistributed, RepeatVector
from keras.utils.vis_utils import plot_model
from keras.backend.tensorflow_backend import set_session  

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
#from imblearn.over_sampling import SMOTE

time_steps = 5
no_features = 20
no_metrics = 21
epochs_1 = 300
epochs_2 = 300

def get_dataset():
    path = os.path.join("..", "CK num of defect")
    path = os.path.join("..", path)
    path = os.path.join(path, "ant")
    for i in os.walk(path):
        files = i[2]
        if(len(files)>1):
            list2 = files.copy()
            for j in range(len(files)):
                #print(files[j][:-4])
                files[j] = float(files[j][:-4].split('-')[1])
            dictionary = {}
            for j in range(len(files)):
                dictionary[files[j]] = list2[j]
            final_vals = []
            for j in sorted(dictionary):
                final_vals.append(dictionary[j])
            #print(final_vals)
            module_names = []
            for file in final_vals:
                csv_file = pd.read_csv(os.path.join(path, file))
                total_modules = list(set(csv_file['name.1']))
                module_names.extend(total_modules)
                module_names = list(set(module_names))
            module_names_dict = {}
            for j in range(len(module_names)):
                module_names_dict[module_names[j]] = j
            final_dataset = np.zeros((len(module_names), len(final_vals), no_metrics))
            final_Y = np.empty((len(module_names), len(final_vals)))
            for file in range(len(final_vals)):
                csv_file = pd.read_csv(os.path.join(path, final_vals[file]))
                csv_file = csv_file.drop(['name', 'version'], axis = 1)
                for j in range(csv_file.shape[0]):
                    module_vals = csv_file.iloc[j]
                    y_val = module_vals['bug']
                    module_name = module_vals['name.1']
                    x_vals = np.array(module_vals.drop(['name.1']))
                    final_dataset[module_names_dict[module_name], file, :] = x_vals
                    final_Y[module_names_dict[module_name], file] = y_val
            
            #print(module_names[10])
            #print(final_dataset[10,:,:])
            return final_dataset, final_Y
                    

def check_dataset(X, Y):
    new_dataset_predict = np.empty((X.shape[0], 2, X.shape[2]))
    Y_vals_predict = np.empty((X.shape[0], 2))
    j = 0
    counter = 0
    for i in range(X.shape[0]):
        if(np.any(X[i][0] != 0) and np.any(X[i][1]!=0)):
            counter = counter + 1
            if(np.all(X[i][2] == 0) or np.all(X[i][3] == 0) or np.all(X[i][4] == 0)):
                new_dataset_predict[j] = X[i, :2]
                Y_vals_predict[j] = Y[i,:2]
                j = j + 1
    
    #print(counter)
    return new_dataset_predict[1:j+1], Y_vals_predict[1:j+1]
 
    
def normalize_dataset(X, length):
    scaler = StandardScaler()
    for i in range(length):
        X[:, :, i] = scaler.fit_transform(X[:, :, i])
        

#returns a numpy array of size no_samplesXtime_stepsXno_features and output of size no_samplesXtime_steps
def load_data(path):
    csv_file = pd.read_csv(path)
    ant_modules = csv_file[csv_file['name'] == 'ant']
    ant_modules = ant_modules.drop('name', axis = 1)
    columns = ant_modules.columns
    scaler = StandardScaler()
    for col in columns:
        if(col == "version" or col == "name.1" or col == "bug"):
            continue
        else:
            ant_modules[col] = scaler.fit_transform(np.array(ant_modules[col]).reshape((-1, 1)))
    length = ant_modules.shape[0] 
    final_dataset = np.empty([int(length/time_steps), 2, no_metrics])
    output = np.zeros([int(length/time_steps), 3, no_metrics])
    #output_target = np.empty([int(length/time_steps), time_steps - 1])
    output_target = np.zeros([int(length/time_steps), 3, no_metrics])
    versions = ['1.3', '1.4', '1.5', '1.6', '1.7']
    
    for i in range(2):
        data = ant_modules[ant_modules['version'] == versions[i]]
        data = data.drop(['name.1', 'version'], axis = 1)
        cols = data.columns
        final_dataset[:, i, :] = data
        
    for i in range(3):
        data = ant_modules[ant_modules['version'] == versions[i+2]]
        data = data.drop(['name.1', 'version'], axis = 1)
        output_target[:, i, :] = data
        if(i<2):
            output[:, i+1, :] = data 
        
    return final_dataset, output, output_target, cols
    

def shift_data(X):
    X1 = np.zeros([X.shape[0], X.shape[1], X.shape[2]])
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            X1[i, j, :] = X[i, (j+1)%X1.shape[1], :]
    return X1

def fit_model(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (time_steps, no_features)))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
   
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    
    model.add(Dense(units = 5))
    
    #model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', 'mse', 'mae'])

    model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 100, batch_size = 32)    
    
    print(model.metrics_names)
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
    #print("%s: %.2f%%" % (model.metrics_names[3], scores[3]*100))
    #print(model.evaluate(X_test, Y_test))   
    #_, test_accuracy = model.evaluate(X_test, Y_test)
    #_, train_accuracy = model.evaluate(X_train, Y_train)
    #print("Test Accuracy : %.3f" %(test_accuracy))
    #print("Train Accuracy: %.3f" %(train_accuracy))
    #print(predictions)
    

#encoder input: batch_size x no_time_steps x no_features (This allows no_time_steps to be variable for each input, so we may need to pad the 
#sequences and instead of no_time_steps, this becomes the maximum no. of versions over all the inputs.)
#decoder input: batch_size x no_time_steps x output_size
    
def decoder_network(inputs, initial_state):
    decoder_lstm_l1 = LSTM(256, dropout = 0.3, return_sequences = True)
    decoder_lstm_l2 = LSTM(128, dropout = 0.4, return_sequences = True)
    decoder_lstm_l3 = LSTM(64, dropout = 0.2, return_sequences = True)
    decoder_lstm_l4 = LSTM(128, dropout = 0.3, return_sequences = True)
    decoder_lstm_l5 = LSTM(256, dropout = 0.4, return_sequences = True, return_state = True) #should be of same shape as 1st layer of decoder, as this output
    #is used as input to decoder again when predicting multiple time steps
    return decoder_lstm_l5(decoder_lstm_l4(decoder_lstm_l3(decoder_lstm_l2(decoder_lstm_l1(inputs, initial_state = initial_state)))))


def seq_to_seq_model(X, Y, Y_target):
    #latent_dim = 128
    no_outputs = no_metrics
    #This NONE is for the number of time steps
    encoder_inputs = Input(shape = (None, no_metrics))
    encoder_l1 = LSTM(512, dropout = 0.3, return_sequences = True)
    encoder_l2 = LSTM(256, dropout = 0.4, return_sequences = True)
    encoder_l3 = LSTM(128, dropout = 0.2, return_sequences = True)
    encoder_l4 = LSTM(256, dropout = 0.1, return_state = True)   # This should be same as the shape of the 1st layer of Decoder
    #encoder = LSTM(latent_dim[0], return_state = True, return_sequences = True)(encoder_inputs)
    #encoder1 = LSTM(latent_dim[1], return_state = True)(encoder)
    encoder_outputs, state_h, state_c = encoder_l4(encoder_l3(encoder_l2(encoder_l1(encoder_inputs))))
    encoder_states = [state_h, state_c]
    
    #only the number of bugs needs to be predicted, so the output is 1, else it should be taken as no_features+1
    decoder_inputs = Input(shape = (None, no_outputs))
    decoder_outputs, _, _ = decoder_network(decoder_inputs, encoder_states)
    decoder_dense = Dense(no_outputs, activation = 'softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    #plot_model(model, to_file = 'seq_model_1.png', show_shapes = True)
   
    
    #PREDICTION PHASE
    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape = (256,)) #Same as the shape of the 1st layer of decoder
    decoder_state_input_c = Input(shape = (256,))
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_network(decoder_inputs, decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs]+decoder_states_inputs, [decoder_outputs] + decoder_states)
    
    model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy', 'mse', 'mae'])
    model.fit([X, Y], Y_target, batch_size = 32, epochs = epochs_1, validation_split = 0.2)
    
    scores = model.evaluate([X, Y], Y_target, batch_size = 32)
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
    
    return model, encoder_model, decoder_model
    #plot_model(encoder_model, to_file='encoder_model.png', show_shapes=True)
    #plot_model(decoder_model, to_file='decoder_model.png', show_shapes=True)
    
    '''for input_seq in X_test:
        states_value = encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, 1))
        target_seq[0, 0, ]
    '''

def seq_to_seq_model_2(X, Y):
    X_divisible, Y_divisible = X[:112], Y[:112] #as batch_size should divide the total number of samples
    model = Sequential()
    model.add(LSTM(512, return_sequences = True, batch_input_shape = (8, 2, no_metrics)))
    model.add(LSTM(256, return_sequences = True))
    model.add(LSTM(128, return_sequences = True))
    model.add(Dropout(0.5))
    model.add(LSTM(128, return_sequences = True))
    model.add(LSTM(64, return_sequences = True))
    model.add(Dropout(0.5))
    model.add(LSTM(32, return_sequences = True))
    model.add(LSTM(16, stateful = True))
    model.add(RepeatVector(3))
    model.add(LSTM(256, return_sequences = True))
    model.add(LSTM(128, return_sequences = True))
    model.add(LSTM(64, return_sequences = True))
    model.add(LSTM(32, return_sequences = True))
    model.add(TimeDistributed(Dense(no_metrics)))
    model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy', 'mse', 'mae'])
    model.fit(X_divisible, Y_divisible, epochs = epochs_1, batch_size = 8, verbose = 0)
    
    scores = model.evaluate(X_divisible, Y_divisible, verbose=0, batch_size = 8)
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
    return model

def predict(X_test, model):
    out_Y = model.predict(X_test, batch_size = 8)
    print(out_Y.shape)
    return out_Y
    
    
def predict_seq_to_seq_model(infenc, infdec, source, n_steps, cardinality):
    #Here cardinality is the number of features
    state = infenc.predict(source)
    target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
    
    output = list()
    for t in range(n_steps):    
        yhat, h, c = infdec.predict([target_seq] + state)
        output.append(yhat[0, 0, :])
        state = [h, c]
        target_seq = yhat
        
    return np.array(output)


def predict_unseen_version(X):
    train_x = X[:, :4, :]
    train_y = X[:, 4, :]
    train_y = np.squeeze(train_y)
    #smt = SMOTE()
    #train_x, train_y = smt.fit_sample(train_x, train_y) #train_x: 125 X 4 X 21, train_y: 125 X 4 X 21
    model = Sequential()
    #model.add(Embedding(1024, 1, input_length=20))
    model.add(LSTM(100, input_shape = (time_steps - 1, no_metrics), dropout = 0.2, return_sequences=True))
    #model.add(LSTM(512, dropout = 0.3, return_sequences=True))
    #model.add(LSTM(256, dropout = 0.4, return_sequences=True))
    model.add(LSTM(80, dropout = 0.2, return_sequences=True))
    model.add(LSTM(60, dropout = 0.2, return_sequences=False))
    #model.add(LSTM(16, input_shape=(1,20), dropout = 0.67))
    #model.add(Dense(1, activation='softmax'))
    #model.add(SpatialDropout1D(0.7))
    #model.add(LSTM(64, dropout=0.7))
    #model.add(LSTM(16, dropout=0.7, return_sequences=True))
    #model.add(LSTM(8, dropout=0.7))
    #model.add(Dense(10, input_dim = input_dim , activation = 'relu'))
    #model.add(Dense(10, activation = 'relu'))
    #model.add(Dense(8, activation = 'relu'))
    #model.add(Dense(8, activation = 'relu'))
    #model.add(Dense(6, activation = 'relu'))
    #model.add(Dense(32, activation = 'relu'))
    model.add(Dense(no_metrics, activation = 'softmax'))
    model.compile(loss = 'mse' , optimizer = 'adam' , metrics = ['accuracy','mse', 'mae', 'mape', 'cosine'] )
    history = model.fit(train_x, train_y, validation_split = 0.3, epochs = epochs_2, batch_size = 64)
    scores = model.evaluate(train_x, train_y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
    print("%s: %.2f%%" % (model.metrics_names[3], scores[3]*100))
    print("%s: %.2f%%" % (model.metrics_names[4], scores[4]*100))
    
    output = model.predict(X[:, 1:, :])
    print(output.shape)
    output = np.reshape(output, (output.shape[0], 1, output.shape[1]))
    X_new = np.append(X, output, axis = 1)
    print(X_new.shape)
    return X_new
    
    
X, Y = get_dataset()                      #(751, 5, 21), (751, 5) arrays
X_to_predict, Y_predict = check_dataset(X, Y)
normalize_dataset(X, no_metrics-1)
normalize_dataset(X_to_predict, no_metrics-1)
print(X_to_predict.shape, Y_predict.shape)
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True 
sess = tf.Session(config=config)  
set_session(sess)
#How to fix noumber of time steps here, because the number of versions are different for each software.
X, Y, Y_target, cols = load_data(os.path.join('..', '..', 'CK num of defect', 'maindataset.csv'))
cols = np.array(cols)
print(X.shape, Y.shape, Y_target.shape)
#X_shifted = shift_data(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 10)
scaler = MinMaxScaler()
for i in range(X_train.shape[0]):
    X_train[i] = scaler.fit_transform(X_train[i])
for i in range(X_test.shape[0]):
    X_test[i] = scaler.transform(X_test[i])

#fit_model(X_train, Y_train, X_test, Y_test)



#model = seq_to_seq_model_2(X, Y)
#X = np.append(X, X_to_predict, axis = 0)
#decoded_seq = predict(X_to_predict, model)
#Y = np.append(Y, decoded_seq, axis = 0)
#final_dataset = np.concatenate((X, Y), axis = 1)
#final_dataset = np.reshape(final_dataset, (-1, final_dataset.shape[2]))
#df = pd.DataFrame(data = final_dataset, columns = cols)
#for i in range(decoded_seq.shape[0]):
#    print(decoded_seq[i])
    
#df.to_csv('new_ant_modules_2_with_proper_scaling.csv')




Y = np.reshape(Y, (Y.shape[0], Y.shape[1], no_metrics))
Y_target = np.reshape(Y_target, (Y_target.shape[0], Y_target.shape[1], no_metrics))
model, infenc, infdec = seq_to_seq_model(X, Y, Y_target)
X = np.append(X, X_to_predict, axis = 0)
for i in range(X_to_predict.shape[0]):    
    decoded_sequence = predict_seq_to_seq_model(infenc, infdec, X_to_predict[i].reshape((1, 2, no_metrics)), 3, no_metrics)
    #decoded_seq = predict(X_to_predict[i].reshape((1, 2, no_metrics)), model)
    decoded_sequence = decoded_sequence.reshape((1, 3, no_metrics))
    Y_target = np.append(Y_target, decoded_sequence, axis = 0)
    
final_dataset = np.concatenate((X, Y_target), axis = 1)
new_version = predict_unseen_version(final_dataset)
new_final_dataset = np.reshape(new_version, (-1, no_metrics))
#final_dataset = np.reshape(final_dataset, (-1, no_metrics))
df = pd.DataFrame(data = new_final_dataset, columns = cols)
df.to_csv('6thversion.csv')