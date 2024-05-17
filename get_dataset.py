import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

no_metrics = 21

def get_data(software_name):
    path = os.path.join("..", "CK num of defect")
    path = os.path.join("..", path)
    path = os.path.join(path, "SelectedOnes")
    path = os.path.join(path, software_name)
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
            versions = []
            final_vals = []
            for j in sorted(dictionary):
                versions.append(str(j))
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
            return final_dataset, final_Y, len(final_vals), versions
        
def check_dataset(X, Y, no_versions):
    total_versions_1 = X.shape[1]
    new_dataset_predict = np.empty((X.shape[0], no_versions, X.shape[2]))
    Y_vals_predict = np.empty((X.shape[0], no_versions))
    j = 0
    counter = 0
    for i in range(X.shape[0]):
        val = True
        for l in range(no_versions):
            val = val and (np.any(X[i][l]!=0))
        if(val):
            counter = counter + 1
            val2 = False
            for k in range(no_versions, total_versions_1, 1):
                val2 = val2 or (np.all(X[i][k] == 0))
            
            if(val2):
                new_dataset_predict[j] = X[i, :no_versions]
                Y_vals_predict[j] = Y[i,:no_versions]
                j = j + 1
            
    return new_dataset_predict[:j], Y_vals_predict[:j]
    #return new_dataset_predict[1:j+1], Y_vals_predict[1:j+1]
 
    
def normalize_dataset(X, length):
    scaler = StandardScaler()
    for i in range(length):
        X[:, :, i] = scaler.fit_transform(X[:, :, i])
    return X
    
def handle_outlier_cases_main_dataset(versions, software_name):
    if(software_name == "jEdit"):
        for i in range(len(versions)):
            if(versions[i] == "3.2"):
                versions[i] = "3.2.1"
            elif(versions[i] == "4.0"):
                versions[i] = "4"
    elif(software_name == "poi"):
        for i in range(len(versions)):
            if(versions[i] == "2.0"):
                versions[i] = "2.0RC1"
            elif(versions[i] == "2.5"):
                versions[i] = "2.5.1"
    elif(software_name == "velocity"):
        for i in range(len(versions)):
            if(versions[i] == "1.6"):
                versions[i] = "1.6.1"
    elif(software_name == "xalan"):
        versions = ["2.4.0", "2.5.0", "2.6.0", "2.7.0"]
    elif(software_name == "xerces"):
        versions = ["init", "1.2.0", "1.3.0", "1.4.4"]
    return versions

#returns a numpy array of size no_samplesXtime_stepsXno_features and output of size no_samplesXtime_steps
def load_data(path, software_name, total_versions, no_versions, all_versions):
    csv_file = pd.read_csv(path)
    software_modules = csv_file[csv_file['name'] == software_name]
    software_modules = software_modules.drop('name', axis = 1)
    columns = software_modules.columns
    scaler = StandardScaler()
    for col in columns:
        if(col == "version" or col == "name.1" or col == "bug"):
            continue
        else:
            software_modules[col] = scaler.fit_transform(np.array(software_modules[col]).reshape((-1, 1)))
    length = software_modules.shape[0] 
    final_dataset = np.empty([int(length/total_versions), no_versions, no_metrics])
    output = np.zeros([int(length/total_versions), total_versions - no_versions, no_metrics])
    #output_target = np.empty([int(length/time_steps), time_steps - 1])
    output_target = np.zeros([int(length/total_versions), total_versions - no_versions, no_metrics])
    versions = all_versions
    versions = handle_outlier_cases_main_dataset(versions, software_name)
    print(versions)
    for i in range(len(versions)):
        if((versions[i] == "1.0" or versions[i] == "2.0" or versions[i] == "3.0") and software_name != "camel"):
            versions[i] = versions[i][0]
    for i in range(no_versions):
        print(versions[i])
        data = software_modules[software_modules['version'] == versions[i]]
        #a = input()
        data = data.drop(['name.1', 'version'], axis = 1)
        cols = data.columns
        final_dataset[:, i, :] = data
        
    for i in range(total_versions - no_versions):
        data = software_modules[software_modules['version'] == versions[i+no_versions]]
        data = data.drop(['name.1', 'version'], axis = 1)
        output_target[:, i, :] = data
        if(i<(total_versions - no_versions - 1)):
            output[:, i+1, :] = data
        
    return final_dataset, output, output_target, cols
    

def shift_data(X):
    X1 = np.zeros([X.shape[0], X.shape[1], X.shape[2]])
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            X1[i, j, :] = X[i, (j+1)%X1.shape[1], :]
    return X1
