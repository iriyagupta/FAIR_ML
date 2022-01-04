import json
import tensorflow as tf
import numpy as np
import re,os,sys

class CompasInput():
    def __init__(self,
               mean_std_file=None,
               train_file=None,
               test_file=None,
               vocab_file=None,
               protected_column_values=None):

        self._mean_std_file = mean_std_file
        self._train_file = train_file
        self._test_file = test_file
        self._vocab_file = vocab_file

        self.feature_names = { 'juv_fel_count':0,
                              'juv_misd_count':1, 
                              'juv_other_count':2, 
                              'priors_count':3,
                              'age':4, 
                              'c_charge_degree':5, 
                              'c_charge_desc':6, # Experimental feature
                              'age_cat':7,
                              'sex':8, 
                              'race':9,  
                              'is_recid':10}

        self.protected_column_names = ["sex", "race"]
        self.protected_column_values = [protected_column_values[0], protected_column_values[1]]
        self.weight_column_name = "instance_weight"

    def get_features(self, row, mean_std_dict, vocab_dict, include_protected_columns):
        
        embd_dim = 32
        features=[]
        
        if include_protected_columns:
            if row[self.feature_names['sex']] == "Female":
                features.append(0)
            else: # This is "Male"
                features.append(1)
            if row[self.feature_names['race']] == "Black":
                features.append(0)
            else: # This includes "White" and "Other"
                features.append(1)
                
        for col_name in self.feature_names :
            numeric_col = col_name != "c_charge_degree" and col_name != "sex" and col_name != "race" and col_name != "c_charge_desc" and col_name!="age_cat" and col_name != "is_recid"

            if numeric_col:
                value = (row[self.feature_names[col_name]] - mean_std_dict[col_name][0])/mean_std_dict[col_name][1]
                features.append(value)
                
        if row[self.feature_names["c_charge_degree"]] == "M":
            features.append(0)
        else:
            features.append(1)
                        
        
        if row[self.feature_names["age_cat"]] == "Greater than 45":
            features.append(0)
        elif row[self.feature_names["age_cat"]] == "Less than 25":
            features.append(1)     
        else:
            features.append(2) # Between 25 to 45
    
#     For experimentation using "charge_desc" natural language feature:
#     charges = []
#     for i in vocab_dict["c_charge_desc"]:
#         charges.append(i)

#         if row[self.feature_names["c_charge_desc"]] in charges:
#             features.append(charges.index(row[self.feature_names["c_charge_desc"]]))     
#         else:
#            features.append(len(charges)+1) # When out of vocabulary, all map to OOV token
        
#         features = tf.convert_to_tensor(features)
#         return features


    def get_data(self, mode="train", include_protected_columns=False):

        if mode == "train":
            data_file = self._train_file
        else:
            data_file = self._test_file

        # Types of CSV columns
        record_defaults = [[0.0], [0.0], [0.0], [0.0], [0.0], ["?"], ["?"], ["?"], ["?"], ["?"], ["?"]]

        # Read in feature data
        csv_data = tf.data.experimental.CsvDataset(data_file, record_defaults)
        
        # Read in means and stds
        mean_std = open(self._mean_std_file, 'r')
        mean_std_dict = json.load(mean_std)

        vocab_dir = open(self._vocab_file, 'r')
        vocab_dict = json.load(vocab_dir)

        num_examples =  len(list(csv_data))
        num_features = len(record_defaults) - 2 # 1 if we are using "charge_desc"            
        if not include_protected_columns:
            num_features = num_features - len(self.protected_column_names)

        compas_features = np.zeros(shape=(num_examples, num_features), dtype="float32")
        compas_targets = np.zeros(shape=(num_examples,1), dtype="float32")
        
        i = 0
        for row in csv_data:
            compas_features[i] = self.get_features(row, mean_std_dict, vocab_dict, include_protected_columns)
            if row[self.feature_names["is_recid"]] == "Yes":
                compas_targets[i]=0.0
            else:
                compas_targets[i]=1.0
            i +=1
        return(compas_features, compas_targets)
