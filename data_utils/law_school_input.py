import tensorflow as tf
import numpy as np
import json

class LawSchoolInput():
    def __init__(self,
               train_file=None,
               test_file=None,
               mean_std_file=None,
               protected_column_values=None):
        
        self._train_file = train_file
        self._test_file = test_file
        self._mean_std_file = mean_std_file
        
        self.protected_column_names = ["sex", "race"]
        self.protected_column_values = [protected_column_values[0], protected_column_values[1]] # Female, Black
        self.weight_column_name = "instance_weight"
        self.column_names = {'sex': 0, 
                           'race': 1,
                           'zfygpa': 2,
                           'zgpa': 3,
                           'ugpa': 4,
                           'DOB_yr': 5,
                           'weighted_lsat_ugpa': 6,
                           'family_income': 7,
                           'cluster_tier': 8,
                           'isPartTime': 9,
                           'lsat': 10,
                           'pass_bar': 11}

    
    def get_features(self, row, mean_std_dict, include_protected_columns):  
        """
        Gets features from csv row.

        Params:
          mode: The row in the csv to parse
          include_protected_columns: Whether to include the protected classes in the data (only True to test fairness)
        Returns:
          tensor of features
        """

        features = []
            
        # Maybe append protected features 
        if include_protected_columns:
            if row[self.column_names['sex']] == "Female":
                features.append(0)
            else: # This is "Male"
                features.append(1)
            if row[self.column_names['race']] == "Black":
                features.append(0)
            else: # This includes "White" and "Other"
                features.append(1)

        # Standardize and append numerical features
        for col_name in self.column_names:
            if col_name != "sex" and col_name != "race" and col_name != "isPartTime" and col_name != "pass_bar":
                value = (row[self.column_names[col_name]] - mean_std_dict[col_name][0])/mean_std_dict[col_name][1]
                features.append(value)
        features.append(row[self.column_names["isPartTime"]])
        features = tf.convert_to_tensor(features)
        return features
        

    def get_data(self, mode="train", include_protected_columns=False):
        """
        Gets train and test data for Law School dataset.

        Params:
          mode: The execution mode: "train" or "test".
          include_protected_columns: Whether to include the protected classes in the data (only True to test fairness)
        Returns:
          train features, test features
        """

        if mode == "train":
            data_file = self._train_file
        else:
            data_file = self._test_file
        
        # Types of CSV columns
        record_defaults = [["?"], ["?"], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]

        # Read in feature data
        csv_data = tf.data.experimental.CsvDataset(data_file, record_defaults)
        
        # Read in means and stds
        mean_std = open(self._mean_std_file, 'r')
        mean_std_dict = json.load(mean_std)
        
        num_examples =  len(list(csv_data))
        num_features = len(record_defaults) - 1            
        if not include_protected_columns:
            num_features = num_features - len(self.protected_column_names)
            
        law_school_features = np.zeros(shape=(num_examples, num_features), dtype="float32")
        law_school_targets = np.zeros(shape=(num_examples,1), dtype="float32")
        i = 0

        for row in csv_data:
            law_school_features[i] = self.get_features(row, mean_std_dict, include_protected_columns)
            law_school_targets[i] = row[self.column_names["pass_bar"]]
            i +=1
        return(law_school_features, law_school_targets)
    
    
    
    
