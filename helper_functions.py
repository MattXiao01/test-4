# -*- coding: utf-8 -*-
"""
This file contains all the helper functions needed in poi_id.py
"""
from feature_format import featureFormat, targetFeatureSplit
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn import metrics
from pprint import pprint
import numpy as np
from sklearn import preprocessing

def scatterPlot(data_dict, features_list, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ 
    scatter plot of features selected
    mark the poi to make sure the outliers in the scatter plot are not pois
    """
    data = featureFormat(data_dict, features_list)
    poi, finance_features = targetFeatureSplit(data)
    for i, p in enumerate(poi):
        non_poi = plt.scatter(finance_features[i][0], finance_features[i][1])
        if p:
            poi = plt.scatter(finance_features[i][0], finance_features[i][1], color="r", marker="*")            
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.legend((non_poi, poi), ('Non POI', 'POI'),loc='upper left')
    plt.title('scatter plot of 2 features')
    plt.savefig(name)
    plt.show()
    
    
def checkNaN(data_dict):
    """
    check points with all features (except poi) 'NaN'  
    These points have no information contained and should be removed as outlier 
    """
    for k, v in data_dict.iteritems():
        mark = True
        for feature, value in v.iteritems():
            if (value != 'NaN') and (feature != 'poi'):
                mark = False
                break
        if mark:
            print k
            print v['poi']
            
            
def feature_NaN(data_dict):
    """
    return the number of missing values for all the features as well as 
    valid fraction for POIs and non-POIs
    """
    N_poi = 0
    count_NaN = dict.fromkeys(data_dict.itervalues().next().keys(), {})
    for k in count_NaN:
        count_NaN[k] = {'count': 0, 'valid count POI': 0, 'valid count non-POI': 0, \
                        'valid fraction for POI': 0.0, 'valid fraction for non-POI': 0.0}
    for k, v in data_dict.iteritems():
        mark = v['poi']
        if mark: N_poi += 1
        for feature, value in v.iteritems():   
            if value == 'NaN':
                count_NaN[feature]['count'] += 1
            else:
                if mark:
                    count_NaN[feature]['valid count POI'] += 1
                else:
                    count_NaN[feature]['valid count non-POI'] += 1
                    
    N = len(data_dict)
    for k in count_NaN:
        count_NaN[k]['valid fraction for POI'] = 1.0 * count_NaN[k]['valid count POI']/N_poi
        count_NaN[k]['valid fraction for non-POI'] = 1.0 * count_NaN[k]['valid count non-POI']/(N-N_poi)
    pprint(count_NaN)
    return count_NaN

    
#    df = pd.DataFrame.from_records(list(data_dict.values()))
#    df.replace(to_replace='NaN', value=numpy.nan, inplace=True)
#    print df.isnull().sum()
#    grouped = df.groupby(df['poi'])
#    grouped.isnull().sum()
            

def get_k_best(data_dict, features_list, k):
    """
    get the best K features by SelectKBest modules
    """
    
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)  

    k_best = SelectKBest(k=k).fit(features, labels)
    scores = k_best.scores_
    pairs = zip(scores, features_list[1:])
    pairs.sort(reverse = True)
    pairs_sorted = [(v2,v1) for v1,v2 in pairs]
    k_best_features = dict(pairs_sorted[:k])
    pprint(pairs_sorted)
    return k_best_features
    
    
    
def add_financial_aggregate(data_dict, features_list):
    """ 
    add a new feature 'financial_aggregate' to the dataset and feature_list    
    """
    fields = ['total_stock_value', 'exercised_stock_options', 'total_payments']
    for name in data_dict:
        person = data_dict[name]
        is_valid = True
        for field in fields:
            if person[field] == 'NaN':
                is_valid = False
        if is_valid:
            person['financial_aggregate'] = sum([person[field] for field in fields])
        else:
            person['financial_aggregate'] = 'NaN'
    features_list += ['financial_aggregate']
    
    
    
    
def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
    """
    fraction = 0.

    if (poi_messages != 'NaN') and (all_messages != 'NaN') and (all_messages != 0):
        fraction = 1.0 * poi_messages/all_messages

    return fraction
    

def add_poi_feature(my_dataset, features_list):
    """
    add new features 'poi_ratio', 'fraction_from_poi' and 'fraction_to_poi'
    to the dataset and features_list.
    """    
    for name in my_dataset:
        data_point = my_dataset[name]
    
        from_poi_to_this_person = data_point["from_poi_to_this_person"]
        to_messages = data_point["to_messages"]
        fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
        data_point["fraction_from_poi"] = fraction_from_poi
    
        from_this_person_to_poi = data_point["from_this_person_to_poi"]
        from_messages = data_point["from_messages"]
        fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
        data_point["fraction_to_poi"] = fraction_to_poi
        
        if from_poi_to_this_person == 'NaN' or from_this_person_to_poi == 'NaN' \
                                        or to_messages == 'NaN' or from_messages == 'NaN':
            data_point["poi_ratio"] = 0
        else:
            data_point["poi_ratio"] = computeFraction( from_poi_to_this_person + from_this_person_to_poi, \
                                                        to_messages + from_messages)
        
    features_list += ["fraction_from_poi", "fraction_to_poi", "poi_ratio"]
    
    
    
def feature_scale(data_dict, features_list):
    """ 
    scale features using MinMaxScalar and update the data_dict
    """
    for feature in features_list:
        tmp_list = []
        if feature == 'poi': 
            continue
        else:
            for name in data_dict:
                value = data_dict[name][feature]
                if value == 'NaN':
                    value = 0
                    data_dict[name][feature] = 0
                tmp_list.append( [float(value)] )
            
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(np.array(tmp_list))
            
            for name in data_dict:
                data_dict[name][feature] = scaler.transform([float(data_dict[name][feature])])[0]

        
    
    
def scoring(estimator, features_test, labels_test):
    """
    score used in GridSearchCV to find out the best esitmator
    """
    pred = estimator.predict(features_test)
    p = metrics.precision_score(labels_test, pred, average='micro')
    r = metrics.recall_score(labels_test, pred, average='micro')
    if p > 0.3 and r > 0.3:
        return metrics.f1_score(labels_test, pred, average='macro')
    return 0
    
    