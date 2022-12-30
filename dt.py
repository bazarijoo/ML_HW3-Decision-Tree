from __future__ import division
import pandas as pd
import numpy as np
import operator
from sklearn.model_selection import train_test_split
import scikit_learn_results

import math

class Node:
    def __init__(self,value, num_samples, num_samples_per_class, predicted_class,criterion='gini'):
        self.criterion = criterion
        self.value=value
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature=None
        self.threshold = 0
        self.left = None
        self.right = None

class dt_classifier:

    def __init__(self,data,max_depth=math.inf):
        self.dataset=data
        self.max_depth=max_depth

    def cal_entropy(self,taret_col):
        elements,counts=np.unique(taret_col,return_counts=True)
        entropy=np.sum(-counts[i]/np.sum(counts) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements)))
        return entropy

    def cal_information_gain(self,data,entropy,splitting_index,target_name='num'):
        partial_entropy_sum=(splitting_index* self.cal_entropy(data.iloc[:splitting_index][target_name])+
                             (len(data)-splitting_index)*self.cal_entropy(data.iloc[splitting_index:][target_name]))/len(data)
        return entropy-partial_entropy_sum

    def cal_gini_index(self,taret_col):
        elements,counts=np.unique(taret_col,return_counts=True)
        gini=1-np.sum((counts[i]/np.sum(counts))**2 for i in range(len(elements)))
        return gini

    def find_best_feature_gini(self,gini_costs):
        best_feature=None
        best_threshold=None
        min_gini=1
        for key,val in gini_costs.items():
            if val[0]<min_gini:
                min_gini=val[0]
                best_threshold=val[1]
                best_feature=key
        return best_feature,best_threshold,min_gini

    def cal_threshold_gini_cost(self,data,split_label,target_col='num'):
        #find unique values and sort according to them
        unique_vals,counts=np.unique(data[split_label],return_counts=True)
        #find best threshold with left side <= that threshhold and right side > threshold
        best_gini=1 #max value for gini
        best_threshhold = unique_vals[0]

        for i in range(len(unique_vals)):
            threshhold = unique_vals[i]

            left=data.where(data[split_label]<=threshhold).dropna()[target_col]
            right=data.where(data[split_label]>threshhold).dropna()[target_col]
            gini=(len(left)*self.cal_gini_index(left)+len(right)*self.cal_gini_index(right))/len(data)

            if gini<best_gini:
                best_gini=gini
                best_threshhold=threshhold

        return best_gini,best_threshhold


    def CART(self,data,attributes,target_attribute='num',depth=0):

        if len(np.unique(data[target_attribute]))<=1:
            num_samples_per_class=[]
            target_class=np.int(np.unique(data[target_attribute])[0])
            if target_class==0:
                num_samples_per_class.append(len(data))
                num_samples_per_class.append(0)
            else:
                num_samples_per_class.append(0)
                num_samples_per_class.append(len(data))

            return Node(value=0, num_samples=len(data), num_samples_per_class=num_samples_per_class,
                        predicted_class=target_class)

        elif len(data)==0 : #return leaf node with most occuring target label
            return

        elif len(attributes)==0:#return most common label of that class
            vals, counts = np.unique(data[target_attribute], return_counts=True)
            num_samples_per_class = [count for count in counts]

            return Node(value=self.cal_gini_index(data[target_attribute]), num_samples=len(data), num_samples_per_class=num_samples_per_class,
                        predicted_class=np.unique(data[target_attribute])[
                            np.argmax(np.unique(data[target_attribute], return_counts=True)[1])])
        else:
                gini_costs = {attribute: self.cal_threshold_gini_cost(data, attribute, target_attribute) for attribute in attributes}
                best_feature,best_threshold,min_gini = self.find_best_feature_gini(gini_costs)
                attributes.remove(best_feature)      #remove current separating feature from set of features in the next runs

                vals,counts=np.unique(data[target_attribute],return_counts=True)

                num_samples_per_class= [count for count in counts]
                predicted_class=np.int(vals[np.argmax(counts)])

                node=Node(value=min_gini, num_samples=len(data), num_samples_per_class=num_samples_per_class, predicted_class=predicted_class)
                node.threshold = best_threshold
                node.feature = best_feature

                if depth < self.max_depth:
                    # separating data based on best feature and threshold
                    data.sort_values(by=best_feature)
                    left_data = data.where(data[best_feature] <= best_threshold).dropna()
                    right_data = data.where(data[best_feature] > best_threshold).dropna()

                    node.left=self.CART(left_data,attributes.copy(),depth=depth+1)
                    node.right=self.CART(right_data,attributes.copy(),depth=depth+1)

                return node

    def find_best_feature_entropy(self,data,attributes,entropy):
        attribute_threshold=dict()

        for attribute in attributes:
            sorted_data=data.sort_values(by=attribute).reset_index(drop=True)
            index_gain_dict=dict()
            for i in  range(1,len(sorted_data)):   # target labels of data
                if sorted_data['num'][i-1] != sorted_data['num'][i]:
                    info_gain=self.cal_information_gain(sorted_data,entropy,i)
                    index_gain_dict[i]=info_gain

            #caculating maximum information gain of that attribute and correspoint separating index
            split_index,info_gain=max(index_gain_dict.items(),key=operator.itemgetter(1))
            threshold=(sorted_data[attribute][split_index]+sorted_data[attribute][split_index-1])/2
            attribute_threshold[attribute,threshold]=info_gain
        #return best attribute with best threshold
        best_feature=max(attribute_threshold.items(),key=operator.itemgetter(1))[0]
        feature,threshold=best_feature[0],best_feature[1]
        return feature,threshold


    def ID3(self, data, attributes, target_attribute='num', depth=0):
        if len(np.unique(data[target_attribute])) <= 1:
            num_samples_per_class = []
            target_class = np.int(np.unique(data[target_attribute])[0])
            if target_class == 0:
                num_samples_per_class.append(len(data))
                num_samples_per_class.append(0)
            else:
                num_samples_per_class.append(0)
                num_samples_per_class.append(len(data))

            return Node(value=0, num_samples=len(data), num_samples_per_class=num_samples_per_class,
                        predicted_class=target_class)

        elif len(data) == 0:  # return leaf node with most occuring target label
            return

        elif len(attributes) == 0:  # return most common label of that class
            vals, counts = np.unique(data[target_attribute], return_counts=True)
            num_samples_per_class = [count for count in counts]

            return Node(value=self.cal_gini_index(data[target_attribute]), num_samples=len(data),
                        num_samples_per_class=num_samples_per_class,
                        predicted_class=np.unique(data[target_attribute])[
                            np.argmax(np.unique(data[target_attribute], return_counts=True)[1])])

        else:
            entropy=self.cal_entropy(data['num'])
            best_feature, best_threshold =self.find_best_feature_entropy(data, attributes,entropy)

            vals, counts = np.unique(data[target_attribute], return_counts=True)

            num_samples_per_class = [count for count in counts]
            predicted_class = np.int(vals[np.argmax(counts)])

            node = Node(value=entropy, num_samples=len(data), num_samples_per_class=num_samples_per_class,
                        predicted_class=predicted_class)
            node.threshold = best_threshold
            node.feature = best_feature

            if depth < self.max_depth:
                # separating data based on best feature and threshold
                data.sort_values(by=best_feature)
                left_data = data.where(data[best_feature] <= best_threshold).dropna()
                right_data = data.where(data[best_feature] > best_threshold).dropna()

                node.left = self.CART(left_data, attributes.copy(), depth=depth + 1)
                node.right = self.CART(right_data, attributes.copy(), depth=depth + 1)

            return node

    def train(self,criterion='gini'):
        # implement cross validation
        attributes=set(self.dataset.columns[:-1])

        if criterion=='entropy':
            self.tree = self.ID3(self.dataset, attributes)
        else:
            self.tree=self.CART(self.dataset,attributes)

    def predict(self,data):
        node=self.tree
        while node.left:
            if data[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.predicted_class

    def predict_class(self,test_data):
        predicted_class=[]
        for _,data in test_data.iterrows():
            predicted_class.append(self.predict(data))

        return predicted_class

    def compute_accuracy(self,predicted_class,actual_vals):
        errors = []

        for i in range(len(predicted_class)):
            if predicted_class[i] != actual_vals[i]:
                errors.append(predicted_class[i])

        return 1 - len(errors) / len(predicted_class)


    def split_cross_validation(self):
        pass




if __name__ == '__main__' :
    data = pd.read_csv("processed.cleveland.data", header=None,
                        names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs','restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal','num'], na_values='?') #(num is the predicted attribute)

    # handling missing values
    data['thal'] = data['thal'].fillna(value=data['thal'].median())
    data['ca'] = data['ca'].fillna(value=data['ca'].median())
    #create binary classification
    data.loc[data['num']>0,'num']=1

    training_set, test_set = train_test_split(data, test_size=0.2,random_state=5)

    #test with gini index
    dt=dt_classifier(training_set,max_depth=3)
    dt.train()
    predicted_class=dt.predict_class(test_set)
    gini_accuracy=dt.compute_accuracy(predicted_class,list(test_set['num']))

    # test with entropy
    dt=dt_classifier(training_set,max_depth=3)
    dt.train()
    predicted_class=dt.predict_class(test_set)
    entropy_accuracy=dt.compute_accuracy(predicted_class,list(test_set['num']))

    print('Manual  Acuracy  with gini index criteria : ', "{0:.0%}".format(gini_accuracy))
    print('Manual  Acuracy  with entropy criteria : ', "{0:.0%}".format(entropy_accuracy))
    print()

    # test with scikit learn
    scikit_learn_results.estimate_with_scikitlearn(data,criterion='gini')
    scikit_learn_results.estimate_with_scikitlearn(data,criterion='entropy')





