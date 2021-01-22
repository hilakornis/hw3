import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

def dataframe_to_numpy(df):
    df['diagnosis'] = df['diagnosis'].replace(['M','B'],[0.,1.])
    return df.to_numpy()

def get_data(path="train.csv"):
    data =  pd.read_csv(path)
    return dataframe_to_numpy(data)

def compute_h(data):
    num_samples, num_features = data.shape
    num_features -= 1

    healthy = np.sum(data[:, 0])
    sick = len(data) - healthy
    total = len(data)
    if total == 0 :
        return 0

    h_healthy = healthy / total
    h_sick = sick / total

    healthy_part = h_healthy * np.log2(h_healthy) if h_healthy != 0 else 0
    sick_part = h_sick * np.log2(h_sick) if h_sick != 0 else 0

    h = -(healthy_part + sick_part)

    return h

def compute_IG_for_feature_threhold(data,threshold,feature):

    total = len(data)

    h_self = compute_h(data)

    under_threshold = data[:,feature] <= threshold
    above_threshold = data[:,feature] > threshold

    samples_under = data[under_threshold]
    samples_above = data[above_threshold]

    total_under = len(samples_under)
    h_under = compute_h(samples_under)

    total_above = len(samples_above)
    h_above = compute_h(samples_above)

    ig = h_self - ((total_under/total) *h_under + (total_above/total)*h_above)

    return ig

def find_best_feature_threshold_per(data,features_set):

    num_samples, num_features = data.shape
    num_features -= 1   # num_features shouldn't include index 0 where the label is

    best_feature = list(features_set)[0]
    best_threshold = 0
    best_ig = 0

    #todo I think it needs to move through f's we didn't move through
    # for f_index in range(1,num_features):
    for f_index in features_set:
        data = data[data[:, f_index].argsort()]

        for s_index in range(num_samples - 1):

            threshold = (data[s_index,f_index] + data[s_index + 1,f_index])/2

            ig = compute_IG_for_feature_threhold(data,threshold , f_index)

            if best_ig <= ig:
                best_ig = ig
                best_threshold = threshold
                best_feature = f_index

    return (best_ig, best_feature, best_threshold)


class Node:

    def __init__(self, data, features, parent=None, default_value=0, M=-1):

        # todo remove the 3 lines below

        self.m = M
        self.trues = -1
        if len(data) != 0:
            self.trues = [np.sum(data[:,0])/len(data), len(data), len(features)]

        self.parent = parent
        self.features = features
        self.children = []
        self.h = 0
        self.threshold = None
        self.feature_index = None

        if len(data) == 0:
            self.leaf = True
            self.default_value = default_value

            if len(data) > 0:
                self.h = compute_h(data)

            return

        if len(self.features) == 0:

            self.leaf = True
            self.default_value = 1 if np.sum(data[:, 0]) >= (len(data) / 2) else 0

            if len(data) > 0:
                self.h = compute_h(data)

            return

        healthy_leaf = np.sum(data[:,0]) == len(data)
        sick_leaf = np.sum(data[:,0]) == 0
        self.leaf = (healthy_leaf or sick_leaf)

        self.default_value = 1 if np.sum(data[:, 0]) >= (len(data)/2) else 0

        self.h = compute_h(data)

        if not self.leaf and self.split_condition(data):
            self.threshold, self.feature_index = self.split(data, self.default_value)


    def split(self, data, default_value):

        if self.leaf:
            return None, None

        ig, feature, threshold = find_best_feature_threshold_per(data, self.features)
        self.features.remove(feature)

        split1 = data[data[:, feature] < threshold]
        split2 = data[data[:, feature] >= threshold]

        self.children = [
            Node(split1, self.features, self, default_value, M=self.m),
            Node(split2, self.features, self, default_value, M=self.m)
        ]
        # self.features.append(feature)
        return threshold, feature

    def predict(self, data): #the Node data structure is initialize with the feature set, and arguments it according to it's progression.
        if self.leaf:
            return self.default_value

        if len(data.shape) == 1:
            if data[self.feature_index] <= self.threshold:
                pred = self.children[0].predict(data)
            else:
                pred = self.children[1].predict(data)
            return pred

        predictions = []
        if len(data.shape) > 1:
            for i in range(data.shape[0]):
                d = data[i]
                thresh = d[self.feature_index] <= self.threshold
                if thresh:
                    pred = self.children[0].predict(d)
                else:
                    pred = self.children[1].predict(d)
                predictions.append(pred)

        return predictions

    def split_condition(self, data):
        if len(data) <= self.m and self.m != -1:
            return False
        return True

    def traverse(self, depth=0):
        print(self, "\n depth: ", depth)
        print()
        for c in self.children:
            c.traverse(depth+1)

    def __str__(self):
        return 'feature: {}\n threshold {}\n leaf {}\n trues {}'.format(self.feature_index, self.threshold, self.leaf, self.trues)



def get_k_fold_validation(data_train):
    index_array = [sp for sp in KFold(n_splits=5, random_state=203439989, shuffle=True).split(data_train)]
    validation_folds = [np.concatenate([[data_train[i]] for i in index_array[j][1]]) for j in range(len(index_array))]
    train_folds = [np.concatenate([[data_train[i]] for i in index_array[j][0]]) for j in range(len(index_array))]
    return train_folds, validation_folds


def get_id3_tree_from(data_train , M=-1):
    num_samples, num_features = data_train.shape
    num_features -= 1
    set_features = list(range(1, num_features))
    if M==-1:
        tree = Node(data_train, set_features)
    else:
        tree = Node(data_train, set_features,M)
    return tree


def test_id3_q1(data_train, data_test):
    # data_train = get_data("train.csv")
    # train predictions
    tree = get_id3_tree_from(data_train)

    # data_train = get_data("train.csv")
    # labels = data_train[:,0]
    # predictions = np.array(tree.predict(data_train))
    # print("accuracy on train set", np.sum(labels == predictions)/len(predictions))

    # test_predictions
    # data_test = get_data("test.csv")
    labels = data_test[:, 0]
    predictions = np.array(tree.predict(data_test))
    print("accuracy on test set", np.sum(labels == predictions) / len(predictions))
    print(np.sum(labels == predictions), " predictions were correct, out of: ", len(predictions))

def test_id3_q3(data_train):
    # data_train = get_data("train.csv")
    # train predictions
    train_folds, validation_folds = get_k_fold_validation(data_train)

    m_array = [1, 2 , 3, 4, 5]

    for i in range(5):
        tree = get_id3_tree_from(train_folds[i], 5)
        labels = validation_folds[i][:, 0]
        predictions = np.array(tree.predict(validation_folds[i]))
        print('\ni: ',i, ' m is: ',m_array[i])
        print("accuracy on test set", np.sum(labels == predictions) / len(predictions))
        print(np.sum(labels == predictions), " predictions were correct, out of: ", len(predictions))

    return

data_train = get_data("train.csv")
data_test = get_data("test.csv")
test_id3_q1(data_train,data_test)
# test_id3_q3(data_train)


# index_array= [sp for sp in KFold(n_splits=5, random_state=203439989, shuffle=True).split(data_train)]
# validation_folds = [np.concatenate([[data_train[i]] for i in index_array[j][1]]) for j in range(len(index_array))]
# train_folds = [np.concatenate([[data_train[i]] for i in index_array[j][0]]) for j in range(len(index_array))]
# print(train_folds[-1].shape)
