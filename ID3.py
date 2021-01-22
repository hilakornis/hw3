import numpy as np
import pandas as pd

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

            if best_ig < ig:
                best_ig = ig
                best_threshold = threshold
                best_feature = f_index

    return (best_ig, best_feature, best_threshold)


class Node:

    def __init__(self, data, features, parent=None, default_value=0):

        # todo remove the 3 lines below
        self.trues = -1
        if len(data) != 0:
            self.trues = [np.sum(data[:,0])/len(data), len(data), len(features)]

        self.parent = parent
        self.features = features
        self.children = []
        self.h = 0
        self.threshold = None
        self.feature_index = None

        if len(data) == 0 or len(self.features) == 0 :
            self.leaf = True
            self.default_value = default_value

            if len(data) > 0:
                self.h = compute_h(data)

            return

        healthy_leaf = np.sum(data[:,0]) == len(data)
        sick_leaf = np.sum(data[:,0]) == 0

        self.leaf = (healthy_leaf or sick_leaf)
        self.default_value = 1 if np.sum(data[:, 0]) >= (len(data)/2) else 0

        self.h = compute_h(data)

        if not self.leaf:
            self.threshold, self.feature_index = self.split(data)


    def split(self, data):

        if self.leaf:
            return None, None

        ig, feature, threshold = find_best_feature_threshold_per(data, self.features)
        self.features.remove(feature)

        split1 = data[data[:, feature] <= threshold]
        split2 = data[data[:, feature] > threshold]

        self.children = [
            Node(split1, self.features, self, self.default_value),
            Node(split2, self.features, self, self.default_value)
        ]

        return threshold, feature

    def predict(self, data):
        if self.leaf:
            return self.default_value

        # print(data.shape, len(data))

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

    def traverse(self, depth=0):
        print(self, "\n depth: ", depth)
        print()
        for c in self.children:
            c.traverse(depth+1)

    def __str__(self):
        return 'feature: {}\n threshold {}\n leaf {}\n trues {}'.format(self.feature_index, self.threshold, self.leaf, self.trues)

data_train = get_data("train.csv")


num_samples, num_features = data_train.shape
num_features -= 1
set_features = set(range(1,num_features))

tree = Node(data_train, set_features)


print(tree.predict(data_train))
print([int(k) for k in data_train[:,0]])

print()

data_test = get_data("test.csv")
print(tree.predict(data_test))
print([int(k) for k in data_test[:,0]])
