# https://github.com/ibm321/AI_Examples/tree/master/DecisionTreeId-3
# Ibtisam Ur Rehman 6 Apr 2018 1:23 AM


# If you want to Understand the code then start reading Code from Main method
# It's well Commented


import pandas as pd
import math
import operator


def add_to_tree(entropy_dict, ig_select, tree):
    for i, l in entropy_dict.items():
        if i == ig_select:
            tree[i] = l


# This Function gets IG of all features and returns feature with maximum feature
def calc_max(ig_dict):
    maxim = 0
    for i, j in ig_dict.items():
        j = list(j)
        j = float(j[0])
        if j > maxim:
            maxim = j
            name = i
    return name


# When IG is calculated and feature is selected, it returns childs for that selected feature
def get_child(dataset):
    keys = dataset.value_counts().to_dict()
    for i, j in keys.items():
        keys[i] = {}
    return keys


# This Function deletes the feature that is selected from feature list
def del_feature(features, ig_select):
    delete = 0
    for i in range(len(features)):
        if ig_select == features[i]:
            delete = i
    del features[delete]


# If an attribute has entropy ZERO, this function computes whehter it should be YES OR NO
# Basically it checks whether Items of "YES" or "NO" is greater and returns greater.
def get_target(dataset):
    dataset = dataset.iloc[:,-1]
    counts = dataset.value_counts()
    counts = counts.to_dict()
    counts = max(counts.items(), key=operator.itemgetter(1))[0]
    return counts


# This method returns Entropy of dataset.
def calc_entropy(dataset):
    # Target Column
    target_col = dataset.iloc[:,-1]
    total_size = target_col.size
    # print("total size in ent", total_size)
    values_counts = target_col.value_counts()
    # len_features = len(values_counts)
    tar_options = values_counts.to_dict()
    # print("target_options in ent",tar_options)
    total_entropy = 0
    for key, value in tar_options.items():
        pi = value / total_size
        # print("pi",pi)
        ent = (-pi * (
            math.log(pi, 2)))
        # print("entropy in ent",ent)
        total_entropy = total_entropy + ent
    return total_entropy


# This function Recursively build tree
# First time it receives whole dataset,
# Entropy Dictionary like {'Outlook': {'rain': {0.9709505944546686}, 'sunny': {0.9709505944546686}, 'overcast': {0.0}},
#                          'Temperature': {'mild': {0.9182958340544896}, 'cool': {0.8112781244591328}, 'hot': {1.0}},
#                           'Humidity': {'high': {0.9852281360342516}, 'normal': {0.5916727785823275}},
#                           'Wind': {'weak': {0.8112781244591328}, 'strong': {1.0}}}
# Tree like {'Outlook': {'rain': {}, 'sunny': {}, 'overcast': {}}}
def build_Tree(entropy_dict, dataset, features, ig_dict, tree, ig_select):
    for i, j in tree.items():
        # This Loop iterates tree if we iterate tree above then, in first iteration, i = outlook
        #                                                                  j = {'rain': {}, 'sunny': {}, 'overcast': {}}
        for k, l in j.items():
            # This Loop iterates the j value so that k = rain and l = {}
            # Then we get value of entropy from entropy dictionary
            sub_entropy = list(entropy_dict[i][k])

            sub_entropy = sub_entropy[0]

            # We only get rows of k from dataset like k = rain
            # It returns all rows with sunny in data_subset
            data_subset = dataset[dataset[i].values == k]

            # if entropy of rain is zero then we don't need to iterate and find IG
            if sub_entropy == 0:
                target = get_target(data_subset)
                tree[i][k] = target

            else:
                ig_select = dict()
                ig_dict = dict()

                # Now We have dataset of only rows of rain we passed to it, it returns information gain
                # of all features like outlook = rain and humidity also with all other features.
                # Stores information gain of all features in ig_dict
                calc_info_gain(sub_entropy, features, data_subset, entropy_dict, ig_dict)

                # Then find max in dictionary
                ig_select = calc_max(ig_dict)

                # Delete feature that has selected
                del_feature(features, ig_select)

                # Gen childs like if humidity is selected as sunny root then childs will be like
                # { high: [], normal: {}}
                childs = dict()
                childs = get_child(dataset[ig_select])

                tree_child = dict()

                # It makes childs like {Humidity: { high: [], normal: {}}}
                tree_child[ig_select] = childs

                # Then we call this function recursively and pass tree childs we generated
                tree_temp = build_Tree(entropy_dict,data_subset, features, ig_dict, tree_child, ig_select)

                # When Humidity is totally build to its depth then it updates the original tree
                # It looks like {'Outlook': {'rain': {}, 'sunny': {'Humidity': {'high': 'no', 'normal': 'yes'}}, 'overcast': {}}}
                tree[i][k] = tree_temp
        return tree


# It calculates Information gain and stores IG of each feature in the form:
# {'Outlook': {0.2467498197744391}, 'Temperature': {0.029222565658954647},
# 'Humidity': {0.15183550136234136}, 'Wind': {0.04812703040826927}}
def calc_info_gain(entropy, features, data, entropy_dict, ig_dict):
    ig_var = 0
    for i in features:
        entropy_temp = dict()
#       Count Rows
        attributes = data[i]
        attributes = attributes.value_counts()
        for j, k in attributes.items():
            total_size = len(data)
            data_subset = data[data[i].values == j]
            obt_size = len(data_subset)
            entropy_subset = calc_entropy(data_subset)
            if entropy_subset != 0 or entropy_subset != 1:
                entropy_temp[j] = {entropy_subset}
            ig_var += (obt_size / total_size) * entropy_subset
        total_gain = entropy - ig_var
        ig_var = 0
        entropy_dict[i] = entropy_temp
        ig_dict[i] = {total_gain}


# Then Prints Tree
def myprint(d):
  for k, v in d.items():
      if isinstance(v, dict):
          print(k)
          myprint(v)

      else:
          print("")
          print("{0} : {1}".format(k, v))


def main():
    # Loading Dataset
    # You need your dataset in csv form
    # I am Loading Dataset in Pandas DataFrame ( Pandas DataFrame shows data in tabular form, rows and columns )
    dataset = pd.read_csv('data.csv')
    tree = dict()
    entropy_dict = dict()
    ig_dict = dict()
    childs = dict()

    # We need to iterate through all the features when we calculate IG
    # So we need to maintain a list of features and we removed target attribute from features
    features = list(dataset.columns.values)
    del features[-1]

    # Calculate Entropy of Whole Dataset, Passing Dataframe to function it returns entropy of whole dataset
    entropy = calc_entropy(dataset)

    # Calculate Information Gain of all features (Excluding target attribute)
    # save IG of all features in a dictionary like
    calc_info_gain(entropy, features, dataset, entropy_dict, ig_dict)

    # This method gets the feature with maximum IG store in ig_select
    ig_select = calc_max(ig_dict)

    # Delete Selected feature from list because we don't want to calculate IG again of that feature
    # For Example when we have selected Outlook we don't calculate IG of Outlook when we select Sunny
    del_feature(features, ig_select)

    # Get the childs of Selected Feature
    # If Outlook is selected then childs are Sunny, Overcast and Rain
    childs = get_child(dataset[ig_select])

    # Now add these childs to Tree Dictionary
    tree[ig_select] = childs

    # This Function build Decision Tree and return the Nested Dictionary
    tree = build_Tree(entropy_dict, dataset, features, ig_dict, tree, ig_select)

    # Visualize Nested Dictionary of Tree
    myprint(tree)


if __name__ == '__main__':
    main()
