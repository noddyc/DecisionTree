import re
import pandas as pd
import numpy as np
import math
import random
import sys
import os
random.seed(630)
"""
file: wiki.py
Lab 2
Author: Jian He
"""
##############################
## arguments

# ensure there are 4 arguments
model_type = None
Predict_text = None
Dutch_train = None
Italian_train = None


if(len(sys.argv) != 4):
    print("Usage - python3 wiki.py train")
    print("Usage - python3 wiki.py predict model-type datafile")
    sys.exit(0)

if(len(sys.argv) == 4):
    if sys.argv[1] == "predict":
        model_type = sys.argv[2]
        Predict_text = os.path.abspath(sys.argv[3])
    else:
        Dutch_train = os.path.abspath(sys.argv[2])
        Italian_train = os.path.abspath(sys.argv[3])

##############################
## Implementation of Decision Tree
def entropy_calculation(data, smaller_row_lst, larger_row_lst):
    """
    This calcualates the entropy of subset
    that contains labels that are smaller than the selected threshold
    and lables are equal or larger than the selected threshold
    :param data: data to be classified
    :param smaller_row_lst:
    :param larger_row_lst:
    :return: the entropy
    """
    total_smaller= len(smaller_row_lst)
    total_larger = len(larger_row_lst)
    class_Dutch_sm = 0
    class_Italian_sm = 0
    class_Dutch_lg = 0
    class_Italian_lg = 0
    for ind in smaller_row_lst:
        if data["Class"][ind] == "Dutch":
            class_Dutch_sm += 1
        else:
            class_Italian_sm += 1
    for ind in larger_row_lst:
        if data["Class"][ind] == "Dutch":
            class_Dutch_lg += 1
        else:
            class_Italian_lg += 1
    a = total_smaller / (total_smaller + total_larger)
    b = total_larger / (total_smaller + total_larger)
    f1 = class_Dutch_sm / total_smaller
    f2 = class_Italian_sm / total_smaller
    f3 = class_Dutch_lg / total_larger
    f4 = class_Italian_lg / total_larger
    if(f1 == 0):
        f1 = 1
    if (f2 == 0):
        f2 = 1
    if(f3 == 0):
        f3 = 1
    if(f4 == 0):
        f4 = 1
    weighted_entropy = -(a*(-f1*math.log2(f1)-f2*math.log2(f2)))-(b*(-f3*math.log2(f3)-f4*math.log2(f4)))
    return weighted_entropy

def best_split(data, label_lst):
    """
    This iterates through all the combination of attribtutes and
    threshold and evaluate them with entropy
    :param data: data to be classified
    :param label_lst: list of attributes to be tested
    :return: best threshold and best attribute to split on
    and subsets after split
    """
    best_attribute = None
    best_threshold = None
    best_entropy = -99
    for label in label_lst:
        col_min = int(data[label].min())
        col_max = int(data[label].max())
        larger_lst = []
        smaller_lst = []
        larger_row_lst = []
        smaller_row_lst = []
        for num in range(col_min+1, col_max):
            for ind in data.index:
                if (data[label][ind] < num).all():
                    smaller_lst.append(data[label][ind])
                    smaller_row_lst.append(ind)
                else:
                    larger_lst.append(data[label][ind])
                    larger_row_lst.append(ind)
            weighted_entropy = entropy_calculation(data, smaller_row_lst, larger_row_lst)
            if weighted_entropy > best_entropy:
                best_entropy = weighted_entropy
                best_attribute = label
                best_threshold = num
            larger_lst = []
            smaller_lst = []
    df_sm = data[data[best_attribute] < best_threshold]
    df_lg = data[data[best_attribute] >= best_threshold]
    return df_sm, df_lg, best_attribute, best_threshold


class Node:
    """
    This is data structure of a node, it has children of 'left' and 'right'.
    It is used to store the decision tree
    """
    left = right = None

    def __init__(self, data, attribute = None, threshold = None,
                 sign = None, parent = "", tag = None):
        """
        Initial function of node
        :param data: data stored in this node
        :param attribute: attribute split on this node, none if it is root
        :param threshold: threshold split on this node, none if it is root
        :param sign: larger or smaller than threshold, none if it is root
        :param parent: parent path to this node, none if it is root
        :param tag: the result of classification on this node, none if it is root
        """
        self.data = data
        self.attribute = attribute
        self.threshold = threshold
        self.sign = sign
        self.parent = parent
        self.tag = tag


def inorder(root, lst):
    """
    Traseverse all the leaf node
    :param root: root to be traseverse
    :param lst: list storing all the leaf nodes
    :return: list of all the leaf nodes
    """
    if root is None:
        return
    inorder(root.left, lst)
    if root.left is None and root.right is None:
        lst.append(root)
    inorder(root.right, lst)
    return lst


def tree(root, labels, tree_depth):
    """
    Create a binary split decision tree
    :param root: root to be inserted
    :param labels: attributes that can be used for split
    :return: a binary split decision tree
    """
    cur = root
    depth = 0
    while True:
        all_leaf = inorder(cur, [])
        if depth > tree_depth:
            break
        for leaf in all_leaf:
            if len(leaf.data) < 9:
                continue
            if (leaf.data["Class"] == "Italian").sum()/len(leaf.data) > 0.95 or\
                    (leaf.data["Class"] == "Dutch").sum()/len(leaf.data) > 0.95:
                continue
            temp = best_split(leaf.data, labels)
            tagLeft = (majority_tag(temp[0]))
            tagRight = (majority_tag(temp[1]))
            leaf.left = Node(temp[0], temp[2], temp[3], "<",
                          leaf.parent + "\nif (data['{}'][ind] {} {}):".format(leaf.attribute, leaf.sign, leaf.threshold), tagLeft)
            leaf.right = Node(temp[1], temp[2], temp[3], ">=",
                           leaf.parent + "\nif (data['{}'][ind] {} {}):".format(leaf.attribute, leaf.sign, leaf.threshold), tagRight)
        depth += 1
    return cur

def majority_tag(data):
    """
    It returns the major label of attribute 'Class' in the data
    :param data: data
    :return: the major labelf of attribute 'Class' in the data
    """
    if (data["Class"] == "Italian").sum() > (data["Class"] == "Dutch").sum():
        return "Italian"
    else:
        return "Dutch"

def add_tab(txt):
    """
    It adds correct number of indent on each line of the code
    :param txt: text going to be used as classifier code
    :return: text ready to be copied as python code
    """
    count = 0
    result =""
    for i in txt.split("\n"):
        result =  result + "\t"*count + i + "\n"
        count += 1
    result = result + "\t"*(count-1) + "continue"
    return result

def text_conversion(tree_result):
    """
    It converts the tree into text python code
    :param tree_result: tree
    :return: text that is as python code
    """
    result = ""
    for i in tree_result:
        formattext = ""
        for txt in i.parent.split("\n"):
            formattext = formattext + txt + "\n"
        formattext = formattext + "if (data['{}'][ind]{}{}):\n".format(i.attribute, i.sign, i.threshold)
        formattext = formattext + "result.append('{}')".format(i.tag)
        find_index = formattext.find(":")
        formattext = formattext[find_index + 1:]
        cur_text = add_tab(formattext)
        result = result + cur_text
    return result


def confusion_matrix(data, lst):
    """
    It generates four numbers for calculating accuracy
    :param data: data to be classified
    :param lst: empty list
    :return: four numbers: number of Dutch classified as Dutch,
    number of Dutch classified as Italian,
    number of Italian classified as Italian,
    number of Italian classified as Dutch
    """
    A_as_A =0
    A_as_B =0
    B_as_B =0
    B_as_A =0
    for i in range(0,len(lst)):
        if data['Class'].iloc[i] == 'Dutch' and lst[i] == 'Dutch':
            A_as_A +=1
            continue
        if data['Class'].iloc[i] == 'Dutch' and lst[i] == 'Italian':
            A_as_B +=1
            continue
        if data['Class'].iloc[i] == 'Italian' and lst[i] == 'Italian':
            B_as_B +=1
            continue
        if data['Class'].iloc[i] == 'Italian' and lst[i] == 'Dutch':
            B_as_A +=1
            continue
    return [A_as_A, A_as_B, B_as_B, B_as_A]

##############################
## Data Cleaning for decision tree

# These are two most frequent word list
Dutch_WordList = ["de", "van", "een", "en", "het", "is", "op","te", "met"]
Italian_WordList = ["di", "e", "il", "la", "che", "a", "per", "un", "del"]


def divide_portion(lst, length):
    """
    This function is used to divide list into list of list with certain length
    :param lst: original list
    :param length: length desired
    :return:
    """
    for i in range(0, len(lst), length):
        yield lst[i:i + length]

def classifier_depth1(data):
    """
    Decision tree classifier of depth limit 1
    :param data: data
    :return: predicted label
    """
    result = []
    for ind in data.index:
        if (data['diI'][ind] < 1):
            if (data['eI'][ind] < 1):
                result.append('Dutch')
                continue
        if (data['diI'][ind] < 1):
            if (data['eI'][ind] >= 1):
                result.append('Italian')
                continue
        if (data['diI'][ind] >= 1):
            result.append('Italian')
            continue
    return result

def classifier_depth2(data):
    """
    Decision tree classifier of depth limit 2
    :param data: data
    :return: predicted label
    """
    result = []
    for ind in data.index:
        if (data['diI'][ind] < 1):
            if (data['eI'][ind] < 1):
                if (data['laI'][ind] < 1):
                    result.append('Dutch')
                    continue
        if (data['diI'][ind] < 1):
            if (data['eI'][ind] < 1):
                if (data['laI'][ind] >= 1):
                    result.append('Italian')
                    continue
        if (data['diI'][ind] < 1):
            if (data['eI'][ind] >= 1):
                result.append('Italian')
                continue
        if (data['diI'][ind] >= 1):
            result.append('Italian')
            continue
    return result

def classifier_depth4(data):
    """
    Decision tree classifier of depth limit 4
    :param data: data
    :return: predicted label
    """
    result = []
    for ind in data.index:
        if (data['diI'][ind] < 1):
            if (data['eI'][ind] < 1):
                if (data['laI'][ind] < 1):
                    if (data['ilI'][ind] < 1):
                        result.append('Dutch')
                        continue
        if (data['diI'][ind] < 1):
            if (data['eI'][ind] < 1):
                if (data['laI'][ind] < 1):
                    if (data['ilI'][ind] >= 1):
                        result.append('Italian')
                        continue
        if (data['diI'][ind] < 1):
            if (data['eI'][ind] < 1):
                if (data['laI'][ind] >= 1):
                    result.append('Italian')
                    continue
        if (data['diI'][ind] < 1):
            if (data['eI'][ind] >= 1):
                result.append('Italian')
                continue
        if (data['diI'][ind] >= 1):
            result.append('Italian')
            continue
    return result


##############################
## adaboost
def classifier_stump1(data):
    """
    Hard coded First Stump
    :param data: data to classify
    :return: result after classify
    """
    result = []
    for ind in data.index:
        if (data['eI'][ind] < 1):
            if (data['diI'][ind] < 1):
                result.append('Dutch')
                continue
        if (data['eI'][ind] < 1):
            if (data['diI'][ind] >= 1):
                result.append('Italian')
                continue
        if (data['eI'][ind] >= 1):
            result.append('Italian')
            continue
    return result


def classifier_stump2(data):
    """
    Hard coded Second Stump
    :param data: data to classify
    :return: result after classify
    """
    result = []
    for ind in data.index:
        if (data['diI'][ind] < 1):
            if (data['ilI'][ind] < 1):
                result.append('Dutch')
                continue
        if (data['diI'][ind] < 1):
            if (data['ilI'][ind] >= 1):
                result.append('Italian')
                continue
        if (data['diI'][ind] >= 1):
            result.append('Italian')
            continue
    return result


def classifier_stump3(data):
    """
    Hard coded Third Stump
    :param data: data to classify
    :return: result after classify
    """
    result = []
    for ind in data.index:
        if (data['eI'][ind] < 1):
            result.append('Dutch')
            continue
        if (data['eI'][ind] >= 1):
            result.append('Italian')
            continue
    return result


def error_rate(result_stump, data, sample_weight):
    """
    Calculate performance of the stump and data misclassified
    :param result_stump: the stump used for evaluation
    :param data: data used for classify
    :param sample_weight: sample weight of data point
    :return: performance and data misclassified
    """
    misclassified = []
    total_error = 0
    for i in range(0,len(data.index)):
        if result_stump[i] != data["Class"].iloc[i]:
            misclassified.append(i)
    for j in misclassified:
        total_error += sample_weight[j]
    performance = 1/2 * math.log((1-total_error)/total_error)
    return performance, misclassified


def adaboost(data):
    """
    Function to create three stumps for adaboost
    :param data: train data used for adaboost
    :return: three stumps
    """
    stumps = []
    train_ada = data
    train_adaCol = train_ada.columns.tolist()
    train_adaCol = train_adaCol[:-1]
    root_ada = Node(train_ada, None, None, None, "", None)
    stump1 = inorder(tree(root_ada, train_adaCol, 1), [])
    txt_stump1 = text_conversion(stump1)
    stumps.append(txt_stump1)

    sample_weight = []
    for i in range(0, len(train_ada.index)):
        sample_weight.append(1 / len(train_ada.index))
    result_stump1 = classifier_stump1(train_ada)

    error_result = error_rate(result_stump1, train_ada, sample_weight)
    performance = error_result[0]
    misclassified = error_result[1]

    for j in range(0, len(sample_weight)):
        if j in misclassified:
            false_updated_weight = sample_weight[j] * math.exp(performance)
            sample_weight[j] = false_updated_weight
        else:
            true_updated_weight = sample_weight[j] * math.exp(-performance)
            sample_weight[j] = true_updated_weight

    cumulative_weight = []
    sum_weight = sum(sample_weight)
    for i in range(0, len(sample_weight)):
        sample_weight[i] = sample_weight[i] / sum_weight
    for i in range(0, len(sample_weight)):
        cumulative_weight.append(sum(sample_weight[:i + 1]))

    temp = []
    for i in range(0, len(train_ada.index)):
        num = random.random()
        for j in range(0, len(cumulative_weight)):
            if num <= cumulative_weight[j]:
                new_data = train_ada.iloc[j]
                temp.append(new_data.values.tolist())
                break

    temp = pd.DataFrame(temp, columns=train_ada.columns)
    root_ada = Node(temp, None, None, None, "", None)
    stump2 = inorder(tree(root_ada, train_adaCol, 1), [])
    txt_stump2 = text_conversion(stump2)
    stumps.append(txt_stump2)
    result_stump2 = classifier_stump2(temp)

    sample_weight = []
    for i in range(0, len(train_ada.index)):
        sample_weight.append(1 / len(train_ada.index))

    error_result = error_rate(result_stump2, temp, sample_weight)
    performance = error_result[0]
    misclassified = error_result[1]

    for j in range(0, len(sample_weight)):
        if j in misclassified:
            false_updated_weight = sample_weight[j] * math.exp(performance)
            sample_weight[j] = false_updated_weight
        else:
            true_updated_weight = sample_weight[j] * math.exp(-performance)
            sample_weight[j] = true_updated_weight

    cumulative_weight = []
    sum_weight = sum(sample_weight)
    for i in range(0, len(sample_weight)):
        sample_weight[i] = sample_weight[i] / sum_weight
    for i in range(0, len(sample_weight)):
        cumulative_weight.append(sum(sample_weight[:i + 1]))

    temp = []
    for i in range(0, len(train_ada.index)):
        num = random.random()
        for j in range(0, len(cumulative_weight)):
            if num <= cumulative_weight[j]:
                new_data = train_ada.iloc[j]
                temp.append(new_data.values.tolist())
                break

    temp = pd.DataFrame(temp, columns=train_ada.columns)
    root_ada = Node(temp, None, None, None, "", None)
    stump3 = inorder(tree(root_ada, train_adaCol, 0), [])
    txt_stump3 = text_conversion(stump3)
    stumps.append(txt_stump3)
    return stumps


def ada_result(data):
    """
    Generate classification results of 3 stumps with
    majority vote
    :param data:
    :return: classification result of adaboost
    """
    result1 = classifier_stump1(data)
    result2 = classifier_stump2(data)
    result3 = classifier_stump3(data)
    final_result = []
    for i in range(0, len(data.index)):
        temp = []
        temp.append(result1[i])
        temp.append(result2[i])
        temp.append(result3[i])
        if temp.count("Dutch") > temp.count("Italian"):
            final_result.append("Dutch")
        else:
            final_result.append("Italian")
    return confusion_matrix(data,final_result)


##############################
## Decision Tree
if Dutch_train is not None and Italian_train is not None:
    ## Cleaning of Dutch words training sample
    DeWordLst = []
    DeWordLst_cleaned = []

    with open(Dutch_train,'r') as file:
        for line in file:
            for word in line.split():
                DeWordLst.append(word)

    for i in DeWordLst:
        i = re.sub('[0-9]+', '', i)
        i = re.sub('[^\w\s]',"", i)
        i = re.sub(' ', '', i)
        i = i.lower()
        DeWordLst_cleaned.append(i)

    while("" in DeWordLst_cleaned) :
        DeWordLst_cleaned.remove("")

    DeWordLst_df = []

    for i in divide_portion(DeWordLst_cleaned, 20):
        temp = []
        temp.append(' '.join(i))
        DeWordLst_df.append(temp)

    for i in DeWordLst_df:
        for j in Dutch_WordList:
            count = i[0].split().count(j)
            i.append(count)
        for k in Italian_WordList:
            count = i[0].split().count(k)
            i.append(count)
        i.append("Dutch")

    DeWordLst_df = DeWordLst_df[:-1]

    ## Cleaning of Italian words training sample
    ItWordLst = []
    ItWordLst_cleaned = []

    with open(Italian_train,'r') as file:
        for line in file:
            for word in line.split():
                ItWordLst.append(word)

    for i in ItWordLst:
        i = re.sub('[0-9]+', '', i)
        i = re.sub('[^\w\s]',"", i)
        i = re.sub(' ', '', i)
        i = i.lower()
        ItWordLst_cleaned.append(i)

    while("" in ItWordLst_cleaned) :
        ItWordLst_cleaned.remove("")

    ItWordLst_df = []

    for i in divide_portion(ItWordLst_cleaned, 20):
        temp = []
        temp.append(' '.join(i))
        ItWordLst_df.append(temp)

    for i in ItWordLst_df:
        for j in Dutch_WordList:
            count = i[0].split().count(j)
            i.append(count)
        for k in Italian_WordList:
            count = i[0].split().count(k)
            i.append(count)
        i.append("Italian")

    ItWordLst_df = ItWordLst_df[:-1]

    ## Combine two wordlist into a dataframe
    data = ItWordLst_df + DeWordLst_df
    df = pd.DataFrame(data, columns = ["words", "deD", "vanD", "eenD", "enD", "hetD",
                                       "isD", "opD", "teD", "metD", "diI", "eI",
                                       "ilI", "laI", "cheI", "aI", "perI", "unI",
                                       "delI", "Class"])

    df = df.drop(columns=["words"])

    ## Split data into 6:2:2 for train, validate and test set
    train, validate, test = np.split(df.sample(frac=1, random_state=630),
                           [int(.6*len(df)), int(.8*len(df))])
    ##############################
    ## Train Decision Tree

    #  Obtain attributes for decision tree
    TrainCol = train.columns.tolist()
    # exclude last column because it is the target column
    TrainCol = TrainCol[:-1]
    parameters_depth = [1,2,4,6,8]
    for i in parameters_depth:
        # create an empty root node
        root = Node(validate, None, None, None, "", None)
        # create a tree with depth 8
        Tree = inorder(tree(root, TrainCol, i), [])
        # convert the tree to text for classifier code
        txt = text_conversion(Tree)

    print("Training:")
    print("For parameter tuning of Decision Tree")
    dt_1 = confusion_matrix(validate,classifier_depth1(validate))
    print("##############################")
    print("From the decision tree with depth 1, the result on validate set is: ")
    print("{} Dutch was classified as Dutch".format(dt_1[0]))
    print("{} Dutch was classified as Italian".format(dt_1[1]))
    print("{} Italian was classified as Italian".format(dt_1[2]))
    print("{} Italian was classified as Dutch".format(dt_1[3]))
    print("The accuracy is {}".format((dt_1[0] + dt_1[2]) / sum(dt_1)))

    dt_2 = confusion_matrix(validate, classifier_depth2(validate))
    print("##############################")
    print("From the decision tree with depth 2, the result on validate set is: ")
    print("{} Dutch was classified as Dutch".format(dt_2[0]))
    print("{} Dutch was classified as Italian".format(dt_2[1]))
    print("{} Italian was classified as Italian".format(dt_2[2]))
    print("{} Italian was classified as Dutch".format(dt_2[3]))
    print("The accuracy is {}".format((dt_2[0] + dt_2[2]) / sum(dt_2)))

    dt_4 = confusion_matrix(validate, classifier_depth4(validate))
    print("##############################")
    print("From the decision tree with depth 4, the result on validate set is: ")
    print("{} Dutch was classified as Dutch".format(dt_4[0]))
    print("{} Dutch was classified as Italian".format(dt_4[1]))
    print("{} Italian was classified as Italian".format(dt_4[2]))
    print("{} Italian was classified as Dutch".format(dt_4[3]))
    print("The accuracy is {}".format((dt_4[0] + dt_4[2]) / sum(dt_4)))

    print("##############################")
    dt_4_test = confusion_matrix(test, classifier_depth4(test))
    print("The best decision tree is depth of 4")
    print("From the decision tree with depth 4, the result on test set is: ")
    print("{} Dutch was classified as Dutch".format(dt_4_test[0]))
    print("{} Dutch was classified as Italian".format(dt_4_test[1]))
    print("{} Italian was classified as Italian".format(dt_4_test[2]))
    print("{} Italian was classified as Dutch".format(dt_4_test[3]))
    print("The accuracy is {}".format((dt_4_test[0] + dt_4_test[2]) / sum(dt_4_test)))

    ada_matrix = ada_result(train)
    print("##############################")
    print("From the adaboost, the result on train set is: ")
    print("{} Dutch was classified as Dutch".format(ada_matrix[0]))
    print("{} Dutch was classified as Italian".format(ada_matrix[1]))
    print("{} Italian was classified as Italian".format(ada_matrix[2]))
    print("{} Italian was classified as Dutch".format(ada_matrix[3]))
    print("The accuracy is {}".format((ada_matrix[0] + ada_matrix[2]) / sum(ada_matrix)))

    ada_matrix = ada_result(test)
    print("##############################")
    print("From the adaboost, the result on test set is: ")
    print("{} Dutch was classified as Dutch".format(ada_matrix[0]))
    print("{} Dutch was classified as Italian".format(ada_matrix[1]))
    print("{} Italian was classified as Italian".format(ada_matrix[2]))
    print("{} Italian was classified as Dutch".format(ada_matrix[3]))
    print("The accuracy is {}".format((ada_matrix[0] + ada_matrix[2]) / sum(ada_matrix)))


##############################
## predict
def data_cleaning(text):
    """
    Clean data for prediction
    :param text: data for prediction
    :return: cleaned_data
    """
    DeWordLst = []
    DeWordLst_cleaned = []
    result = []
    with open(text, 'r') as file:
        for line in file:
            temp = []
            temp.append(line)
            DeWordLst.append(temp)
    for j in DeWordLst:
        for i in j:
            string = ""
            for k in i.split():
                k = re.sub('[0-9]+', '', k)
                k = re.sub('[^\w\s]', "", k)
                k = re.sub(' ', '', k)
                k = k.lower()
                string = string + k + " "
            temp = []
            temp.append(string)
            result.append(temp)

    for i in result:
        for j in Dutch_WordList:
            count = i[0].split().count(j)
            i.append(count)
        for k in Italian_WordList:
            count = i[0].split().count(k)
            i.append(count)
    return result


def create_dataframe(clean_data):
    """
    Create dataframe from cleaned data
    :param clean_data: cleaned_data
    :return: Pandas dataframe
    """
    df = pd.DataFrame(clean_data, columns=["words", "deD", "vanD", "eenD", "enD", "hetD",
                                     "isD", "opD", "teD", "metD", "diI", "eI",
                                     "ilI", "laI", "cheI", "aI", "perI", "unI",
                                     "delI"])

    df = df.drop(columns=["words"])
    return df

def ada_result_prediction(data):
    """
    Generate classification results of 3 stumps with
    majority vote for prediction
    :param data:
    :return: classification result of adaboost
    """
    result1 = classifier_stump1(data)
    result2 = classifier_stump2(data)
    result3 = classifier_stump3(data)
    final_result = []
    for i in range(0, len(data.index)):
        temp = []
        temp.append(result1[i])
        temp.append(result2[i])
        temp.append(result3[i])
        if temp.count("Dutch") > temp.count("Italian"):
            final_result.append("Dutch")
        else:
            final_result.append("Italian")
    return final_result

if Dutch_train is None and Italian_train is None:
    dataframe = create_dataframe(data_cleaning(Predict_text))
    decision_tree_result = classifier_depth4(dataframe)
    print("Predicting")
    if model_type == "DecisionT":
        print("This is the predicted label of each data point from decision tree with depth 4: ")
        print(decision_tree_result)
    elif model_type == "Adaboost":
        print("##############################")
        print("This is the predicted label of each data point from adaboost: ")
        print(ada_result_prediction(dataframe))
    else:
        print("Model type can only be 'DecisionT' or 'Adaboost'")

