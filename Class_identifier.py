import math
import sys
import pandas as pd
from decison_tree import decision_btree
from file_healper import file_operation

"""
    this program trains a model using decision tree and generates a new program which
    uses that model to make classification of data
    @author      Meet Shah
"""


def generate_decision_tree(classification_training_data, bin_size, decision_node: decision_btree,split_type):
    """
    generate decision tree using certain features in data
    :param classification_training_data: data set that contains record for both assam and bhuttan people for training
    :param bin_size: size of bins used
    :param decision_node: decision tree node for which we are doing decision
    :param split_type:  how to split data using entropy or gini index
    :return: generated decision tree
    """

    # checking if reached base case of decision tree or not
    if is_leaf_node(classification_training_data, decision_node):
        # setting node as leaf node and can be considered as decision node
        decision_node.set_is_leaf_node(True)

        # setting the classification of node 1 indicates the node is assam
        # and -1 considered node as bhuttan
        decision_node.set_classification(get_classification(classification_training_data))

        # return the decision tree
        return decision_node

    # initialize value of best weighted average for split
    best_weighted_avg = 1

    # initialize value of best threshold
    best_threshold = 0

    # initialize column name where we have found minimum gini index(best split)
    column_name_min_gini = ""

    # iterate over each features and find best threshold for
    # each feature and find best possible split for the
    # given data using gin index
    for column_name in classification_training_data.columns:

        if column_name == "Class":
            continue

        # initialize start value of threshold for finding best threshold
        start_threshold = classification_training_data[column_name].min()

        # initialize end value of threshold for finding best threshold
        end_threshold = classification_training_data[column_name].max() + bin_size

        # function that finds best threshold based on start and end thresholds and column name
        threshold, weighted_avg = get_best_threshold_column(classification_training_data, start_threshold,
                                                            end_threshold, column_name,
                                                            bin_size,split_type)

        # if the given gini index is less then the best gini index found then we have
        # found best split for our data
        if best_weighted_avg > weighted_avg:
            best_weighted_avg = weighted_avg
            best_threshold = threshold
            column_name_min_gini = column_name

    # set value of column name of node for which we have found best split for decision node
    decision_node.set_column_name(column_name_min_gini)

    # set value of threshold of node for which we have found best split for decision node
    decision_node.set_threshold(best_threshold)

    # initialize left side of decision node
    decision_node.left = decision_btree(decision_node.depth + 1)

    # initialize right side of decision node
    decision_node.right = decision_btree(decision_node.depth + 1)

    # gives data which is less then or equal to threshold for column name in form of dataframes
    classification_training_data_left = classification_training_data[
        classification_training_data[column_name_min_gini] <= best_threshold]

    # generate decision tree on left side of data
    generate_decision_tree(classification_training_data_left, bin_size, decision_node.left,split_type)

    # gives data which is grater then  threshold for column name in form of dataframes
    classification_training_data_right = classification_training_data[
        classification_training_data[column_name_min_gini] > best_threshold]

    # generate decision tree on right side of data
    generate_decision_tree(classification_training_data_right, bin_size, decision_node.right,split_type)

    return decision_node


def get_classification(classification_training_data):
    """
    this function checks data and returns whether data set belong to assam or bhutan
    :param classification_training_data: data set that contains record for both assam and bhuttan people for training
    :return: return classification of given data if data set belongs to assam then 1  if bhuttan then -1
    """
    # counts total number of people in assam and bhuttan in the in the provided training set
    total_bhuttan, total_assam = get_total_assam_bhuttan(classification_training_data)
    return 1 if total_assam > total_bhuttan else -1


def is_leaf_node(classification_training_data, decision_node: decision_btree):
    """
    checkes whether we have reached to leaf node or not
    :param classification_training_data:  data set that contains record for both assam and bhuttan people for training
    :param decision_node: decision tree node for which we are doing decision
    :return: True if we have reached to leaf node of decision tree otherwise False
    """
    # counts total number of people in assam and bhuttan in the in the provided training set
    total_bhuttan, total_assam = get_total_assam_bhuttan(classification_training_data)

    # checks if total data in the node is less then 13 or not if Yes then stop
    if len(classification_training_data.index) < 13:
        return True

    # calculates percentage of assam records in data set
    total_assam_per = (total_assam / (total_bhuttan + total_assam)) * 100

    # calculates percentage of bhuttan records in data set
    total_bhutan_per = (total_bhuttan / (total_bhuttan + total_assam)) * 100

    # checks if percentage of any class of records are grater than 95 or not
    if total_assam_per >= 95 or total_bhutan_per >= 95:
        return True

    # checks if depth of decision tree is grater than 6 or not
    if decision_node.depth >= 6:
        return True

    return False


def get_best_threshold_column(classification_training_data, start_threshold, end_threshold, column_name, bin_size,
                              split_type="Gini"):
    """
    find the best threshold value from start and end threshold for respective bin size
    :param classification_training_data: data set that contains record for both assam and bhuttan people
    :param start_threshold: start point to find best threshold
    :param end_threshold: end point to find best threshold
    :param column_name: column to use to find threshold
    :param bin_size: bin size used to find thresholds
    :param split_type: type of split used to split data entropy or gini index
    :return: best threshold value for the column
    """
    # initialize threshold
    threshold = start_threshold

    # Stores value of best thresholds
    best_threshold = sys.maxsize

    # stores value of best minimum error
    best_split_error = 1

    # check each threshold and find error rate for each thresholds
    while threshold < end_threshold:
        # count of left hand side , right hand side , count of miscalculation on left side
        # count of miscalculation on right side of decision tree
        total_bhuttan_left, total_assam_left, total_bhuttan_right, total_assam_right \
            = get_bifurcation_values(threshold, classification_training_data, column_name)
        if split_type == "Entropy":
            # function that computes gini index on left side of data
            gini_or_entropy_left = calculate_entropy(total_bhuttan_left, total_assam_left)

            # function that computes gini index on right side of data
            gini_or_entropy_right = calculate_entropy(total_bhuttan_right, total_assam_right)

        else:
            # function that computes gini index on left side of data
            gini_or_entropy_left = claclulate_gini_index(total_bhuttan_left, total_assam_left)

            # function that computes gini index on right side of data
            gini_or_entropy_right = claclulate_gini_index(total_bhuttan_right, total_assam_right)

        # function that computes weighted gini index based on left and right side of data
        weighted_gini = weighted_average(gini_or_entropy_left, total_bhuttan_left + total_assam_left,
                                         gini_or_entropy_right, total_bhuttan_right + total_assam_right)

        # assign best gini index to value
        if best_split_error >= weighted_gini:
            best_split_error = weighted_gini
            best_threshold = threshold
        threshold += bin_size

    return best_threshold, best_split_error


def get_total_assam_bhuttan(classification_training_data):
    """
    calculates total number of assam and bhuttan people in given dataset
    :param classification_training_data: data set that contains record for both assam and bhuttan people
    :return: number of assam and bhuttan people
    """

    # count of bhutan people in the data frame using column name class in panda
    total_bhuttan = len(classification_training_data[classification_training_data["Class"] == "Bhuttan"].index)

    # count of assam people in the data frame using column name class in panda
    total_assam = len(classification_training_data[classification_training_data["Class"] == "Assam"].index)

    return total_bhuttan, total_assam


def claclulate_gini_index(classification1, classification2):
    """
    calculate gini index for the split
    :param classification1: total count of records in classification 1
    :param classification2: total count of records in classification 2
    :return:  calculated gini index for the split
    """
    if classification1 + classification2 == 0:
        return 0.5
    # square of fraction of data in classification 1
    c1_square = (classification1 / (classification1 + classification2)) ** 2

    # square of fraction of data in classification 2
    c2_square = (classification2 / (classification1 + classification2)) ** 2

    return 1 - c1_square - c2_square


def weighted_average(gini_or_entropy_left, left_cnt, gini__or_entropy_right, right_cnt):
    """
    calculate weighted average  for Gini index or Entropy
    :param right_cnt: count of total records on the right side of node
    :param left_cnt: count of total records on left side of node
    :param gini_or_entropy_left: gini index or Entropy of left side of node
    :param gini__or_entropy_right: gini index or Entropy of right side of node
    :return: weighted average of entire node
    """

    # formula used to calculate weighted gini index
    weighted_avg = ((left_cnt / (left_cnt + right_cnt)) * gini_or_entropy_left) + (
            (right_cnt / (left_cnt + right_cnt)) * gini__or_entropy_right)

    return weighted_avg


def calculate_entropy(classification1, classification2):
    """
    calculate entropy for the split
    :param classification1: total count of records in classification 1
    :param classification2: total count of records in classification 2
    :return:  calculated gini index for the split
    """
    if classification1 == 0 or classification2 == 0:
        return 0
    # computation of classification1
    c1_comp = (classification1 / (classification1 + classification2))

    # computation of classification2
    c2_comp = (classification2 / (classification1 + classification2))

    return -1 * (c1_comp * math.log(c1_comp, 2) + c2_comp * math.log(c2_comp, 2))


def get_bifurcation_values(threshold, classification_training_data, column_name):
    """
    count of grass and weed and error in that identification based on thresholds
    :param threshold: threshold value to divide data into left and right side
    :param classification_training_data: data set that contains record for both assam and bhuttan people
    :param column_name: name of column in the data frame on which we want to bifurcate data
    :return: return count for classification based on threshold and error in classification
    """

    # get the total data that is not left side of the threshold using pandas column filter
    # which gives us rows based on column validation mentioned inside the [] bracket
    left_side_data = classification_training_data[classification_training_data[column_name] <= threshold]

    # get the total data that is not right side of the threshold using pandas column filter
    # which gives us rows based on column validation mentioned inside the [] bracket
    right_side_data = classification_training_data[classification_training_data[column_name] > threshold]

    # counts total number of people in assam and bhuttan in the in the left side of training set
    total_bhuttan_left, total_assam_left = get_total_assam_bhuttan(left_side_data)

    # counts total number of people in assam and bhuttan in the in the right side of training set
    total_bhuttan_right, total_assam_right = get_total_assam_bhuttan(right_side_data)

    return total_bhuttan_left, total_assam_left, total_bhuttan_right, total_assam_right


def trim_decision_tree(decision_tree: decision_btree):
    """
    removes extra branches which are not needed in decision trees
    ex removes leaf nodes whose parent is same and yield same result of classification
    :param decision_tree: decision tree which needed to trim
    :return:
    """
    # if decision tree is empty
    if decision_tree is None:
        return
    # explore left child of decision tree
    if decision_tree.left is not None and not decision_tree.left.isLeafNode:
        trim_decision_tree(decision_tree.left)

    # explore right child of decision tree
    if decision_tree.right is not None and not decision_tree.right.isLeafNode:
        trim_decision_tree(decision_tree.right)

    # if both right and left child are present in decision tree
    # and both are leaf node and classification is same for both leaf node
    # remove both leaf nodes and make their parent a leaf node
    if decision_tree.left is not None and decision_tree.right is not None:
        if decision_tree.left.isLeafNode and decision_tree.right.isLeafNode:
            if decision_tree.left.classification == decision_tree.right.classification:
                decision_tree.copy_decision_node(decision_tree.left)


def print_tree(decision_tree: decision_btree, file_write):
    """
    convert decision tree into nested if else statement that can be used
    to write in python file
    :param file_write: file instance in which used for writing into file
    :param decision_tree: decision tree which need to write into file
    :return:
    """
    # defines tab which used for indentation in python file
    tab_space = "\t"
    if decision_tree.isLeafNode:
        file_write.writelines(
            ["\n\t", tab_space * decision_tree.depth, "print(", str(decision_tree.classification), ")"])
        file_write.writelines(
            ["\n\t", tab_space * decision_tree.depth, "classification_result.append(",
             str(decision_tree.classification), ")"])
    else:

        file_write.writelines(
            ["\n\t", tab_space * decision_tree.depth, "if cls_validation_data[\"", decision_tree.column_name,
             "\"][idx]<=",
             str(decision_tree.threshold), ":"])
        print_tree(decision_tree.left, file_write)
        file_write.writelines(["\n\t", tab_space * decision_tree.depth, "else: "])
        print_tree(decision_tree.right, file_write)


def generate_classifier_program(file_name, decision_tree, validation_file_name):
    """
    function generates a new classifier program based on the trained decision tree
    :param file_name: name of classifier program
    :param decision_tree: decision tree based on which we need to make decision
    :param validation_file_name: add validation file name
    :return:
    """
    # open file for writing if not exist then create a new file
    file_write = open(file_name, "w")

    # function used to write import statements in code
    insert_import_statement(file_write)

    # code to write a new classifier function using decision tree
    file_write.writelines(["\ndef classify_validation_data(cls_validation_data):"])
    file_write.writelines(["\n\tclassification_result = list()"])
    file_write.writelines(["\n\tfor idx in cls_validation_data.index:"])

    # writes the decision tree structure in if then else statements in code
    print_tree(decision_tree, file_write)
    file_write.writelines(["\n\treturn classification_result"])

    # checks validation file name is provided or not if not it will pass default name
    insert_main_function(file_write, validation_file_name)


def insert_main_function(file_write, validation_file_name):
    """
    write the content of main function in the file
    :param validation_file_name: add validation file name
    :param file_write: file instance in which used for writing into file
    :return:
    """

    # write the main() function call in the code
    file_write.writelines(["\ndef main():"])
    file_write.writelines(["\n\tfile_obj = file_operation()"])
    file_write.writelines(["\n\tcls_validation_data = file_obj.read_csv(\"", validation_file_name, "\")"])
    file_write.writelines(["\n\tclassification_result = classify_validation_data(cls_validation_data)"])
    file_write.writelines(["\n\tcls_validation_data.insert(8, \"cls_result\", classification_result, True)"])
    file_write.writelines(["\n\tcls_validation_data.to_csv('MyClassifications.csv')"])
    file_write.writelines(["\n\nif __name__ == '__main__':"])
    file_write.writelines(["\n\tmain()"])


def insert_import_statement(file_write):
    """
    import header into new files
    :param file_write: file instance in which used for writing into file
    :return:
    """
    # import necessary library which is require to run program
    file_write.writelines(["\nimport math"])
    file_write.writelines(["\nimport sys"])
    file_write.writelines(["\nimport pandas as pd"])
    file_write.writelines(["\nfrom decison_tree import decision_btree"])
    file_write.writelines(["\nfrom file_healper import file_operation"])


def main():
    # this function reads data from a .csv file and put it into
    if len(sys.argv) >= 2:
        # training data file name to read from
        training_data_file_name = sys.argv[1]

        # checks whether the validation data file name passed in command line or not
        # if passed then take data from that file else use default file name to read data from
        validation_file_name = sys.argv[3] if len(sys.argv) == 4 \
            else "Abominable_VALIDATION_Data.csv"

        # checks whether the type of split passed in command line or not
        # if passed then take from command line else use default type of split
        split_type = sys.argv[2] if len(sys.argv) >= 3 else "Gini"

        # creating object of file helper class
        file_obj = file_operation()

        # read training data from file and quantize that data for noise reduction
        classification_training_data = file_obj.read_csv(training_data_file_name)

        # generate decision tree based on training data
        decision = generate_decision_tree(classification_training_data, 1, decision_btree(1),split_type)

        # function removes all the unnecessary branches from decision tree and reduce it
        trim_decision_tree(decision)

        # function generate a new classifier program from decision tree
        generate_classifier_program("Classifier_shah.py", decision, validation_file_name)

    else:
        print("command argument is not valid please pass in below formats")
        print("training_data_filename")
        print("training_data_filename Gini/Entropy validation_file_name")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
