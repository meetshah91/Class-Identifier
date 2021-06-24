"""
    define decision tree structure which stores decision on each node
    and its corresponding values which allows validation data to make decision
    @author      Meet Shah
"""


class decision_btree:
    """
    :param threshold: threshold value on which we want to split data into 2 parts (for leaf node the threshold will
    not be initialized)
    :param left: left side of thresholds decision tree :param right: right side of thresholds
    decision tree
    :param depth: depth of a node of decision tree :param isLeafNode: is node is a leaf node or not to
    make classification
    :param column_name: name of column on which we have made split
    :param classification:
    indicate data is whether it is assam or bhutan (for non leaf nodes its null)
    """
    __slots__ = "threshold", "left", "right", "depth", "isLeafNode", "column_name", "classification"

    def __init__(self, depth):
        self.depth = depth
        self.left = None
        self.right = None
        self.threshold = None
        self.isLeafNode = False
        self.column_name = ""
        self.classification = 0

    def set_classification(self, _classification):
        """
        set classification for the decision node
        :param _classification: value of classification to set
        :return:
        """
        self.classification = _classification

    def set_column_name(self, _column_name):
        """
        set column_name which used to split data of that node
        :param _column_name: value of column_name to set
        :return:
        """
        self.column_name = _column_name

    def set_is_leaf_node(self, _is_leaf_node):
        """
        set value of isLeafNode column of decision node
        :param _is_leaf_node: value of isLeafNode to set
        :return:
        """
        self.isLeafNode = _is_leaf_node

    def set_threshold(self, _threshold):
        """
        set threshold column which used to split data into two part
        :param _threshold: value of isLeafNode to set
        :return:
        """
        self.threshold = _threshold

    def copy_decision_node(self, decision_node):
        """
        copy value of decision nodes into current node
        :param decision_node: copy value of decision node from
        :return:
        """
        self.isLeafNode = decision_node.isLeafNode
        self.classification = decision_node.classification
        self.column_name = decision_node.column_name
        self.left = decision_node.left
        self.right = decision_node.right

    def __str__(self):

        if self.isLeafNode:
            return "\n\t" + str(self.classification)
        else:
            return "( L  : " + str(self.depth) + " :" + str(self.left) + ")" + "(R " + str(self.depth) + " : " + str(
                self.right) + ")"
