import numpy as np
from collections import Counter

def get_entropy_of_dataset(data: np.ndarray) -> float:
    """
    Calculate the entropy of the entire dataset using the target variable (last column).
    
    Args: data (np.ndarray): Dataset where the last column is the target variable
    
    Returns:
        float: Entropy value calculated using the formula: 
               Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i
    
    Example:
        data = np.array([[1, 0, 'yes'],
                        [1, 1, 'no'],
                        [0, 0, 'yes']])
        entropy = get_entropy_of_dataset(data)
        # Should return entropy based on target column ['yes', 'no', 'yes']
    """
    target = data[:, -1]                       
    _, counts = np.unique(target, return_counts=True)

    total = counts.sum()
    entropy = 0.0

    for c in counts:
        prob = c / total
        if prob > 0:
            entropy += -(float(prob) * np.log2(float(prob) + 1e-9)) 
    return round(float(entropy), 6)


def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the average information (weighted entropy) of a specific attribute.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
        attribute (int): Index of the attribute column to calculate average information for
    
    Returns:
        float: Average information calculated using the formula:
               Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) 
               where S_v is subset of data with attribute value v
    
    Example:
        data = np.array([[1, 0, 'yes'],
                        [1, 1, 'no'],
                        [0, 0, 'yes']])
        avg_info = get_avg_info_of_attribute(data, 0)  # For attribute at index 0
        # Should return weighted average entropy for attribute splits
    """
    total_rows = len(data)
    col = data[:, attribute]
    unique_vals, counts = np.unique(col, return_counts=True)

    wt_ent = 0.0
    for v, cnt in zip(unique_vals, counts):
        subset = data[col == v]
        sub_ent = get_entropy_of_dataset(subset)
        wt = cnt / total_rows
        wt_ent += wt * sub_ent
    return round(float(wt_ent), 6)


def get_information_gain(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the Information Gain for a specific attribute.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
        attribute (int): Index of the attribute column to calculate information gain for
    
    Returns:
        float: Information gain calculated using the formula:
               Information_Gain = Entropy(S) - Avg_Info(attribute)
               Rounded to 4 decimal places
    
    Example:
        data = np.array([[1, 0, 'yes'],
                        [1, 1, 'no'],
                        [0, 0, 'yes']])
        gain = get_information_gain(data, 0)  # For attribute at index 0
        # Should return the information gain for splitting on attribute 0
    """
    ent = get_entropy_of_dataset(data)
    avg = get_avg_info_of_attribute(data, attribute)

    ig = ent - avg
    return round(float(ig), 4)

def get_selected_attribute(data: np.ndarray) -> tuple:
    """
    Select the best attribute based on highest information gain.
    
    Args: data (np.ndarray): Dataset where the last column is the target variable
    
    Returns:
        tuple: A tuple containing:
            - dict: Dictionary mapping attribute indices to their information gains
            - int: Index of the attribute with the highest information gain
    
    Example:
        data = np.array([[1, 0, 2, 'yes'],
                        [1, 1, 1, 'no'],
                        [0, 0, 2, 'yes']])
        result = get_selected_attribute(data)
        # Should return something like: ({0: 0.123, 1: 0.456, 2: 0.789}, 2)
        # where 2 is the index of the attribute with highest gain
    """
    num_attributes = data.shape[1] - 1   # '- 1' is to not include last column (target variable)
    gains = {}
    for attr in range(num_attributes): gains[attr] = get_information_gain(data, attr)

    best_attr = max(gains, key=gains.get)
    return gains, best_attr