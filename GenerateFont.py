#!/usr/bin/python

import sys

usage = """For the program to execute correctly, you must specify which features you would like in your font with true or false for each argument.
            Usage: 
                python3 GenerateFont.py ${italicFeature} ${boldFeature} ${serifFeature}
            
            For example:
                python3 GenerateFont.py true false true"""

# Declare weights for NNs
weightsDict = {str([True, True, True]): [],
           str([True, True, False]): [],
           str([True, False, True]): [],
           str([True, False, False]): [],
           str([False, True, True]): [],
           str([False, True, False]): [],
           str([False, False, True]): [],
           str([False, False, False]): []}
weights = []

# Declare feature vector that user must input
features = dict()
featureKeys = ['italic', 'bold', 'serif']

def main():
    if(len(sys.argv) - 1 != len(featureKeys)):
        print('You have entered ', len(sys.argv), ' arguments. This program requires exactly ', len(featureKeys), ' arguments to run correctly.')
        print(usage)
        sys.exit()

    # Iterate through args and features and add them two the dictionary. The key is the feature name the value is T/F
    for arg, feature in zip(sys.argv[1:], featureKeys):
        print('Argument is', str(arg))
        if str(arg).lower() == 'true':
            features[feature] = True
        else:
            features[feature] = False

    # Get weights from weights dictionary.
    weights = get_architecture(weights_dict)

def get_architecture(weights_dict):
    return weights_dict[str(list(features.values()))]

if __name__ == "__main__":
    main()