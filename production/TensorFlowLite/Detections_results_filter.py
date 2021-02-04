# Import packages
import os
import argparse
class ResultFilter(object):

    def __init__(self):
        self.wanted_objects = []
    
    def get_wanted_objects(self, wanted_objects_file):
        """
        This function retrieves the wanted objects from a file
        """
        # Load the objects map
        with open(wanted_objects_file, 'r') as f:
            self.wanted_objects = [line.strip() for line in f.readlines()]

    def get_boolean_labels(self, labels, detected_classes):
        """
        This funtion take ALL labels, compared those with wantedobjects (wanted by the
        user), and finally return and array of boolean, where true values are values that correspond to ob
        ject wanted by the user. This will avoid comparing every single object with the hole labels in the caller script.
        """
        booleanlabels = []
        for i in range(len(detected_classes)):
            label = labels[int(detected_classes[i])]
            if label in self.wanted_objects:
                booleanlabels.append(True)
            else:
                booleanlabels.append(False)
        return booleanlabels
