# ScanNet util_3d: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/util_3d.py

import json, numpy as np

def load_ids(filename):
    """
    Loads IDs from a file.

    Parameters:
    filename (str): The name of the file to load IDs from.

    Returns:
    np.ndarray: An array of IDs.
    """
    ids = open(filename).read().splitlines()
    ids = np.array(ids, dtype=np.int64)
    return ids


# ------------ Instance Utils ------------ #

class Instance(object):
    """
    A class to represent an instance in a 3D scene.
    """
    instance_id = 0
    label_id = 0
    vert_count = 0
    med_dist = -1
    dist_conf = 0.0

    def __init__(self, mesh_vert_instances, instance_id):
        """
        Initializes an instance with its ID and the vertices of the mesh it belongs to.

        Parameters:
        mesh_vert_instances (np.ndarray): The vertices of the mesh the instance belongs to.
        instance_id (int): The ID of the instance.
        """
        if (instance_id == -1):
            return
        self.instance_id     = int(instance_id)
        self.label_id    = int(self.get_label_id(instance_id))
        self.vert_count = int(self.get_instance_verts(mesh_vert_instances, instance_id))

    def get_label_id(self, instance_id):
        """
        Gets the label ID of the instance.

        Parameters:
        instance_id (int): The ID of the instance.

        Returns:
        int: The label ID of the instance.
        """
        return int(instance_id // 1000)

    def get_instance_verts(self, mesh_vert_instances, instance_id):
        """
        Gets the vertices of the instance.

        Parameters:
        mesh_vert_instances (np.ndarray): The vertices of the mesh the instance belongs to.
        instance_id (int): The ID of the instance.

        Returns:
        int: The number of vertices of the instance.
        """
        return (mesh_vert_instances == instance_id).sum()

    def to_json(self):
        """
        Converts the instance to a JSON string.

        Returns:
        str: A JSON string representation of the instance.
        """
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def to_dict(self):
        """
        Converts the instance to a dictionary.

        Returns:
        dict: A dictionary representation of the instance.
        """
        dict = {}
        dict["instance_id"] = self.instance_id
        dict["label_id"]    = self.label_id
        dict["vert_count"]  = self.vert_count
        dict["med_dist"]    = self.med_dist
        dict["dist_conf"]   = self.dist_conf
        return dict

    def from_json(self, data):
        """
        Loads the instance from a JSON string.

        Parameters:
        data (str): A JSON string representation of the instance.
        """
        self.instance_id     = int(data["instance_id"])
        self.label_id        = int(data["label_id"])
        self.vert_count      = int(data["vert_count"])
        if ("med_dist" in data):
            self.med_dist    = float(data["med_dist"])
            self.dist_conf   = float(data["dist_conf"])

    def __str__(self):
        """
        Returns a string representation of the instance.

        Returns:
        str: A string representation of the instance.
        """
        return "("+str(self.instance_id)+")"


def get_instances(ids, class_ids, class_labels, id2label):
    """
    Gets instances from IDs.

    Parameters:
    ids (np.ndarray): An array of IDs.
    class_ids (np.ndarray): An array of class IDs.
    class_labels (list of str): A list of class labels.
    id2label (dict): A dictionary mapping from IDs to labels.

    Returns:
    dict: A dictionary mapping from labels to instances.
    """
    instances = {}
    for label in class_labels:
        instances[label] = []
    instance_ids = np.unique(ids)
    for id in instance_ids:
        if id == 0:
            continue
        inst = Instance(ids, id)
        if inst.label_id in class_ids:
            instances[id2label[inst.label_id]].append(inst.to_dict())
    return instances





