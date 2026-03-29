class ModelResult:
    def __init__(self, name, accuracy, f1, conf_matrix, labels):
        self.name = name
        self.accuracy = round(accuracy, 4)
        self.f1 = round(f1, 4)
        self.conf_matrix = conf_matrix
        self.labels = labels

    def to_dict(self):
        return {
            "name": self.name,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "matrix": self.conf_matrix
        }