class AttrDict:
    def __init__(self, std_dict):
        self.std_dict = std_dict

    def __getattr__(self, attr):
        return self.std_dict[attr]
