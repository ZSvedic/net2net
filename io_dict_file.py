import os.path as osp
from shutil import copyfile

import torch as th

####################################################################################################

class IODictFile:
    ''' Helper class for reading and writing generic dictionary. '''

    def __init__(self, path, file_name, get_dict_method, set_dict_method):
        if not osp.isdir(path):
            raise FileNotFoundError("Not a valid directory:", path)

        self.file_path = osp.join(path, file_name)
        self.get_dict_method = get_dict_method
        self.set_dict_method = set_dict_method

    def load(self, file_required):
        ''' Loads dictionary file. '''
        if osp.isfile(self.file_path):
            print("Loading from:", self.file_path)
            file_dict = th.load(self.file_path)
            self.set_dict_method(file_dict)
        else:
            if file_required:
                raise FileNotFoundError(self.file_path)
            else:
                print("File not found:", self.file_path)

    def save(self, best_copy=False):
        ''' Saves dictionary file. '''
        print("Saving to:", self.file_path)
        save_dict = self.get_dict_method()
        th.save(save_dict, self.file_path)
        if best_copy:
            copyfile(self.file_path, self.file_path+".best")

####################################################################################################
