import readutils
import os 

class Dataset:
    def read_dataset(self, location):
        '''Read dataset from mashup format'''

        self.location = location 
        
        self.read_newtork()
        self.read_genes()
        # self.read_annotations()
    
    def read_newtork(self):
        path = self.location
        if 'networks' not in os.listdir(path):
            # del self.location
            raise NotADirectoryError
        
        network_path = os.path.join(path, 'networks')
        network_folder_contents = {}
        _ = [network_folder_contents.update(c) for c in readutils.readfoldertext(network_path)]


        is_network = lambda x: len(readutils.split_line(x)) in [2, 3]
        

        

