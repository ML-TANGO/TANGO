import linecache as lc
import numpy as np
from pathlib import Path


txt_file_size = 5000
target_size = 200


randidx = np.random.randint(txt_file_size, size=target_size)

# changed to global dataset location
filename = "datasets/coco/val2017.txt"
generatefile = 'evaluate2017.txt'

with open(generatefile, 'w+') as genfile:
    
    dataset_path = Path("datasets/coco")

    for i in randidx.tolist():
        line = lc.getline(filename, i)
        path = (dataset_path / line).resolve()
        genfile.writelines(str(path))


    
    