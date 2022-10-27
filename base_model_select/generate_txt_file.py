import linecache as lc
import numpy as np

txt_file_size = 5000
target_size = 200


randidx = np.random.randint(txt_file_size, size=target_size)

# changed to global dataset location
filename = "/Data/coco/val.txt"
generatefile = 'evaluate2017.txt'

with open(generatefile, 'w+') as genfile:

    for i in randidx.tolist():
        line = lc.getline(filename, i)
        genfile.writelines(line)


    
    