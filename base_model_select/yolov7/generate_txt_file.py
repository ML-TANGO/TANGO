
# %%
import linecache as lc
import numpy as np
from pathlib import Path

txt_file_size = 5000
target_size = 800


randidx = np.random.randint(txt_file_size, size=target_size)

# changed to global dataset location
# print(Path().absolute())
filename = "coco/val2017.txt"
generatefile = 'evaluate2017.txt'

# assert Path(generatefile).exists()
assert Path(filename).exists()

with open(generatefile, 'w+') as genfile:
    
    dataset_path = Path("coco")

    for i in randidx.tolist():
        line = lc.getline(filename, i)
        path = (dataset_path / line).resolve()
        genfile.writelines(str(path))
        # print(str(path))


    
    
# %%
