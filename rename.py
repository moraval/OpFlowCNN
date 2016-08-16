import os
import sys
from shutil import copyfile

# renaming all files in directory to create dataset for torch
dirname = sys.argv[1]
outdirname = sys.argv[2]
batchsize = int(sys.argv[3])


files = os.listdir(dirname)
print(size(files))
files.sort()

# naming of the files in the directory - will be different for every one
# changing NAMING to: in01a + in01b
#for filename in files:

for i in xrange(batchsize):
    # copy the 2 files
    file1 = files[i]
    file2 = files[i+1]
    num = i+1
    copyfile(dirname + file1, outdirname + "in" + str(num).zfill(2) + "a.png")
    copyfile(dirname + file2, outdirname + "in" + str(num).zfill(2) + "b.png")

