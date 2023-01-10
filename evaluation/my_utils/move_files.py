
import os
import glob
import re


if __name__ == "__main__":

    root_path = "../../results/euroc"
    allMoveTxt = glob.glob(root_path+"/f_*.txt")

    for file in allMoveTxt:
        dstFile = re.sub(root_path+"/f_", root_path+"/of_", file)
        print(">> File: "+file)
        print("-- Rename: "+dstFile)
        os.rename(file, dstFile)
