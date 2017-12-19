import numpy as np
import csv as csv
import random
import os

DatasetFilepath = "../Dataset/kc_house_data.csv"

def main():
    print("In Main Function")
    fid = open(DatasetFilepath, 'r')
    line = fid.readline()
    print(line)
    with open(DatasetFilepath,'r') as csvfid:
        RawData = csv.DictReader(csvfid,delimiter=',')
        nSamples = 0
        for row in RawData:
            nSamples = nSamples+1
            print(row["id"])
        csvfid.close()
        print("Total Number of Samples = ",nSamples)

        # Randomly Split into 5 fold cross-validation sets
        RawIndex = np.arange(nSamples,dtype=np.int64)
        print(RawIndex)
        #np.random.shuffle(RawIndex)

        nCrossValid = 5
        for cv_i in range(nCrossValid):
            print("Cross Validation Set ",cv_i)
            TestingRange = np.arange(cv_i* np.round(nSamples/nCrossValid),(cv_i+1)*np.round(nSamples/nCrossValid)-1,dtype=np.int64)
            TrainingRange = np.setdiff1d(RawIndex,TestingRange)

            # Check Out of Range
            OutOfRangeFlag = 1
            while OutOfRangeFlag:
                if TestingRange.max() >= nSamples:
                    location = np.where(TestingRange==nSamples)
                    TestingRange = np.delete(TestingRange,location)
                else:
                    OutOfRangeFlag=0

            print("Train Set Size = ",TrainingRange.size)
            print("Test Set Size = ",TestingRange.size)

            TrainingIdx = RawIndex[TrainingRange]
            TestingIdx = RawIndex[TestingRange]

            SavePath = "./CrossValid"
            if os.path.isdir(SavePath) == True:
                print("Exist ./CrossValid directory")
            else:
                print("Doesnt Exist Directory ./CrossValid")
                os.mkdir(SavePath)

            SaveFilepath = os.path.join(SavePath,"CrossValid%d"%(cv_i))

            # Save Each Cross Validation Set
            np.savez(SaveFilepath,TrainingIdx=TrainingIdx,TestingIdx=TestingIdx)


if __name__ == "__main__":
    main()