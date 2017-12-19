import numpy as np
import os
import DataSplit
import csv

def LoadAllData(DatasetFilepath):
    with open(DatasetFilepath, 'r') as fid:
        RawData = csv.DictReader(fid, delimiter=',')
        nSamples = 0

        # count number of samples
        for row in RawData:
            nSamples = nSamples + 1

        nFeature = len(row) - 3

        y = np.zeros((nSamples,1))
        x = np.ones((nSamples,nFeature))

        fid.seek(0)
        RawData = csv.DictReader(fid, delimiter=',')

        SampIdx = 0
        for row in RawData:
            print("%d-th Sample\n"%SampIdx)
            y[SampIdx] = float(row["price"])
            x_tmp = np.empty(0)
            for val in row.values():
                #print(val)
                x_tmp = np.append(x_tmp,val)
            # x = np.append(x,x_tmp)
            x[SampIdx,:] = x_tmp[3:x_tmp.size]

            SampIdx = SampIdx + 1

        return {'x':x,'y':y}

#   Evaluate Performance
#
#   Compute RMSE and MAE as performance evaluation
def EvalPerf(X,Y,W):
    Y_hat = np.matmul(X,W)
    # Compute RMSE
    tmp = Y_hat.shape
    nSample = tmp[0]
    RMSE = np.sqrt(1/nSample*(Y_hat-Y).T*(Y_hat-Y))
    NRMSE = RMSE/(Y.max()-Y.min())
    MAE = np.mean(np.abs(Y_hat - Y))
    return {'RMSE':RMSE , 'NRMSE':NRMSE ,'MAE':MAE , 'Prediction':Y_hat}



def main():

    # Load All Data
    DatasetFilepath = "../Dataset/kc_house_data.csv"
    AllData = LoadAllData(DatasetFilepath)
    Y = AllData["y"]
    X = AllData["x"]

    # Do 5 fold cross-validation
    SavePath = "./CrossValid"

    Prediction_Te = np.zeros(0)

    for cv_i in range(5):

        # Load Cross Validation Split
        SaveFilepath = os.path.join(SavePath, "CrossValid%d.npz" % (cv_i))
        Split = np.load(SaveFilepath)

        X_tr = np.mat(X[Split['TrainingIdx'],...])
        Y_tr = np.mat(Y[Split['TrainingIdx']])

        # Solve Linear Regression
        n = X_tr.shape[1]
        gamma = 0.1
        W = np.linalg.inv(X_tr.T * X_tr + gamma*np.identity(n)) * X_tr.T * Y_tr

        # Training Error
        Result = EvalPerf(X_tr, Y_tr, W)
        RMSE_Tr = Result['RMSE']
        MAE_Tr = Result['MAE']

        # Evaluate Error
        X_te = np.mat(X[Split['TestingIdx'],...])
        Y_te = np.mat(Y[Split['TestingIdx']])

        Result = EvalPerf(X_te, Y_te, W)

        RMSE = Result['RMSE']
        MAE = Result['MAE']
        Prediction_Te = np.append(Prediction_Te,Result['Prediction'])

        print('Cross Validation Fold %d \n Training: RMSE=%.2f MAE=%.2f \n RMSE=%.2f MAE=%.2f'%(cv_i,RMSE_Tr,MAE_Tr,RMSE,MAE))


if __name__ == "__main__":
    main()
