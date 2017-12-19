function main()

dataset_filepath = '../../Dataset/kc_house_data.csv';

fid = fopen(dataset_filepath,'r');

line = fgetl(fid);

content = textscan(fid,'%s %s %f %f %f %f %f %s  %f %f %f %f %f %f %f %f %s %f %f %f %f','delimiter',',');

fclose(fid);

Y = content{3};
X = [];

%% Convert Features to matrix
for d_i = 4:7
    X = [X content{d_i}];
end

tmp = content{8};
x_dim = [];
for s_i = 1:length(tmp)
    
    strnum = tmp{s_i}(2:end-1);
    x_dim(s_i,1) = str2double(strnum);
    
end

X = [X x_dim];

for d_i = 9:16
    X = [X content{d_i}];
end

tmp = content{17};
x_dim = [];
for s_i = 1:length(tmp)
    
    strnum = tmp{s_i}(2:end-1);
    x_dim(s_i,1) = str2double(strnum);
    
end

X = [X x_dim];

for d_i = 18:21
    X = [X content{d_i}];
end

%% Training Model
X_te = X(1:4322,:);
Y_te = Y(1:4322,:);
X_tr = X(4323:end,:);
Y_tr = Y(4323:end,:);

W = (X_tr'*X_tr)\X_tr'*Y_tr;
W = inv(X_tr'*X_tr+0.1*eye(size(X_tr,2)))*X_tr'*Y_tr;

[RMSE_tr,MAE_tr,Y_hat_tr] = func_EvalPerf(X_tr,Y_tr,W);

Y_te_hat = X_te*W;

[RMSE_te,MAE_te,Y_hat_te] = func_EvalPerf(X_te,Y_te,W);



function [RMSE,MAE,Y_hat] = func_EvalPerf(X,Y,W)

Y_hat = X*W;
RMSE = sqrt(mean((Y-Y_hat).^2));
MAE = mean(abs(Y-Y_hat));




