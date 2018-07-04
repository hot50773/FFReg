clear;clc;
X_time_grid = load('d:/Work_Jupyter/敏/SIM_1y1x_sparse/sim_X_time_grid.mat');
Y_time_grid = load('d:/Work_Jupyter/敏/SIM_1y1x_sparse/sim_Y_time_grid.mat');
num_train = 200;
num_test = 100;
num_sample = num_train + num_test;
dt = Y_time_grid.grid(2) - Y_time_grid.grid(1);
MISE  = zeros(1,100);
RMSPE = zeros(1,100);
test_Yt = cell(1, num_sample);
for i = 1:num_test
    test_Yt{i} = Y_time_grid.grid;
end

% 設定FPCreg裡主要的參數:
% selection_k: 選eig_pairs的方法
% 'AIC1': use AIC criterion with pseudo-likelihood of measurements (marginal likelihood).
% 'AIC2': use AIC criterion with likelihood of measurements conditional on estimated random coeffecients.
% 'BIC1'[Default]: use BIC criterion with pseudo-likelihood of measurements (marginal likelihood).
% 'BIC2': use BIC criterion with likelihood measurements conditional on estimated random coeffecients.
% 'FVE' : use scree plot approach to select number of principal components), see "FVE_threshold".
% 如果你改成'FVE'，還要添加'FVE_threshold'參數，不加也可以，預設是0.85
% 
% regular: 0表示sparse, 1表示regular但有missing data, 2表示完整的regular data
% newdata: a row vector of user-defined output time grids
% verbose: 要不要把過程顯示出來，預設是on，很討厭= =
param_X = setOptions('selection_k','FVE', 'FVE_threshold', 0.95, 'regular', 0, 'newdata', X_time_grid.grid, 'verbose','off');
param_Y = setOptions('selection_k','FVE', 'FVE_threshold', 0.95, 'regular', 0, 'newdata', Y_time_grid.grid, 'verbose','off');
npred = num_test; % 資料理有幾筆是要拿來test的

for j = 1:10

    X_data = load(['d:/Work_Jupyter/敏/SIM_1y1x_sparse/sim_X_data_',num2str(j),'.mat']);
    Y_data = load(['d:/Work_Jupyter/敏/SIM_1y1x_sparse/sim_Y_data_',num2str(j),'.mat']);

    % 因為FPCreg輸入不能有NA，要先把NA弄掉
    % 且輸入必須是一個1 * num_fun的cell (matlab的資料格式)，所以創新的cell存t和obs
    new_Xt = cell(1, num_sample);
    new_Xobs = cell(1, num_sample);
    non_nan_X = ~isnan(X_data.obs);

    for i = 1:num_sample
        non_nan_Xi = non_nan_X(i, :);
        new_Xt{i} = X_data.t(i, non_nan_Xi);
        new_Xobs{i} = X_data.obs(i, non_nan_Xi);
    end

    % 同理，以下做Y的部分
    new_Yt = cell(1, num_sample);
    new_Yobs = cell(1, num_sample);
    non_nan_Y = ~isnan(Y_data.obs);
    
    for i = 1:num_sample
        non_nan_Yi = non_nan_Y(i, :);
        new_Yt{i} = Y_data.t(i, non_nan_Yi);
        new_Yobs{i} = Y_data.obs(i, non_nan_Yi);
    end


% 他的Regression函式: FPCreg(x, t_x, y, t_y,param_X, param_Y,FIT,K_x,K_y,npred,alpha); %Functional regression
% 可以參考 http://www.stat.ucdavis.edu/PACE/Help/FPCregHELP.txt
    [res, xx, yy] = FPCreg(new_Xobs, new_Xt, new_Yobs, new_Yt,param_X, param_Y,0,[],[],npred);

% FPCpred(res, xx, yy, newx, new_tx, y, new_ty, FIT, alpha)
%======
%  res, xx, yy:  The returned values from FPCreg. See FPCreg() for more details.
%  newx:   1*numNewSub cell array contains measurements for new x functions
%  new_tx: 1*numNewSub cell array contains time points corresponding to the newx
%  y:      Only needed when Y is a scalar.
%  new_ty: 1*numNewSub cell array contains time points.
%  FIT:    an indicator with values 0 or -1.
%          Refer to the input FIT of FPCreg for more details.
%  alpha:  the level of the confidence bands.  alpha = 0.05 if is left empty. 
%          No confidence bands will be produced if the inputted alpha is out of (0, 1).
%  預測
    fit_test_Y = FPCpred(res, xx, yy, new_Xobs((num_train+1):num_sample), new_Xt((num_train+1):num_sample), 0, test_Yt, -1);
    fit_test_Y = transpose(reshape(cell2mat(fit_test_Y), 21, 100));
%  計算 MISE 和 RMSPE   
    MISE(1,j) = mean(sum((Y_data.real_test_Y - fit_test_Y).^2 * dt, 2));
    RMSPE(1,j) = mean(sum((Y_data.real_test_Y - fit_test_Y).^2 * dt, 2) ./ sum((Y_data.real_test_Y).^2 * dt, 2));
    
    % 畫圖
    if j <= 10
        obs_test_Y = Y_data.obs((num_train + 1):num_sample, :);
        obs_test_Yt = Y_data.t((num_train + 1):num_sample, :);
        % 畫前四張圖
        fig = figure;
        for i = 1:4
            subplot(2, 2, i);
            [sorted_obs_yt, index_yt] = sort(obs_test_Yt(i,:));
            plot(sorted_obs_yt, obs_test_Y(i,index_yt), 'go');
            ylim([-10 10]);
            xlim([0 5]);
            hold on;
            plot(Y_time_grid.grid, fit_test_Y(i, :),'r--');
            plot(Y_time_grid.grid, Y_data.real_test_Y(i,:),'b--');
            title(['Predict Y_' num2str(i) ' of Sim ' num2str(j)]);
            xlabel('t');
            ylabel('Y(t)');
            
            legend('obs','pred','real','Location','northeastoutside');
        end
        saveas(fig,['Plot_PACE_SIM'  num2str(j) '.png'])
    end
end

% save MISE and RMSPE as structure
% save('d:/Work_Jupyter/敏/SIM_1y1x_sparse/result_PACE_sparse.mat', 'MISE','RMSPE');

% MISE_and_RNSPE = load('d:/Work_Jupyter/敏/SIM_1y1x_sparse/result_PACE_sparse.mat')
% 儲存工作空間
% save('d:/Work_Jupyter/敏/SIM_1y1x_sparse/PACE_workspace_sparse.mat')
