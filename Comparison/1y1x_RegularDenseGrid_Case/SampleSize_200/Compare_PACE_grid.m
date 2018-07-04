X_time_grid = load('d:/Work_Jupyter/敏/SIM_1y1x_grid1/sim_X_time_grid.mat');
Y_time_grid = load('d:/Work_Jupyter/敏/SIM_1y1x_grid1/sim_Y_time_grid.mat');
num_train = 200;
num_test = 100;
num_sample = num_train + num_test;
dt = Y_time_grid.grid(2) - Y_time_grid.grid(1);
MISE  = zeros(1,100);
RMSPE = zeros(1,100);
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
% options = setOptions('regular',0,'verbose','on','bwmu_gcv',0,'bwxcov_gcv',0);
param_X = setOptions('selection_k','FVE', 'regular', 2, 'newdata', X_time_grid.grid, 'verbose','off','bwmu_gcv',0,'bwxcov_gcv',0);
param_Y = setOptions('selection_k','FVE', 'regular', 2, 'newdata', Y_time_grid.grid, 'verbose','off','bwmu_gcv',0,'bwxcov_gcv',0);
npred = num_test; % 資料理有幾筆是要拿來test的

for j = 1:100

    X_data = load(['d:/Work_Jupyter/敏/SIM_1y1x_grid1/sim_X_data_',num2str(j),'.mat']);
    Y_data = load(['d:/Work_Jupyter/敏/SIM_1y1x_grid1/sim_Y_data_',num2str(j),'.mat']);



    % 因為FPCreg輸入不能有NA，要先把NA弄掉
    % 且輸入必須是一個1 * num_fun的cell (matlab的資料格式)，所以創新的cell存t和obs
    new_Xt = cell(1, num_sample);
    new_Xobs = cell(1, num_sample);
    % non_nan_X = ~isnan(X_data.obs);

    for i = 1:num_sample
    %     non_nan_Xi = non_nan_X(i, :);
        new_Xt{i} = X_data.t(i, :);
        new_Xobs{i} = X_data.obs(i, :);
    end

    % 同理，以下做Y的部分
    new_Yt = cell(1, num_sample);
    new_Yobs = cell(1, num_sample);
    % non_nan_Y = ~isnan(Y_data.obs);

    for i = 1:num_sample
    %     non_nan_Yi = non_nan_Y(i, :);
        new_Yt{i} = Y_data.t(i, :);
        new_Yobs{i} = Y_data.obs(i, :);
    end


% 他的Regression函式: FPCreg(x, t_x, y, t_y,param_X, param_Y,FIT,K_x,K_y,npred,alpha); %Functional regression
% 可以參考 http://www.stat.ucdavis.edu/PACE/Help/FPCregHELP.txt
    [res, xx, yy] = FPCreg(new_Xobs, new_Xt, new_Yobs, new_Yt,param_X, param_Y,0,[],[],npred);

    fit_test_Y = getVal(res,'newy');           % 這個目前用不到
    fit_test_Yt = getVal(res,'new_ty');          
%     fitted_Y = getVal(res, 'fitted_y');
    fit_test_Y = transpose(reshape(cell2mat(fit_test_Y), 21, 100));
    
    MISE(1,j) = mean(sum((Y_data.real_test_Y - fit_test_Y).^2 * dt, 2));
    RMSPE(1,j) = mean(sum((Y_data.real_test_Y - fit_test_Y).^2 * dt, 2) ./ sum((Y_data.real_test_Y).^2 * dt, 2));
end

% save MISE and RMSPE as structure
save('d:/Work_Jupyter/敏/SIM_1y1x_grid1/result_PACE_grid.mat', 'MISE','RMSPE');

obs_test_Y = Y_data.obs((num_train + 1):num_sample, :);
% 畫前四張圖
figure;
for i = 1:4
    subplot(2, 2, i);
    plot(Y_time_grid.grid, obs_test_Y(i,:), 'go');
    hold on
    plot(Y_time_grid.grid, fit_test_Y(i, :),'r--');
    title(['Subject ' num2str(i)]);
        plot(Y_time_grid.grid, Y_data.real_test_Y(i,:),'b--');
    title(['Subject ' num2str(i)]);
    xlabel('t');
    ylabel('Y(t)');
    legend('observed','pred','real');
end
FPCreg_grid_workspace =  load('d:/Work_Jupyter/敏/SIM_1y1x_grid1/PACE_workspace_grid.mat')
% save('d:/Work_Jupyter/敏/SIM_1y1x_grid1/PACE_workspace_grid.mat')
