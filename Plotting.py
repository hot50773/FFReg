import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D



def Plot_Mean(time_grid, fit_mean_func, Mean_Func, name, save_plt = False):
    '''
    --------------------------
    input
        ----------------------
        time_grid    : vector
        fit_mean_func: vector
        Mean_Func    : function
        name: string
    --------------------------
    output
        ----------------------
        plot of mean function
    --------------------------    
    '''
    fig = plt.figure(1)
    plt.title('mean ' + name)
    plt.plot(time_grid, fit_mean_func, '--b',label = 'fit')
    plt.plot(time_grid, Mean_Func(time_grid), 'r', label = 'real')
    leg = plt.legend(loc = 4)
    if save_plt:
        fig.savefig('mean_' + name + '.png')
    plt.show()

    
    
    
######

def Self_Cov_Func(s, t, Eigen_Funcs, eigen_val):
    '''
    -------------------------
    real covariance function
    -------------------------
    input
        ---------------------
        s: vector
        t: vector
        eigen_val  : vector
        Eigen_Funcs: function
        Cov_Func: function
    -------------------------
    output
        ---------------------
        cov: vector
    -------------------------
    '''
    if s.size != t.size:
        raise ValueError("length of s and t must be equal!")
    
    eigen_fns_1 = Eigen_Funcs(s)
    eigen_fns_2 = Eigen_Funcs(t)
    cov = np.zeros(eigen_fns_1.shape[1:])
    for i in range(eigen_val.shape[0]):
        cov += eigen_val[i] * eigen_fns_1[i] * eigen_fns_2[i]
    return(cov)






########
    
def Plot_Cov_of_X(time_grid, fitted_cov, Eigen_Funcs, X_eigen_val, name_of_X, z_lim ,Cov_Func = Self_Cov_Func, save_plt = False):
    '''
    -------------------------
    input
        ---------------------
        time_grid: vector
        X_eigen_val : vector
        ---------------------
        result_X: object
        Eigen_Funcs: function
        Cov_Func: function
        ---------------------
        name_of_X: string
        zlim: length-2 vector
    -------------------------
    output
        ---------------------
        plot1:Real Cov func
        plot2:Fitted Cov func
    -------------------------
    '''
    #reshape time grid to 2d
    t1, t2 = np.meshgrid(time_grid, time_grid)
    
    #plot real cov
    fig1 = plt.figure(1)
    ax = fig1.gca(projection='3d')
    surf = ax.plot_surface(t1, 
                           t2, 
                           Cov_Func(t1, t2, Eigen_Funcs, X_eigen_val),
                           rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0.1, antialiased=False)
    plt.suptitle('Real function of ' + name_of_X)
    if save_plt:
        plt.savefig('real_cov_'+ name_of_X + '.png')
    ax.set_zlim3d(z_lim[0], z_lim[1])
    
    #plot fit cov
    fig2 = plt.figure(2)
    ax = fig2.gca(projection='3d')
    ax.set_zlim3d(z_lim[0], z_lim[1])
    surf = ax.plot_surface(time_grid.repeat(time_grid.size).reshape(time_grid.size, time_grid.size), 
                           np.tile(time_grid, time_grid.size).reshape(time_grid.size, time_grid.size), 
                           fitted_cov,
                            rstride=1, cstride=1, cmap=cm.coolwarm,
                            linewidth=0.1, antialiased=False)
    plt.suptitle('LPR smoothing of ' + name_of_X)
    if save_plt:
        fig1.savefig('fit_cov_'+ name_of_X + '.png')
    plt.show()

    
def Plot_Cov_XY(time_grid_X, time_grid_Y, X_real_val_on_grid, Y_real_val_on_grid, 
                X_mean, Y_mean, fitted_cov, name_of_XY, z_lim, save_plt = False):
    '''
    -------------------------------------
    compare covriance
    (real cov computed by numerical method)
    -------------------------------------
    input
        ---------------------------------
        time_grid_X: len-k vector
        time_grid_Y: len-m vector
        ---------------------------------
        X_real_val_on_grid: (n * k) matrix
        Y_real_val_on_grid: (n * m) matrix
        ---------------------------------
        X_mean: len-k vector
        Y_mean: len-m vector
        ---------------------------------
        fitted_cov: (k * m) matrix
        name_of_X: string
        zlim: length-2 vector
    -------------------------------------
    output
        ---------------------------------
        plot1: Real   Cov func
        plot2: Fitted Cov func
    -------------------------------------
    '''
    #reshape time grid to 2d
    t1, t2 = np.meshgrid(time_grid_X, time_grid_Y)
    
    #compute cov of real value
    raw_XY = np.einsum('ij,ik->ijk',X_real_val_on_grid, Y_real_val_on_grid)
    
    real_cov_XY = np.mean(raw_XY, axis= 0) - np.outer(X_mean, Y_mean)
    
    #plot real cov
    fig1 = plt.figure(1)
    ax = fig1.gca(projection='3d')
    surf = ax.plot_surface(t1.T, 
                           t2.T, 
                           real_cov_XY,
                           rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0.1, antialiased=False)
    plt.suptitle('Real function of cov ' + name_of_XY)
    ax.set_zlim3d(z_lim[0], z_lim[1])
    if save_plt:
        fig1.savefig('real_cov_'+ name_of_XY + '.png')
    #plot fit cov
    fig2 = plt.figure(2)
    ax = fig2.gca(projection='3d')
    ax.set_zlim3d(z_lim[0], z_lim[1])
    surf = ax.plot_surface(t1.T, 
                           t2.T, 
                           fitted_cov,
                            rstride=1, cstride=1, cmap=cm.coolwarm,
                            linewidth=0.1, antialiased=False)
    plt.suptitle('LPR smoothing of cov ' + name_of_XY)
    if save_plt:
        fig2.savefig('fit_cov_'+name_of_XY+'.png')
    plt.show()

    
    
    
########

def Plot_Eigen_Funcs(time_grid, eigen_fun, Eigen_Funcs, num_fit_eigen, adjust_sign_of_eigen, name_of_X, save_plt = False):
    '''
    ----------------------------------------------
    input
        ------------------------------------------
        time_grid: vector
        eigen_fun: array
        X_eigen_val  : vector
        num_fit_eigen: scalar
        ------------------------------------------
        Eigen_Funcs: function
        ------------------------------------------
        adjust_sign_of_eigen: vector with 1 or -1
        name_of_X: string
    ----------------------------------------------
    output
        ------------------------------------------
        plot of eigen functions
    ----------------------------------------------
    '''
    real_eigen_on_grid = Eigen_Funcs(time_grid)
    num_real_eigen = real_eigen_on_grid.shape[0]
    num_eigen = np.min([num_real_eigen, num_fit_eigen])
    
    if adjust_sign_of_eigen.size != num_eigen:
        raise ValueError("'adjust_sign_of_eigen' should be length-" + str(num_eigen) + " vector")
            
    #plot real and fitted eigen functions
    for i in range(num_eigen):
        times = np.sqrt((((real_eigen_on_grid[i])**2).sum() * np.abs(time_grid[2] - time_grid[1])))       
        print(times)        
        fig = plt.figure(i + 1)
        plt.plot(time_grid, adjust_sign_of_eigen[i] * eigen_fun[i], '--b', label = 'Fit')
        plt.plot(time_grid, real_eigen_on_grid [i] / times, 'r', label = 'Real')
        plt.legend(loc = 4)
        plt.suptitle('Eigen Func' + str(i+1) + ' of ' + name_of_X)
        if save_plt:
            fig.savefig('Eig_Fun' + str(i+1) + '_of_'+ name_of_X + '.png')
    plt.show()

    
    
    
    

########


def Plot_FPCAResult(X_data, FpcaResult_X, Mean_Func, Eigen_Funcs, X_eigen_val,
                    adjust_sign_of_eigen, name_of_X = 'X', z_lim_of_cov = np.array([-10, 10]), 
                    Plot_Mean = Plot_Mean, Plot_Cov = Plot_Cov_of_X, Plot_Eigen_Funcs = Plot_Eigen_Funcs,
                    response = False, Plot_Cov_of_Response = Plot_Cov_XY, save_plt = False):
    '''
    ---------------------------------------------
    plots of 1. mean function
             2. covariance function
             3. eigen functions (if response = F)
    ---------------------------------------------
    input
        -----------------------------------------
        X_data: object 
        FpcaResult_X : object
        -----------------------------------------
        Mean_Func  : function
        Eigen_Funcs: function
        -----------------------------------------
        X_eigen_val:vector
        adjust_sign_of_eigen: vector
        -----------------------------------------
        z_lim_of_cov: length-2 vector
        name_of_X  :string
        -----------------------------------------
        Plot_Mean:function
        Plot_Cov :function
        Plot_Eigen_Funcs :function
        -----------------------------------------
        response = logical
        Plot_Cov_of_Response: function
    ---------------------------------------------
    output
        -----------------------------------------
        plots
    ---------------------------------------------
    '''

    Plot_Mean(time_grid = X_data.time_grid, 
              fit_mean_func = FpcaResult_X.mean_fun,
              Mean_Func = Mean_Func, 
              name = name_of_X,
              save_plt = save_plt)

    if response == False:
        Plot_Cov(time_grid   = X_data.time_grid, 
                 fitted_cov = FpcaResult_X.cov_fun, 
                 Eigen_Funcs = Eigen_Funcs,
                 X_eigen_val = X_eigen_val,
                 name_of_X = name_of_X,
                 z_lim = z_lim_of_cov,
                 save_plt = save_plt)

        Plot_Eigen_Funcs(time_grid = X_data.time_grid,
                         eigen_fun = FpcaResult_X.eig_fun,
                         Eigen_Funcs = Eigen_Funcs,
                         num_fit_eigen = FpcaResult_X.num_eig_pairs,
                         adjust_sign_of_eigen = adjust_sign_of_eigen,
                         name_of_X = name_of_X,
                         save_plt = save_plt)
    else:
        Plot_Cov_XY(time_grid_X = X_data.time_grid,
                    time_grid_Y = X_data.time_grid, 
                    X_real_val_on_grid = X_data.real_val_on_grid, 
                    Y_real_val_on_grid = X_data.real_val_on_grid, 
                    X_mean = Mean_Func(X_data.time_grid), 
                    Y_mean = Mean_Func(X_data.time_grid),
                    fitted_cov =  FpcaResult_X.cov_fun, 
                    name_of_XY = name_of_X,
                    z_lim = z_lim_of_cov,
                    save_plt = save_plt)

    
def Compare_Fit_and_Real(fit, real, name, z_lim, save_plt = False, plot_error = True, z_lim_of_error = np.array([-0.15, 0.15]), x_axis_range = None, y_axis_range = None):
    if fit.shape != real.shape:
        raise ValueError("dimension of fit and real must be the same !")
    if np.size(fit.shape) is not 2:
        raise ValueError("input fit must be matrix !")
    
    if ((x_axis_range is None) & (y_axis_range is None)):
        # create pseudo grid
        t1, t2 = np.meshgrid(np.arange(fit.shape[0]), np.arange(fit.shape[1]))
    else:
        t1, t2 = np.meshgrid(np.linspace(x_axis_range[0], x_axis_range[1], fit.shape[0]), 
                             np.linspace(y_axis_range[0], y_axis_range[1], fit.shape[1]))
    
    #plot real
    fig1 = plt.figure(1)
    ax = fig1.gca(projection='3d')
    surf = ax.plot_surface(t1.T, 
                           t2.T, 
                           fit,
                           rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0.1, antialiased=False)
    plt.suptitle('Fit_' + name)
    ax.set_zlim3d(z_lim[0], z_lim[1])
    if save_plt:
        fig1.savefig('fit_'+ name + '.png')

    #     plot fit
    fig2 = plt.figure(2)
    ax = fig2.gca(projection='3d')
    ax.set_zlim3d(z_lim[0], z_lim[1])
    surf = ax.plot_surface(t1.T, 
                           t2.T, 
                           real,
                           rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0.1, antialiased=False)
    
    plt.suptitle('Real_' + name)
    if save_plt:
        fig2.savefig('real_'+ name + '.png')
        
    if plot_error:
        #     plot error
        fig3 = plt.figure(3)
        ax = fig3.gca(projection='3d')
        ax.set_zlim3d(z_lim_of_error[0], z_lim_of_error[1])
        surf = ax.plot_surface(t1.T, 
                               t2.T, 
                               real - fit,
                               rstride=1, cstride=1, cmap=cm.coolwarm,
                               linewidth=0.1, antialiased=False)

        plt.suptitle('Error_of_' + name)
        if save_plt:
            fig3.savefig('error_'+ name + '.png')
    plt.show()    


#
def Plot_Restruct_Funcs(obs_time, obs_X, time_grid, fit_X, real_X, 
                        num_plot, y_lim, legend_location = 2,
                        save_plt = False, plot_obs = False):
    
    for i in range(num_plot):
        fig = plt.figure(i + 1)
        plt.ylim(y_lim[0], y_lim[1])
        plt.plot(time_grid, fit_X[i], 'b', label = 'fitted')
        plt.plot(time_grid,  real_X[i], '--r', label = 'real')
        if plot_obs:
            plt.plot(obs_time[i], obs_X[i], 'og', label = 'observation')
        plt.suptitle('Subject_'+ str(i+1))
        plt.legend(loc=legend_location, bbox_to_anchor=(1.05, 1))
        if save_plt:
            fig.savefig('Subject_'+ str(i+1) + '.png')
        
        plt.show()

