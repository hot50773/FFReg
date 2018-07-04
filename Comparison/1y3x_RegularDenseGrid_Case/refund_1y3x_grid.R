library("refund")

refundFPC_simulation_3X <- function(t_in,
                                    Y_grid_width_in,
                                    X_obs_tran_list_in,
                                    Y_obs_tran_in,
                                    X_obs_test_list_in,
                                    Y_real_grid_val_in,
                                    rankX,
                                    plot_result = FALSE, 
                                    num_plot = 5)
{ 
  if ((length(X_obs_tran_list_in)!=3) || (length(X_obs_test_list_in)!=3))
    stop("length of X_obs_tran_list_in and X_obs_test_list_in must be 3!")
  for (j in 1:3){
    if(dim(X_obs_tran_list_in[[j]])[2] != dim(Y_obs_tran_in)[2])
      stop("tran sample size must be same!")
  }
  if ((dim(X_obs_test_list_in[[1]])[2]!= dim(X_obs_test_list_in[[2]])[2]) || 
      (dim(X_obs_test_list_in[[1]])[2]!= dim(X_obs_test_list_in[[3]])[2]))
    stop("test samp size differ !")
  tran_samp_size <- dim(Y_obs_tran_in)[2]
  test_samp_size <- dim(X_obs_test_list_in[[1]])[2]
  total_samp_size <- tran_samp_size + test_samp_size
  
  X1_obs_tran_in <- t(X_obs_tran_list_in[[1]])
  X2_obs_tran_in <- t(X_obs_tran_list_in[[2]])
  X3_obs_tran_in <- t(X_obs_tran_list_in[[3]])
  Y_obs_tran_in <- t(Y_obs_tran_in)
  
  m.pc <- pffr(Y_obs_tran_in ~  
                 ffpc(X1_obs_tran_in, yind=t_in, decomppars=list(npc=rankX)) +
                 ffpc(X2_obs_tran_in, yind=t_in, decomppars=list(npc=rankX)) + 
                 ffpc(X3_obs_tran_in, yind=t_in, decomppars=list(npc=rankX)),
               yind=t_in)
  
  fit_test_Y  <- predict(m.pc, 
                         newdata = list(X1_obs_tran_in = t(X_obs_test_list_in[[1]]),
                                        X2_obs_tran_in = t(X_obs_test_list_in[[2]]),
                                        X3_obs_tran_in = t(X_obs_test_list_in[[3]])))
  real_test_Y <- t(Y_real_grid_val_in)[(tran_samp_size + 1) : total_samp_size, ]
  MISE <- mean(apply((fit_test_Y - real_test_Y)^2 * Y_grid_width_in, 1, sum))
  if(plot_result){
    for(j in 1:num_plot){
      windows()
      plot(t_in, fit_test_Y[j, ], type = 'l', col = 'blue', ylim = c(-15,15))
      lines(t_in, real_test_Y[j, ], type = 'l', col = 'red', ylim = c(-15,15))
      legend(x = 0.8, y = min(fit_test_Y[j, ]) + 1, legend = c('fit', 'real'),
             lty = 1, col = c('blue', 'red'))
    }
  }
  return(MISE)
}

refundFF_simulation_3X <- function(t_in,
                                   s_list_in,
                                   Y_grid_width_in,
                                   X_obs_tran_list_in,
                                   Y_obs_tran_in,
                                   X_obs_test_list_in,
                                   Y_real_grid_val_in,
                                   plot_result = FALSE, 
                                   num_plot = 5
                                 )
{ 
  if ((length(X_obs_tran_list_in)!=3) || (length(X_obs_test_list_in)!=3) ||
      (length(s_list_in)!=3))
      stop("length of X_obs_tran_list_in and X_obs_test_list_in must be 3!")
  for (j in 1:3){
    if(dim(X_obs_tran_list_in[[j]])[2] != dim(Y_obs_tran_in)[2])
      stop("tran sample size differ !")
  }
  if ((dim(X_obs_test_list_in[[1]])[2]!= dim(X_obs_test_list_in[[2]])[2]) || 
      (dim(X_obs_test_list_in[[1]])[2]!= dim(X_obs_test_list_in[[3]])[2]))
    stop("test samp size differ !")
  
  tran_samp_size <- dim(Y_obs_tran_in)[2]
  test_samp_size <- dim(X_obs_test_list_in[[1]])[2]
  total_samp_size <- tran_samp_size + test_samp_size
  
  X1_obs_tran_in <- t(X_obs_tran_list_in[[1]])
  X2_obs_tran_in <- t(X_obs_tran_list_in[[2]])
  X3_obs_tran_in <- t(X_obs_tran_list_in[[3]])
  Y_obs_tran_in <- t(Y_obs_tran_in)
  
  model <- pffr(Y_obs_tran_in ~  
                  ff(X1_obs_tran_in, xind = s_list_in[[1]]) + 
                  ff(X2_obs_tran_in, xind = s_list_in[[2]]) + 
                  ff(X3_obs_tran_in, xind = s_list_in[[3]]), 
                yind=t_in)
  fit_test_Y  <- predict(model, 
                         newdata = list(X1_obs_tran_in = t(X_obs_test_list_in[[1]]),
                                        X2_obs_tran_in = t(X_obs_test_list_in[[2]]),
                                        X3_obs_tran_in = t(X_obs_test_list_in[[3]])))
  real_test_Y <- t(Y_real_grid_val_in)[(tran_samp_size + 1) : total_samp_size, ]
  MISE <- mean(apply((fit_test_Y - real_test_Y)^2 * Y_grid_width_in, 1, sum))
  if(plot_result){
    for(j in 1:num_plot){
      windows()
      plot(t_in, fit_test_Y[j, ], type = 'l', col = 'blue', ylim = c(-15,15))
      lines(t_in, real_test_Y[j, ], type = 'l', col = 'red', ylim = c(-15,15))
      legend(x = 0.8, y = min(fit_test_Y[j, ]) + 1, legend = c('fit', 'real'),
             lty = 1, col = c('blue', 'red'))
    }
  }
  return(MISE)
}


sim_rep_time <- 100
X_num_grid <- 21
Y_num_grid <- 21
time_grid_Y <- seq(0, 1, l=Y_num_grid)
time_grid_X <- time_grid_Y
time_grid_X_list <- list(time_grid_X, time_grid_X, time_grid_X)
# origin upper bound of time Y
Y_origin_upbound <- 5
# origin grid width
Y_grid_width <- Y_origin_upbound / (Y_num_grid - 1)

set.seed(2017)
refundFPC_MISE <- numeric(sim_rep_time)
refundFF_MISE <- numeric(sim_rep_time)
#refundFPC_MISE[1] <- refundFPC_simulation(i = 1, plot_result = TRUE)
X1_obs_tran <- list()
X2_obs_tran <- list()
X3_obs_tran <- list()
Y_obs_tran <- list()
X1_obs_test <- list()
X2_obs_test <- list()
X3_obs_test <- list()
Y_real_grid_val <- list()
for(i in 1:sim_rep_time){
  X1_obs_tran[[i]] <- read.csv(paste('d://Work_Jupyter//±Ó//SIM_1y3x_grid//tran X1 of samp',i), 
                        quote = '\n', sep = ' ', header = F)
  X2_obs_tran[[i]] <- read.csv(paste('d://Work_Jupyter//±Ó//SIM_1y3x_grid//tran X2 of samp',i), 
                              quote = '\n', sep = ' ', header = F)
  X3_obs_tran[[i]] <- read.csv(paste('d://Work_Jupyter//±Ó//SIM_1y3x_grid//tran X3 of samp',i), 
                              quote = '\n', sep = ' ', header = F)
  Y_obs_tran[[i]] <- read.csv(paste('d://Work_Jupyter//±Ó//SIM_1y3x_grid//tran Y of samp',i), 
                        quote = '\n', sep = ' ', header = F)
  X1_obs_test[[i]] <- read.csv(paste('d://Work_Jupyter//±Ó//SIM_1y3x_grid//test X1 of samp',i),
                              quote = '\n', sep = ' ', header = F)
  X2_obs_test[[i]] <- read.csv(paste('d://Work_Jupyter//±Ó//SIM_1y3x_grid//test X2 of samp',i),
                              quote = '\n', sep = ' ', header = F)
  X3_obs_test[[i]] <- read.csv(paste('d://Work_Jupyter//±Ó//SIM_1y3x_grid//test X3 of samp',i),
                              quote = '\n', sep = ' ', header = F)
  Y_real_grid_val[[i]] <- read.csv(paste('d://Work_Jupyter//±Ó//SIM_1y3x_grid//real_on_grid Y of samp',i), 
                             quote = '\n', sep = ' ', header = F)
  refundFPC_MISE[i] <- refundFPC_simulation_3X(t_in = time_grid_Y,
                                               Y_grid_width_in = Y_grid_width,
                                               X_obs_tran_list_in = list(X1_obs_tran[[i]],
                                                                         X2_obs_tran[[i]],
                                                                         X3_obs_tran[[i]]),
                                               Y_obs_tran_in = Y_obs_tran[[i]],
                                               X_obs_test_list_in = list(X1_obs_test[[i]],
                                                                         X2_obs_test[[i]],
                                                                         X3_obs_test[[i]]),
                                               Y_real_grid_val_in = Y_real_grid_val[[i]],
                                               rankX = 3)
  refundFF_MISE[i] <- refundFF_simulation_3X(t_in = time_grid_Y,
                                             s_list_in = time_grid_X_list,
                                             Y_grid_width_in = Y_grid_width,
                                             X_obs_tran_list_in = list(X1_obs_tran[[i]],
                                                                        X2_obs_tran[[i]],
                                                                        X3_obs_tran[[i]]),
                                             Y_obs_tran_in = Y_obs_tran[[i]],
                                             X_obs_test_list_in = list(X1_obs_test[[i]],
                                                                       X2_obs_test[[i]],
                                                                       X3_obs_test[[i]]),
                                             Y_real_grid_val_in = Y_real_grid_val[[i]])
}


write(refundFPC_MISE, 'd://Work_Jupyter//±Ó//SIM_1y3x_grid//refundFPC_MISE', sep = ' ')
write(refundFF_MISE, 'd://Work_Jupyter//±Ó//SIM_1y3x_grid//refundFF_MISE', sep = ' ')
mean(refundFF_MISE)
mean(refundFPC_MISE)
#refund_MISE <- read.csv(paste('d://Work_Jupyter//±Ó//SIM_1y3x_grid//refund_MISE'), sep = ' ', header = F)
