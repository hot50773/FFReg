library("refund")


refundFPC_simulation <- function(t_in,
                              X_obs_tran_in,
                              Y_obs_tran_in,
                              X_obs_test_in,
                              Y_real_grid_val_in,
                              rankX,
                              grid_width_in,
                              plot_result = FALSE,
                              num_plot = 5)
{ 
  if(dim(X_obs_tran_in)[2] != dim(Y_obs_tran_in)[2])
    stop("sample size of X_obs_tran_in and Y_obs_tran_in are not same !")
  
  tran_samp_size <- dim(X_obs_tran_in)[2]
  test_samp_size <- dim(X_obs_test_in)[2]
  total_samp_size <- tran_samp_size + test_samp_size
  X_obs_tran_in <- t(X_obs_tran_in)
  Y_obs_tran_in <- t(Y_obs_tran_in)
  
  m.pc <- pffr(Y_obs_tran_in ~  ffpc(X_obs_tran_in, yind=t_in, 
                                  decomppars=list(npc=rankX)), yind=t_in)
  fit_test_Y  <- predict(m.pc, newdata = list(X_obs_tran_in = t(X_obs_test_in)))
  real_test_Y <- t(Y_real_grid_val_in)[(tran_samp_size + 1) : total_samp_size, ]
  MISE <- mean(apply((fit_test_Y - real_test_Y)^2 * grid_width_in, 1, sum))
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

refundFF_simulation <- function(t_in,
                                s_in,
                                 X_obs_tran_in,
                                 Y_obs_tran_in,
                                 X_obs_test_in,
                                 Y_real_grid_val_in,
                                 grid_width_in,
                                 plot_result = FALSE, 
                                 num_plot = 5
                                 )
{ 
  if(dim(X_obs_tran_in)[2] != dim(Y_obs_tran_in)[2])
    stop("sample size of X_obs_tran_in and Y_obs_tran_in are not same !")
  
  tran_samp_size <- dim(X_obs_tran_in)[2]
  test_samp_size <- dim(X_obs_test_in)[2]
  total_samp_size <- tran_samp_size + test_samp_size
  X_obs_tran_in <- t(X_obs_tran_in)
  Y_obs_tran_in <- t(Y_obs_tran_in)
  
  model <- pffr(Y_obs_tran_in ~  ff(X_obs_tran_in, xind=s_in), yind=t_in)
  fit_test_Y  <- predict(model, newdata = list(X_obs_tran_in = t(X_obs_test_in)))
  real_test_Y <- t(Y_real_grid_val_in)[(tran_samp_size + 1) : total_samp_size, ]
  MISE <- mean(apply((fit_test_Y - real_test_Y)^2 * grid_width_in, 1, sum))
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
X_num_grid <- 41
Y_num_grid <- 41
time_grid_Y <- seq(0, 1, l=Y_num_grid)
time_grid_X <- time_grid_Y
origin_up_bound_Y <- 5
grid_width <- origin_up_bound_Y / (Y_num_grid - 1)

set.seed(2017)
refundFPC_MISE <- numeric(sim_rep_time)
refundFF_MISE <- numeric(sim_rep_time)
X_obs_tran <- list()
Y_obs_tran <- list()
X_obs_test <- list()
Y_real_grid_val <- list()
for(i in 1:sim_rep_time){
  X_obs_tran[[i]] <- read.csv(paste('d://Work_Jupyter//±Ó//SIM_1y1x_grid1_size600//tran X of samp',i), 
                        quote = '\n', sep = ' ', header = F)
  Y_obs_tran[[i]] <- read.csv(paste('d://Work_Jupyter//±Ó//SIM_1y1x_grid1_size600//tran Y of samp',i), 
                        quote = '\n', sep = ' ', header = F)
  X_obs_test[[i]] <- read.csv(paste('d://Work_Jupyter//±Ó//SIM_1y1x_grid1_size600//test X of samp',i), 
                              quote = '\n', sep = ' ', header = F)
  Y_real_grid_val[[i]] <- read.csv(paste('d://Work_Jupyter//±Ó//SIM_1y1x_grid1_size600//real_on_grid Y of samp',i), 
                             quote = '\n', sep = ' ', header = F)
  refundFPC_MISE[i] <- refundFPC_simulation(t_in = time_grid_Y,
                                      X_obs_tran_in = X_obs_tran[[i]],
                                      Y_obs_tran_in = Y_obs_tran[[i]],
                                      X_obs_test_in = X_obs_test[[i]],
                                      Y_real_grid_val_in = Y_real_grid_val[[i]],
                                      grid_width_in = grid_width,
                                      rankX = 3)
  refundFF_MISE[i] <- refundFF_simulation(t_in = time_grid_Y,
                                          s_in = time_grid_X,
                                          X_obs_tran_in = X_obs_tran[[i]],
                                          Y_obs_tran_in = Y_obs_tran[[i]],
                                          X_obs_test_in = X_obs_test[[i]],
                                          Y_real_grid_val_in = Y_real_grid_val[[i]],
                                          grid_width_in = grid_width)
}

write(refundFPC_MISE, 'd://Work_Jupyter//±Ó//SIM_1y1x_grid1_size600//refundFPC_MISE_600size', sep = ' ')
write(refundFF_MISE, 'd://Work_Jupyter//±Ó//SIM_1y1x_grid1_size600//refundFF_MISE_600size', sep = ' ')

#refund_MISE <- read.csv(paste('d://Work_Jupyter//±Ó//SIM_1y1x_grid1_size600//refund_MISE_600size'), sep = ' ', header = F)
