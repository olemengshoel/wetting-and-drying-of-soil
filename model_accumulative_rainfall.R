
#' @title Additive Exponential Accumulative rainfall model
#' @import GA
# run.index: used to prevent overwriting plot files in case 
  # of multiple runs
  # 0 input indicate no plot
#' @export
fit.AEAR.model.to.data <- function(data, 
                                  depth, 
                                  tau  = 72,
                                  max.iter = 10,
                                  run.index = 0,
                                  return_parameters = FALSE) {
  
  moisture <- data[[depth]]
  rain <- data$rainfall
  
  model <- function(params) {
    kd = 10 ^ params[1]
    kw = 10 ^ params[2]
    eta = params[3]
    kd2 = kd / params[4]
    
    saturation_awi = max(moisture)
    rain = rain / eta
    y.hat <- moisture
    
    for(t in (tau+1):length(moisture)) {
      rainfall_effect = rain[(t-tau):t] * (1-exp(-kw*(rev(0:tau)))) *
        exp(-kd*(rev(0:tau)))
      total_rainfall_effect = sum(rainfall_effect)
      y.hat[t] = drying_model_2_exp(moisture[t-tau], tau, kd, kd2, 
        total_rainfall_effect) + total_rainfall_effect;

      if((y.hat[t] > saturation_awi) ||
           (sum(rainfall_effect) == 0 
            && y.hat[t] > y.hat[t-1])) {
        # could be original or fitted
        # moisture or y.hat
        y.hat[t] = drying_model_2_exp(moisture[t-1], 1, kd, kd2,
          total_rainfall_effect)

      }
    }
    y.hat
  }
  
  fitness <- function(params) {
    y.hat <- model(params)
    mse <- mean(sqrt((moisture - y.hat) ^ 2))
    -mse
  }

  fitness.to.min <- function(params) {
    y.hat <- model(params)
    mse <- mean(sqrt((moisture - y.hat) ^ 2))
    mse
  }
  
  
  MinParams = c(-6, -1, 1, 1)
  MaxParams = c(0, 5, 4e3, 1e3)

if (0) {
  GAModel <- ga(type = "real-valued", 
                fitness = fitness, 
                monitor = NULL,
                min = MinParams, 
                max = MaxParams,
                maxiter = max.iter)
  fitted.parameters = GAModel@solution
  }
# -------------------------------------------------------------
if (0) {
  # print(max.iter)
  outGenSA <- GenSA(fn = fitness.to.min, lower = MinParams, 
    upper = MaxParams, control = list(maxit = 4 * max.iter))
  fitted.parameters = outGenSA$par
}
# -------------------------------------------------------------
if (1) {
outDEoptim <- DEoptim(fn = fitness.to.min, lower = MinParams, 
  upper = MaxParams, control = DEoptim.control(itermax = max.iter, 
    trace = FALSE))
  fitted.parameters = tail(outDEoptim$member$bestmemit,1)
}
# -------------------------------------------------------------
if (0) {
  hook.run = function(experiment, config = list()) {
    candidate <- experiment$candidate
    x = c(candidate[["x1"]], candidate[["x2"]], candidate[["x3"]], candidate[["x4"]])
    # print(x)
    y = fitness.to.min(x)
    y
  }
  parameters.table <- paste('x1 "" r (', MinParams[1], ', ', MaxParams[1], 
    ')\n x2 "" r (', MinParams[2], ', ', MaxParams[2], 
    ')\n x3 "" r (', MinParams[3], ', ', MaxParams[3],
    ')\n x4 "" r (', MinParams[4], ', ', MaxParams[4], ')')
  parameters <- readParameters(text=parameters.table)
  print('Read parameters')
  irace.weights <- rnorm(200, mean = 0.9, sd = 0.02)
  expt.budget = max(max.iter, 210)
  print(expt.budget)
  result <- irace(tunerConfig = list(
    hookRun = hook.run,
    instances = irace.weights[1:100],
    maxExperiments = expt.budget,
    # nbIterations = 1,
    logFile = ""),
    parameters = parameters)

  fitted.parameters = c(result[1,]$x1, result[1,]$x2, result[1,]$x3, result[1,]$x4)
}
# -------------------------------------------------------------
  if (return_parameters) {
    return(fitted.parameters)
  }

  y.hat <- model(fitted.parameters)
  standard.error = mean((y.hat - moisture) ^ 2)
  max.abs.error = max(abs(y.hat - moisture))

  
  print("fitted.parameters:")
  print(fitted.parameters)
  print(paste("SE, MaxE = ", standard.error, max.abs.error)) 
 
  errors = c(standard.error, max.abs.error)

  fitted.parameters
}


#' @title Naive Accumulative rainfall model
#' @import GA
# run.index: used to prevent overwriting plot files in case 
  # of multiple runs
  # 0 input indicate no plot
#' @export
fit.NAR.model.to.data <- function(data, 
                                 depth, 
                                 tau  = 72,
                                 max.iter = 10,
                                 run.index = 0,
                                 return_parameters = FALSE) {

  moisture <- data[[depth]]
  rain <- data$rainfall
  
  model <- function(params) {
    kd = 10 ^ params[1]
    kw = 10 ^ params[2]
    eta = params[3]
    # kd2 = kd / params[4]
    
    saturation_awi = max(moisture)
    rain = rain / eta
    y.hat <- moisture
    
    for(t in (tau+1):length(moisture)) {
      rainfall_effect = rain[(t-tau):t] * (1-exp(-kw*(rev(0:tau)))) *
        exp(-kd*(rev(0:tau)))
      
      y.hat[t] = drying_model_single_exp(
        moisture[t-tau], tau, kd) + sum(rainfall_effect);

      if((y.hat[t] > saturation_awi) ||
           (sum(rainfall_effect) == 0 
            && y.hat[t] > y.hat[t- 1])) {
        # could be original or fitted
        # moisture or y.hat
        y.hat[t] = drying_model_single_exp(moisture[t- 1], 1, kd)
      }
    }
    y.hat
  }
  
  fitness <- function(params) {
    y.hat <- model(params)
    mse <- mean(sqrt((moisture - y.hat) ^ 2))
    -mse
  }

  fitness.to.min <- function(params) {
    y.hat <- model(params)
    mse <- mean(sqrt((moisture - y.hat) ^ 2))
    mse
  }
  
  
  MinParams = c(-6, -1, 1, 1)
  MaxParams = c(0, 5, 4e3, 1e3)

# -------------------------------------------------------------
if (0) {
  GAModel <- ga(type = "real-valued", 
                fitness = fitness, 
                monitor = NULL,
                min = MinParams, 
                max = MaxParams,
                maxiter = max.iter)
  fitted.parameters = GAModel@solution
  }
# -------------------------------------------------------------
if (0) {
  outGenSA <- GenSA(fn = fitness.to.min, lower = MinParams, 
    upper = MaxParams, control = list(maxit = 4 * max.iter))
  fitted.parameters = outGenSA$par
}
# -------------------------------------------------------------
if (1) {
outDEoptim <- DEoptim(fn = fitness.to.min, lower = MinParams, 
  upper = MaxParams, control = DEoptim.control(itermax = max.iter, 
    trace = FALSE))
  fitted.parameters = tail(outDEoptim$member$bestmemit,1)
}
# -------------------------------------------------------------
if (0) {
  hook.run = function(experiment, config = list()) {
    candidate <- experiment$candidate
    x = c(candidate[["x1"]], candidate[["x2"]], candidate[["x3"]])
    # print(x)
    y = fitness.to.min(x)
    y
  }
  parameters.table <- paste('x1 "" r (', MinParams[1], ', ', MaxParams[1], 
    ')\n x2 "" r (', MinParams[2], ', ', MaxParams[2], 
    ')\n x3 "" r (', MinParams[3], ', ', MaxParams[3], ')')
  parameters <- readParameters(text=parameters.table)
  print('Read parameters')
  irace.weights <- rnorm(200, mean = 0.9, sd = 0.02)
  expt.budget = max(max.iter, 210)
  print(expt.budget)
  result <- irace(tunerConfig = list(
    hookRun = hook.run,
    instances = irace.weights[1:100],
    maxExperiments = expt.budget,
    # nbIterations = 1,
    logFile = ""),
    parameters = parameters)

  fitted.parameters = c(result[1,]$x1, result[1,]$x2, result[1,]$x3)
}
# -------------------------------------------------------------
  if (return_parameters) {
    return(fitted.parameters)
  }

  y.hat <- model(fitted.parameters)
  standard.error = mean((y.hat - moisture) ^ 2)
  max.abs.error = max(abs(y.hat - moisture))

  print(fitted.parameters)
  print(paste("SE, MaxE = ", standard.error, max.abs.error)) 
 
  errors = c(standard.error, max.abs.error)

  errors
}


#' @title NAR.model.prediction
#' @export
AEAR.model.prediction <- function(params, data, depth, tau=72) {
  moisture <- data[[depth]]
  rain <- data$rainfall
    
    kd = 10 ^ params[1]
    kw = 10 ^ params[2]
    eta = params[3]
    kd2 = kd / params[4]
    
    saturation_awi = max(moisture)
    rain = rain / eta
    y.hat <- moisture
    
    for(t in (tau+1):length(moisture)) {
      rainfall_effect = rain[(t-tau):t] * (1-exp(-kw*(rev(0:tau)))) *
        exp(-kd*(rev(0:tau)))
      total_rainfall_effect = sum(rainfall_effect)
      y.hat[t] = drying_model_2_exp(moisture[t-tau], tau, kd, kd2, 
        total_rainfall_effect) + total_rainfall_effect;
      
      if((y.hat[t] > saturation_awi) ||
           (sum(rainfall_effect) == 0 
            && y.hat[t] > y.hat[t-1])) {
        # could be original or fitted
        # moisture or y.hat
        y.hat[t] = drying_model_2_exp(moisture[t-1], 1, kd, kd2,
          total_rainfall_effect)
      }
    }
    y.hat
  }


#' @title AEAR.model.prediction.irregular
#' @export
AEAR.model.prediction.irregular <- function(params, data, depth, tau=72) {
  moisture <- data[[depth]]
  rain <- data$rainfall
    
    kd = 10 ^ params[1]
    kw = 10 ^ params[2]
    eta = params[3]
    kd2 = kd / params[4]
    
    saturation_awi = max(moisture)
    rain = rain / eta
    y.hat <- moisture
    
    for(t in (tau+1):length(moisture)) {
      rainfall_effect = rain[(t-tau):t] * (1-exp(-kw*(rev(0:tau)))) *
        exp(-kd*(rev(0:tau)))
      total_rainfall_effect = sum(rainfall_effect)
      y.hat[t] = drying_model_2_exp(y.hat[t-tau], tau, kd, kd2, 
        total_rainfall_effect) + total_rainfall_effect;
      
      if((y.hat[t] > saturation_awi) ||
           (sum(rainfall_effect) == 0 
            && y.hat[t] > y.hat[t-1])) {
        # could be original or fitted
        # moisture or y.hat
        y.hat[t] = drying_model_2_exp(y.hat[t-1], 1, kd, kd2,
          total_rainfall_effect)
      }
    }
    y.hat
  }


#' @title NAR.model.prediction
#' @export
NAR.model.prediction <- function(params, data, depth, tau=72) {
  moisture <- data[[depth]]
  rain <- data$rainfall
  
    kd = 10 ^ params[1]
    kw = 10 ^ params[2]
    eta = params[3]
    
    saturation_awi = max(moisture)
    rain = rain / eta
    y.hat <- moisture
    
    for(t in (tau+1):length(moisture)) {
      rainfall_effect = rain[(t-tau):t] * (1-exp(-kw*(rev(0:tau)))) *
        exp(-kd*(rev(0:tau)))
      
      y.hat[t] = drying_model_single_exp(
        moisture[t-tau], tau, kd) + sum(rainfall_effect);
      
      if((y.hat[t] > saturation_awi) ||
           (sum(rainfall_effect) == 0 
            && y.hat[t] > y.hat[t- 1])) {
        # could be original or fitted
        # moisture or y.hat
        y.hat[t] = drying_model_single_exp(moisture[t- 1], 1, kd)
      }
    }
    y.hat
  }

#' @title drying_model_2_exp
drying_model_2_exp <- function(moisture_t, del_t, kd, kd2, rain_effect) {
    w_steep = max(0.1, min(0.9, rain_effect*10));
    w_flat = 1 - w_steep;
    y = moisture_t * (w_steep * exp(- kd * del_t) + 
             w_flat * exp(- kd2 * del_t));
    y
  }

#' @title drying_model_single_exp
drying_model_single_exp <- function(moisture_t, del_t, kd) {
    y = moisture_t * exp(- kd * del_t);
    y
  }


#' @title subsample.data
#' @export
subsample.data <- function(data, subsampling.rate = 10) {
  d = data[seq(1, nrow(data), subsampling.rate), ]
  rain = data$rainfall
  accum.rain = cumsum(rain)
  sub.accum.rain = accum.rain[seq(1, length(accum.rain), subsampling.rate)]
  l = length(sub.accum.rain)
  sub.rain = c(0, sub.accum.rain[-1] - sub.accum.rain[-l])
  d$rain = sub.rain

  d
}