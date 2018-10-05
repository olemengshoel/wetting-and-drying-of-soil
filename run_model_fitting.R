
source("model_accumulative_rainfall.R")

# library(GA)
# library(GenSA)
library(DEoptim)
library(gplots)

num.runs = 1
max.iter.per.run = 2
data.directory = "data/";

d =  read.csv(paste(data.directory, "non-leaky-bukcet-data.csv", sep = ""), stringsAsFactors=FALSE)
# d$date <- as.POSIXct(d$date, format = "%Y-%m-%d %H:%M:%S")
d$date <- d[['Measurement.Time']]
d$date <- as.POSIXct(d$date, format = "%m/%d/%Y %I:%M %p")

# Remove the first part of the time-series, to remove noisy measurements. 
d2 = d[6e3:3.6e4,]
d2.s = subsample.data(d2, 10)
# d2.s = d

erros.runtimes.all.runs = matrix(0, nrow = num.runs, ncol = 3)
for (i in 1:num.runs) {
	start.time <- Sys.time()
	# errors = fit.NAR.model.to.data(data = d2.s, depth = "X5cm", 
	# tau = 72, max.iter = max.iter.per.run, i)

	errors = fit.AEAR.model.to.data(data = d2.s, depth = "D..28cm", 
	tau = 72, max.iter = max.iter.per.run, 0)

	end.time <- Sys.time() 
	run.time <- diff(c(start.time, end.time))
	units(run.time) <- "mins"
	
	erros.runtimes.all.runs[i, ] = c(errors, as.numeric(run.time))
}

print(erros.runtimes.all.runs)

