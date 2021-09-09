suppressMessages(library(zoo))
suppressMessages(library(chron))
suppressMessages(library(fpp2))
suppressMessages(library(config))
library(RPostgres)
library(DBI)


calcAnomalies <- function(station, channel,destdir){
    db_config <- config::get("akutan")
    con <- dbConnect(RPostgres::Postgres(),
                    user = db_config$user,
                    password = db_config$password,
                    dbname = db_config$database,
                    host = db_config$server)
                    
    SQL <- "SELECT freq_max10, sd_freq_max10, rsam, datetime as date
FROM data
WHERE station=(SELECT id FROM stations WHERE name=$1)
AND channel=$2
ORDER BY datetime"

    query <- dbSendQuery(con, SQL)
    dbBind(query, list(station, channel))
    data = dbFetch(query)
    dbClearResult(query)

    #data=data.frame(read.csv(file = "/Users/israel/Downloads/AUCH-BHZ-2021-06-29T00_00_00Z-2021-09-06T23_59_59Z.csv"))
    print("Beginning calc anomalies function")
    date_plot = c();corcoef1_low = c();corcoef1_up = c();corcoef1_estimate = c()
    date = chron(substr(as.character(data$date),1,10),format=c('y-m-d'))
    days = unique(date)

    for (i in seq(10,length(days), by = 1)){
        if (length(which(date==days[i]))>3600){
            training=data[which(date<days[i]),]
            test=data[which(date==days[i]),]

            a1 = hist(training$freq_max10, breaks = seq(0,20, by=0.1), plot='FALSE')
            a2 = hist(test$freq_max10, breaks = seq(0,20, by=0.1), plot = 'FALSE')
        
            test=cor.test(a1$density,a2$density,method = c("pearson"))
            corcoef1_low[i]=abs(test$conf.int[1])
            corcoef1_up[i]=abs(test$conf.int[2])
            corcoef1_estimate[i]=test$estimate
            date_plot[i]=days[i]
        }
    }
    dp_dates <- as.Date(date_plot)
    
    filename <- paste(
        destdir,
        '/',
        station,
        ".png",
        sep = ""
    )
    print(filename)
    
    png(file=filename, width=7.5, height = 4, units="in", res=600)
    plot(dp_dates,
         100 * (1 - corcoef1_estimate),
         type = "l",
         ylim = c(0, 100),
         xlim = c(dp_dates[10],
         dp_dates[length(date_plot)] + 7),
         ylab = "Anomaly level (%)",
         xlab = "")
    polygon(c(dp_dates, rev(dp_dates)),
            c(100 * (1 - corcoef1_low), rev(100 * (1 - corcoef1_up))),
            col = "#87CEEB", border = "black")

    set.seed(1)
    fit <- nnetar(100 * (1 - corcoef1_estimate), lambda = 0.7)
    fcast <- forecast(fit, PI = TRUE, h = 7)
    x_vector = (dp_dates[length(date_plot)] + 1):(dp_dates[length(date_plot)] + 7)

    polygon(c(x_vector, rev(x_vector)),
            c(fcast$lower[1:7,1], rev(fcast$upper[1:7, 1])),
            col = rgb(1, 0, 0, 0.3), border = "black")
    abline(h = 50, col = "gray", lty = 2)
    
    dev.off()
}

args <- commandArgs(trailingOnly = TRUE)

calcAnomalies(args[1], args[2], args[3])