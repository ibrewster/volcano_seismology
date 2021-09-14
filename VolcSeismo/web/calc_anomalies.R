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
    date_plot=c();
    corcoef1_low_longterm=c();corcoef1_up_longterm=c();corcoef1_estimate_longterm=c()
    corcoef1_low_shortterm=c();corcoef1_up_shortterm=c();corcoef1_estimate_shortterm=c()
    date=chron(substr(as.character(data$date),1,10),format=c('y-m-d'))
    days=unique(date)

    for(i in seq(10,length(days), by=1)){
        if(length(which(date==days[i]))>3600){
            
            training_longterm=data[which(date<days[i]),]
            training_shortterm=data[which(date>=(days[i]-3) & date<days[i]),]
            test=data[which(date==days[i]),]

            a1_longterm=hist(training_longterm$freq_max10[which(training_longterm$freq_max10>=0.5)], breaks = seq(0.5,20, by=0.1), plot='FALSE')
            a1_shortterm=hist(training_shortterm$freq_max10[which(training_shortterm$freq_max10>=0.5)], breaks = seq(0.5,20, by=0.1), plot='FALSE')     
            a2=hist(test$freq_max10[which(test$freq_max10>=0.5)], breaks = seq(0.5,20, by=0.1), plot='FALSE')     

            test_longterm=cor.test(a1_longterm$density,a2$density,method = c("pearson"))
            corcoef1_low_longterm[i]=abs(test_longterm$conf.int[1])
            corcoef1_up_longterm[i]=abs(test_longterm$conf.int[2])
            corcoef1_estimate_longterm[i]=test_longterm$estimate
        
            test_shortterm=cor.test(a1_shortterm$density,a2$density,method = c("pearson"))
            corcoef1_low_shortterm[i]=abs(test_shortterm$conf.int[1])
            corcoef1_up_shortterm[i]=abs(test_shortterm$conf.int[2])
            corcoef1_estimate_shortterm[i]=test_shortterm$estimate        
            
            date_plot[i]=days[i]
        }
    }

    filename <- paste(
        destdir,
        '/',
        station,
        "-long.png",
        sep = ""
    )

    png(file=filename, width=7.5, height = 4, units="in", res=600)

    dp_asdate=as.Date(date_plot)

    par(mar=c(2,4.1,.5,2.1))

    plot(dp_asdate,100*(1-corcoef1_estimate_longterm), 
         type = "l",ylim = c(0,100),
         xlim=c(dp_asdate[10], dp_asdate[length(date_plot)]), 
         ylab = "Long-term Anomaly (%)",xlab="")

    polygon(c(dp_asdate, rev(dp_asdate)), 
            c(100*(1-corcoef1_low_longterm), rev(100*(1-corcoef1_up_longterm))),
            col = "#87CEEB", border='black')
    abline(h=50, col="gray", lty=2)

    filename <- paste(
        destdir,
        '/',
        station,
        "-short.png",
        sep = ""
    )

    png(file=filename, width=7.5, height = 4, units="in", res=600) 
    par(mar=c(2,4.1,.5,2.1))
    
    plot(dp_asdate,100*(1-corcoef1_estimate_shortterm), 
         type = "l", ylim = c(0,100),
         xlim=c(dp_asdate[10], dp_asdate[length(date_plot)]), 
         ylab = "Short-term Anomaly (%)", xlab="")

    polygon(c(dp_asdate, rev(dp_asdate)), 
            c(100*(1-corcoef1_low_shortterm), rev(100*(1-corcoef1_up_shortterm))),
            col = "#87CEEB", border='black')
    abline(h=50, col="gray", lty=2)
    dev.off()
}

args <- commandArgs(trailingOnly = TRUE)

calcAnomalies(args[1], args[2], args[3])