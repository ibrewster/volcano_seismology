suppressMessages(library(zoo))
suppressMessages(library(chron))
suppressMessages(library(fpp2))
suppressMessages(library(config))
library(RPostgres)
library(DBI)


calcAnomalies <- function(station, channel, destdir){
    db_config <- config::get("DATABASE")
    con <- dbConnect(RPostgres::Postgres(),
                    user = db_config$user,
                    password = db_config$password,
                    dbname = db_config$database,
                    host = db_config$server)
                    
    
    STATION_SELECT <- "SELECT id FROM stations WHERE name=$1"
    SQL <- "SELECT freq_max10, sd_freq_max10, rsam, datetime as date
FROM data
WHERE station=$1
AND channel=$2
ORDER BY datetime"

    station_query <- dbSendQuery(con,STATION_SELECT)
    dbBind(station_query,list(station))
    station_id <- dbFetch(station_query)
    station_id <- as.integer(station_id['id'])
    dbClearResult(station_query)
    
    query <- dbSendQuery(con, SQL)
    dbBind(query, list(station_id, channel))
    data = dbFetch(query)
    dbClearResult(query)
    dbDisconnect(con)

    #data=data.frame(read.csv(file = "/Users/israel/Downloads/AUCH-BHZ-2021-06-29T00_00_00Z-2021-09-06T23_59_59Z.csv"))
    print("Beginning calc anomalies function")
    date_plot=c();
    corcoef1_low_longterm_A=c();corcoef1_up_longterm_A=c();corcoef1_estimate_longterm_A=c()
    corcoef1_low_shortterm_A=c();corcoef1_up_shortterm_A=c();corcoef1_estimate_shortterm_A=c()
    date=chron(substr(as.character(data$date),1,10),format=c('y-m-d'))
    days=unique(date)
    if(length(days)<10) {
        stop(paste("Not enough data to generate a graph for",station,channel))
    }

    for(i in seq(10,length(days), by=1)){
        if(length(which(date==days[i]))>3600){
            
            training_longterm=data[which(date<days[i]),]
            training_shortterm=data[which(date>=(days[i]-3) & date<days[i]),]
            test=data[which(date==days[i]),]
            
            ## A: FREQ_MAX
            vect_training_longterm=training_longterm$freq_max[which(training_longterm$freq_max10>=0.5)]
            vect_training_shortterm=training_shortterm$freq_max[which(training_shortterm$freq_max10>=0.5)]
            vect_test=test$freq_max[which(test$freq_max10>=0.5)]

            a1_longterm=hist(vect_training_longterm, breaks = seq(0,20, by=0.1), plot='FALSE')
            a1_shortterm=hist(vect_training_shortterm, breaks = seq(0,20, by=0.1), plot='FALSE')     
            a2=hist(vect_test, breaks = seq(0,20, by=0.1), plot='FALSE')    

            test_longterm=cor.test(a1_longterm$density,a2$density,method = c("pearson"))
            corcoef1_low_longterm_A[i]=abs(test_longterm$conf.int[1])
            corcoef1_up_longterm_A[i]=abs(test_longterm$conf.int[2])
            corcoef1_estimate_longterm_A[i]=abs(test_longterm$estimate)
         
            test_shortterm=cor.test(a1_shortterm$density,a2$density,method = c("pearson"))
            corcoef1_low_shortterm_A[i]=abs(test_shortterm$conf.int[1])
            corcoef1_up_shortterm_A[i]=abs(test_shortterm$conf.int[2])
            corcoef1_estimate_shortterm_A[i]=abs(test_shortterm$estimate)         
            
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

    plot(dp_asdate,100*(1-corcoef1_estimate_longterm_A), 
         type = "l",ylim = c(0,100),
         xlim=c(dp_asdate[10], dp_asdate[length(date_plot)]), 
         ylab = "Long-term Anomaly (%)",xlab="")

    polygon(c(dp_asdate, rev(dp_asdate)), 
            c(100*(1-corcoef1_low_longterm_A), rev(100*(1-corcoef1_up_longterm_A))),
            col = "#87CEEB", border='black')
    abline(h=50, col="gray", lty=2)
    current_long_term_anomaly=100*(1-corcoef1_estimate_longterm_A)[end(100*(1-corcoef1_estimate_longterm_A))][1]
    if(current_long_term_anomaly>50){
        rect(par("usr")[1],par("usr")[3],par("usr")[2],par("usr")[4],col= rgb(1,0,0,alpha=0.3))
    }

    filename <- paste(
        destdir,
        '/',
        station,
        "-short.png",
        sep = ""
    )

    png(file=filename, width=7.5, height = 4, units="in", res=600) 
    par(mar=c(2,4.1,.5,2.1))
    
    plot(dp_asdate,100*(1-corcoef1_estimate_shortterm_A), 
         type = "l", ylim = c(0,100),
         xlim=c(dp_asdate[10], dp_asdate[length(date_plot)]), 
         ylab = "Short-term Anomaly (%)", xlab="")

    polygon(c(dp_asdate, rev(dp_asdate)), 
            c(100*(1-corcoef1_low_shortterm_A), rev(100*(1-corcoef1_up_shortterm_A))),
            col = "#87CEEB", border='black')
    abline(h=50, col="gray", lty=2)
    current_short_term_anomaly=100*(1-corcoef1_estimate_shortterm_A)[end(100*(1-corcoef1_estimate_shortterm_A))][1]
    if(current_short_term_anomaly>50){
        rect(par("usr")[1],par("usr")[3],par("usr")[2],par("usr")[4],col= rgb(1,0,0,alpha=0.3))
    }
    dev.off()
    
    return(list(current_long_term_anomaly,current_short_term_anomaly,station_id))
}

args <- commandArgs(trailingOnly = TRUE)

current_anomalies <- calcAnomalies(args[1], args[2], args[3])

db_config <- config::get("DATABASE")
con <- dbConnect(RPostgres::Postgres(),
                user = db_config$user,
                password = db_config$password,
                dbname = db_config$database,
                host = db_config$server)
                
SQL="UPDATE STATIONS SET long_anomaly=$1, short_anomaly=$2 WHERE id=$3"
update_query <- dbSendQuery(con,SQL)
dbBind(update_query,current_anomalies)
rows <- dbGetRowsAffected(update_query)
dbClearResult(update_query)
dbDisconnect(con)