library(dplyr)
library(chron)
library(e1071)
library(tseries)
library(fpp2)
library(zoo)
library(caret)
library(randomForest)
library(eseis)
library(gatepoints)
library(RPostgres)
library(DBI)

# The directory to save the generated graphs in
DESTDIR <- 'static/img/events'

db_config <- config::get("DATABASE")
con <- dbConnect(RPostgres::Postgres(),
                user = db_config$user,
                password = db_config$password,
                dbname = db_config$database,
                host = db_config$server)


STATION_SELECT <- "SELECT * FROM
(SELECT station, name,  unnest(channels) as chan
FROM station_channels
INNER JOIN stations ON stations.id=station_channels.station) s1
WHERE chan like '%HZ'"

SQL <- "SELECT
    datetime as date,freq_max1,freq_max10,freq_max20,freq_max30,freq_max40,freq_max50,sd_freq_max10,
    sd_freq_max20,sd_freq_max30,sd_freq_max40,sd_freq_max50,ssa_max1,ssa_max10,ssa_max20,
    ssa_max30,ssa_max40,ssa_max50,sd_ssa_max10,sd_ssa_max20,sd_ssa_max30,sd_ssa_max40,
    sd_ssa_max50,rsam,sd_rsam,s2n
FROM"

# Get a list of station ID's and channels
station_query <- dbSendQuery(con,STATION_SELECT)
station_info <- dbFetch(station_query)
dbClearResult(station_query)


for(idx in 1:length(station_info)){
  station <- station_info[idx,]
  station_id <- as.integer(station['station'])
  station_name <- as.character(station['name'])
  station_channel <- as.character(station['chan'])
  
  table_name <- paste('data',tolower(station_name), tolower(station_channel),sep = '_' )

  paste("Fetching data for",table_name)
  query_string <- paste(SQL,table_name,"ORDER BY datetime")

  query <- dbSendQuery(con, query_string)
  matrix_of_features <- dbFetch(query)
  dbClearResult(query)


  matrix_of_features=na.omit(matrix_of_features)  #remove rows with NA

  # NORMALIZATION OF THE S2N RATIO
  s2n_ratio=as.numeric(matrix_of_features[,'s2n'])/mean(as.numeric(matrix_of_features[,'s2n']),na.rm = FALSE)
  matrix_of_features=cbind(matrix_of_features,s2n_ratio)

  # CALCULATES PREDICTIONS FROM EACH MACHINE LEARNING MODEL PREVIOUSLY TRAINED. THE MODELS NEED TO BE SAVED IN THE SAME FOLDER AS THIS SCRIPT
  temp = list.files(pattern="*.rda")
  for(ensemble in 1:length(temp)){
    load(temp[ensemble])
    aux=(predict(rf,matrix_of_features,data=admitg,type="raw"))
    aux[1]=0; aux[length(aux)]=0
    #this is introduced to impose that the first element is always 0; otherwise the duration of the events detected is negative
    matrix_of_features=cbind(matrix_of_features,aux)
  }

  # EXTRACT INITIATION, ENDING, AND DURATION OF THE EVENTS FROM THE PREDICTIONS
  for(j in 32:length(matrix_of_features)){
    ensemble=j-31
    fre_event=c();ampl_event=c()
    aux=matrix_of_features[,j]
    if(length(which(diff(as.numeric(aux))==1))>0){
      begin_event=matrix_of_features[which(diff(as.numeric(aux))==1),1]
      end_event=matrix_of_features[which(diff(as.numeric(aux))==-1),1]
      duration_event=as.numeric(difftime(end_event, begin_event, units = "secs"))
    }

    for(i in 1:length(begin_event)){
      print(i*100/length(begin_event))
      a=which(matrix_of_features[,1]==begin_event[i])
      b=which(matrix_of_features[,1]==end_event[i])
      fre_event=append(fre_event,mean(matrix_of_features[a:b,2]))
      ampl_event=append(ampl_event,mean(matrix_of_features[a:b,28]))
    }
    if(ensemble==1){date_events_1=cbind(begin_event,end_event,duration_event,ampl_event,fre_event)}
    if(ensemble==2){date_events_2=cbind(begin_event,end_event,duration_event,ampl_event,fre_event)}
    if(ensemble==3){date_events_3=cbind(begin_event,end_event,duration_event,ampl_event,fre_event)}

    # SAVES THE RESULTS FROM EACH MODEL
    # write.csv(date_events, file = paste(filename_output,"_model_",ensemble,".csv",sep=""))
  }

  date_events_1=date_events_1[which(as.numeric(date_events_1[,3])>=6 & as.numeric(date_events_1[,3])<=20),]
  date_events_2=date_events_2[which(as.numeric(date_events_2[,3])>=6 & as.numeric(date_events_2[,3])<=20),]
  date_events_3=date_events_3[which(as.numeric(date_events_3[,3])>=6 & as.numeric(date_events_3[,3])<=20),]

  days_1=unique(substr(date_events_1[,2],1,10))
  number_of_events_per_day_1=c()
  for(i in 1:length(days_1)){number_of_events_per_day_1[i]=length(which(substr(date_events_1[,2],1,10)==days_1[i]))}

  days_2=unique(substr(date_events_2[,2],1,10))
  number_of_events_per_day_2=c()
  for(i in 1:length(days_2)){number_of_events_per_day_2[i]=length(which(substr(date_events_2[,2],1,10)==days_2[i]))}

  days_3=unique(substr(date_events_3[,2],1,10))
  number_of_events_per_day_3=c()
  for(i in 1:length(days_3)){number_of_events_per_day_3[i]=length(which(substr(date_events_3[,2],1,10)==days_3[i]))}

  ylimmax=max(max(number_of_events_per_day_1),max(number_of_events_per_day_2),max(number_of_events_per_day_3))

  filename <- paste(
      DESTDIR,
      '/',
      station_name,
      sep = ""
  )

  png(file=filename, width=7.5, height = 4, units="in", res=600)

  plot(as.POSIXct(days_1),number_of_events_per_day_1,cex.lab=1.1,cex.axis=1.1,col="black",las=1,ylim=c(0,ylimmax),type='l',xlab="Date",ylab="Number of events per day")
  lines(as.POSIXct(days_2),number_of_events_per_day_2,cex.lab=1.1,cex.axis=1.1,col="red",las=1,ylim=c(0,ylimmax),type='l',xlab="Date",ylab="Number of events per day")
  lines(as.POSIXct(days_3),number_of_events_per_day_3,cex.lab=1.1,cex.axis=1.1,col="blue",las=1,ylim=c(0,ylimmax),type='l',xlab="Date",ylab="Number of events per day")
}
dbDisconnect(con)




# THIS IS THE NAME OF THE OUTPUT FILE FROM VolcSeismo
#filename_input='KOKV_data1_complete_matrix_of_features.csv'

# READS THE OUTPUT FILE FROM VolcSeismo AND REMOVE ROWS WITH NA, IF ANY
#matrix_of_features=as.data.frame(read.csv(paste("/Users/tarsilo_girona/Dropbox/PROJECTS_FAIRBANKS/projects/PROJECT_AVO/ATKA_RESPONSE/",filename_input,sep="")))    #load data
