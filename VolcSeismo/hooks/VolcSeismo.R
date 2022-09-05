#Load libraries and data
suppressMessages(library(dplyr))
suppressMessages(library(chron))
suppressMessages(library(e1071))
suppressMessages(library(tseries))
suppressMessages(library(fpp2))
suppressMessages(library(zoo))
suppressMessages(library(caret))
suppressMessages(library(randomForest))
suppressMessages(library(RSEIS))
suppressMessages(library(gatepoints))

runAnalysis <- function(time, data, station, chan, script_dir){
    time=as.POSIXct(time,format = "%Y-%m-%dT%H:%M:%S")
    #plot(time,data,las=1,type='l')

    window=500   #the features are calculated in time windows of 'window'/50 seconds
    LP=c();freq_max1=c();freq_max10=c();freq_max20=c();freq_max30=c();freq_max40=c();freq_max50=c();
    freq_max60=c();freq_max70=c();freq_max80=c();freq_max90=c();freq_max100=c();time_parameters=c();
    ssa_max1=c();ssa_max10=c();ssa_max20=c();ssa_max30=c();ssa_max40=c();ssa_max50=c();ssa_max60=c();
    ssa_max70=c();ssa_max80=c();ssa_max90=c();ssa_max100=c();rsam=c();sd_freq_max10=c();sd_freq_max20=c();
    sd_freq_max30=c();sd_freq_max40=c();sd_freq_max50=c();sd_freq_max60=c();sd_freq_max70=c();
    sd_freq_max80=c();sd_freq_max90=c();sd_freq_max100=c();
    sd_ssa_max10=c();sd_ssa_max20=c();sd_ssa_max30=c();sd_ssa_max40=c();sd_ssa_max50=c();sd_ssa_max60=c();
    sd_ssa_max70=c();sd_ssa_max80=c();sd_ssa_max90=c();sd_ssa_max100=c();sd_rsam=c();s2n=c();
    time_parameters=c()


    steps=50  #50 to have 1s sliding window
    for (j in seq(window,length(data),steps)){
    #for (j in window){
    aux_signal=data[(j-window+1):j]    #backward 1s-sliding windows of 10 s
    #aux_time=time[(j-window+1):j]
    #plot(aux_time,aux_signal,type='l')
    spectrum=Spectrum(aux_signal, 1/50, one_sided = TRUE, type = 3, method = 1)
    spectrum=cbind(spectrum$f,spectrum$spectrum)
    #plot_spectrum(data=spectrum, unit="linear")
    spectrum[,1]=spectrum[,1][order(spectrum[,2],decreasing = TRUE)]    #order the amplitudes in decreasing order
    spectrum[,2]=spectrum[,2][order(spectrum[,2],decreasing = TRUE)]    #order the amplitudes in decreasing order

    freq_max1=append(freq_max1,spectrum[1,1])
    freq_max10=append(freq_max10,median(spectrum[1:10,1]))
    freq_max20=append(freq_max20,median(spectrum[1:20,1]))
    freq_max30=append(freq_max30,median(spectrum[1:30,1]))
    freq_max40=append(freq_max40,median(spectrum[1:40,1]))
    freq_max50=append(freq_max50,median(spectrum[1:50,1]))

    sd_freq_max10=append(sd_freq_max10,sd(spectrum[1:10,1]))
    sd_freq_max20=append(sd_freq_max20,sd(spectrum[1:20,1]))
    sd_freq_max30=append(sd_freq_max30,sd(spectrum[1:30,1]))
    sd_freq_max40=append(sd_freq_max40,sd(spectrum[1:40,1]))
    sd_freq_max50=append(sd_freq_max50,sd(spectrum[1:50,1]))

    ssa_max1=append(ssa_max1,spectrum[1,2])
    ssa_max10=append(ssa_max10,median(spectrum[1:10,2]))
    ssa_max20=append(ssa_max20,median(spectrum[1:20,2]))
    ssa_max30=append(ssa_max30,median(spectrum[1:30,2]))
    ssa_max40=append(ssa_max40,median(spectrum[1:40,2]))
    ssa_max50=append(ssa_max50,median(spectrum[1:50,2]))

    sd_ssa_max10=append(sd_ssa_max10,sd(spectrum[1:10,2]))
    sd_ssa_max20=append(sd_ssa_max20,sd(spectrum[1:20,2]))
    sd_ssa_max30=append(sd_ssa_max30,sd(spectrum[1:30,2]))
    sd_ssa_max40=append(sd_ssa_max40,sd(spectrum[1:40,2]))
    sd_ssa_max50=append(sd_ssa_max50,sd(spectrum[1:50,2]))

    rsam=append(rsam,median(abs(aux_signal)))
    sd_rsam=append(sd_rsam,sd(abs(aux_signal)))
    s2n=append(s2n,mean(abs(aux_signal))/mean(abs(data)))

    time_parameters=append(time_parameters,substr(time[j],1,23))
    }

    V1 <- as.character(time_parameters)
    matrix_of_features=as.data.frame(cbind.data.frame(V1,freq_max1,freq_max10,freq_max20,freq_max30,freq_max40,freq_max50,sd_freq_max10,sd_freq_max20,sd_freq_max30,sd_freq_max40,sd_freq_max50,ssa_max1,ssa_max10,ssa_max20,ssa_max30,ssa_max40,ssa_max50,sd_ssa_max10,sd_ssa_max20,sd_ssa_max30,sd_ssa_max40,sd_ssa_max50,rsam,sd_rsam,s2n))
    
    return(matrix_of_features)
}

genEventGraphs1 <- function(matrix_of_features, station, script_file){
    SCRIPT_DIRECTORY <- dirname(script_file)
    ## SCRIPT TO SEND TO ISRAEL (26 AUGUST 2022) - EXTRACTION OF EVENTS
    matrix_of_features=na.omit(as.data.frame((matrix_of_features)))  #remove rows with NA

    # NORMALIZATION OF THE S2N RATIO
    s2n_ratio=as.numeric(matrix_of_features[,26])/mean(as.numeric(matrix_of_features[,26]),na.rm = FALSE)
    matrix_of_features=cbind(matrix_of_features,s2n_ratio)
    
    # CALCULATES PREDICTIONS FROM EACH MACHINE LEARNING MODEL PREVIOUSLY TRAINED. THE MODELS NEED TO BE SAVED IN THE SAME FOLDER AS THIS SCRIPT  
    temp = list.files(path=SCRIPT_DIRECTORY, pattern="*.rda")
    for(ensemble in 1:length(temp)){
        model_file <- paste(SCRIPT_DIRECTORY, temp[ensemble], sep='/')
        load(model_file)
        aux=(predict(rf,matrix_of_features,data=admitg,type="raw"))
        aux[1]=0; aux[length(aux)]=0
        #this is introduced to impose that the first element is always 0; otherwise the duration of the events detected is negative 
        matrix_of_features=cbind(matrix_of_features,aux)
    }
    matrix_of_features_with_predictions=matrix_of_features
    events = genEventGraphs2(matrix_of_features_with_predictions, station, script_file)
    return(as.data.frame(events))
}

genEventGraphs2 <- function(matrix_of_features, station, script_file){
    # EXTRACT INITIATION, ENDING, AND DURATION OF THE EVENTS FROM THE PREDICTIONS
    events=c()
    for(j in 28:length(matrix_of_features)){
      ensemble=j-27
      fre_event=c()
      ampl_event=c()
      aux=matrix_of_features[,j]
      if(length(which(diff(as.numeric(aux))==1))>0){
        begin_event=matrix_of_features[which(diff(as.numeric(aux))==1),1]
        end_event=matrix_of_features[which(diff(as.numeric(aux))==-1),1]
        duration_event=as.numeric(difftime(end_event, begin_event, units = "secs"))

        for(i in 1:length(begin_event)){
          a=which(matrix_of_features[,1]==begin_event[i])
          b=which(matrix_of_features[,1]==end_event[i])
          fre_event=append(fre_event,mean(matrix_of_features[a:b,2]))
          ampl_event=append(ampl_event,mean(matrix_of_features[a:b,24]))
        }
      }
      else{begin_event=NA;end_event=NA;duration_event=NA;ampl_event=NA;fre_event=NA}
      events_ensemble=cbind(ensemble,begin_event,end_event,duration_event,ampl_event,fre_event)
      events=rbind(events,events_ensemble)
    }
    return(events)
}


genEventGraphs <- function(matrix_of_features, station, script_file){
    print("Making Predictions")
    SCRIPT_DIRECTORY <- dirname(script_file)
    matrix_of_features=na.omit(matrix_of_features)  #remove rows with NA
    
    # NORMALIZATION OF THE S2N RATIO
    s2n_ratio=as.numeric(matrix_of_features[,26])/mean(as.numeric(matrix_of_features[,26]), na.rm = FALSE)
    matrix_of_features=cbind(matrix_of_features,s2n_ratio)
    

    # CALCULATES PREDICTIONS FROM EACH MACHINE LEARNING MODEL PREVIOUSLY TRAINED. THE MODELS NEED TO BE SAVED IN THE SAME FOLDER AS THIS SCRIPT  
    temp = list.files(path=SCRIPT_DIRECTORY, pattern="*.rda")
    for(ensemble in 1:length(temp)){
        model_file <- paste(SCRIPT_DIRECTORY, temp[ensemble], sep='/')
        load(model_file)
        aux=(predict(rf,matrix_of_features,data=admitg,type="raw"))
        aux[1]=0; aux[length(aux)]=0
        #this is introduced to impose that the first element is always 0; otherwise the duration of the events detected is negative 
        matrix_of_features=cbind(matrix_of_features,aux)
    }
    
    # Setup for plot creation
    filename <- paste(
        SCRIPT_DIRECTORY,
        '../web/static/img/events',
        station,
        sep = "/"
    )
    
    filename <- paste(filename,'.png',sep="")
    filename <- normalizePath(filename)

    png(file=filename, width=7.5, height = 4, units="in", res=600)
    
    plot_created <- FALSE
    plot_colors <- c("black","red","blue","green")
    # EXTRACT INITIATION, ENDING, AND DURATION OF THE EVENTS FROM THE PREDICTIONS
    for(j in 27:length(matrix_of_features)){
        ensemble=j-26
        fre_event=c();ampl_event=c()
        aux=matrix_of_features[,j]
                
        if(length(which(diff(as.numeric(aux))==1))>0){
            begin_event=matrix_of_features[which(diff(as.numeric(aux))==1),1]
            end_event=matrix_of_features[which(diff(as.numeric(aux))==-1),1]
            duration_event=as.numeric(difftime(end_event, begin_event, units = "secs"))
        }
        else{
            next
        }
        
        for(i in 1:length(begin_event)){
            print(i*100/length(begin_event))
            a=which(matrix_of_features[,1]==begin_event[i])
            b=which(matrix_of_features[,1]==end_event[i])
            fre_event=append(fre_event,mean(matrix_of_features[a:b,2]))
            ampl_event=append(ampl_event,mean(matrix_of_features[a:b,28]))
        }
        
        date_events=cbind(begin_event,end_event,duration_event,ampl_event,fre_event)
        
        print("***BEGINING GRAPH GENERATION***")
        print("FILENAME:")
        print(filename)

        # SAVES THE RESULTS FROM EACH MODEL
        # write.csv(date_events, file = paste(filename_output,"_model_",ensemble,".csv",sep=""))
        days=unique(substr(date_events[,2],1,10))
        print("Days:")
        print(days)
        print("____")
        number_of_events_per_day=c()
        for(i in 1:length(days)){ 
            number_of_events_per_day[i]=length(which(substr(date_events[,2],1,10)==days[i]))
        }
        
        line_color <- plot_colors[ensemble]
        if(!plot_created){
            print("Creating plot from")
            print(number_of_events_per_day)
            plot_created <- TRUE
            plot(as.POSIXct(days),number_of_events_per_day,cex.lab=1.1,cex.axis=1.1,col="black",las=1,,type='l',xlab="Date",ylab="Number of events per day")
        }else{
            print("Adding line")
            print(number_of_events_per_day)
            lines(as.POSIXct(days),number_of_events_per_day,cex.lab=1.1,cex.axis=1.1,col="blue",las=1,,type='l',xlab="Date",ylab="Number of events per day")
        }
    }
    dev.off()
    #if(found_events){
        #print("^^^^^^^^^EVENTS FOUND ON FINAL LOOP. GENERATING GRAPHS^^^^^^^^^")

        #days=unique(substr(date_events[,2],1,10))
        #number_of_events_per_day=c()
        #for(i in 1:length(days)){ 
            #number_of_events_per_day[i]=length(which(substr(date_events[,2],1,10)==days[i]))
        #}
        

        
        #plot(as.POSIXct(days2),number_of_events_per_day_model2,cex.lab=1.1,cex.axis=1.1,col="black",las=1,,type='l',xlab="Date",ylab="Number of events per day")
        #lines(as.POSIXct(days3),number_of_events_per_day_model3,cex.lab=1.1,cex.axis=1.1,col="red",las=1,,type='l',xlab="Date",ylab="Number of events per day")
        #lines(as.POSIXct(days4),number_of_events_per_day_model4,cex.lab=1.1,cex.axis=1.1,col="blue",las=1,,type='l',xlab="Date",ylab="Number of events per day")
    #}
}