var map;
var global_graph_div = null;
var divid = 0;
var volcano;
var vectorGuard = false;
let dVvStation=null;

var iconColors = {
    'campaign': '#0CDA3B',
    'continuous': '#4900F4',
    'pbo': '#EAFF00'
}

stationMarkers = []

var iframe_id = 0;
var menuMode = false;

$(document).ready(function() {

    $(document).on('change', '#volcSelect', setMapLocation);
    $(document).on('change','.dvvSelect',selectdVvPair);
    $(document).on('click', 'img.closeBtn', closeGraph);
    $(document).on('click', 'span.dateBtns button', dateRangeClicked);
    $(document).on('click', 'input.channelOption', generateGraphs);
    $(document).on('click', 'button.downloadData', downloadData);
    $(document).on('click', '.tabs button', setTab);
    $(document).on('click', 'button.downloadEvents',downloadEvents);
    $('#eventsTab').click(getEvents);
    $('img.menu').click(showMenu);

    $(document).on('change', 'div.chartHeader input.date', changeDateRangeInput);
    $(document).on('plotly_relayout', 'div.graphArea', setZoomRange);
    $(document).on('plotly_doubleclick', 'div.graphArea', registerDoubleClick);

    $('#download').click(download_view);

    volcano = getUrlVars()['volcano']

    initMap();
    initFinal();
    getAnomaliesDebounce();
})

function setTab(){
    $(this).closest('.tabs').find('button').removeClass('current');
    $(this).addClass('current');
    $('.tabView').hide();
    closeAllGraphs();

    const dest=$(this).data('target');
    if(dest!='None'){
        $(`#${dest}`).show();
    }

    if($(this).is('#stationsTab')){
        $('#content').removeClass('anomaliestab');
    }
    else{
        $('#content').addClass('anomaliestab');
    }

}

function downloadData() {
    //context should be a button in a graph div header.
    var chartDiv = $(this).closest('div.chart')
    var graphDiv = chartDiv.find("div.plotlyPlot")[0];
    //get date range displayed
    var x_range = graphDiv.layout.xaxis.range;
    //get station and channel
    var chanInfo = chartDiv.find('.channelOption:checked').data();

    var args = {
        'factor': 100,
        'dateFrom': x_range[0],
        'dateTo': x_range[1],
    }

    Object.assign(args, chanInfo);

    var params = $.param(args);
    var url = 'get_full_data?' + params;
    window.location.href = url;
}

function downloadEvents(){
    const volc=$(this).data('volc');
    args ={'volc':volc}
    //TODO: filter by date
    const params=$.param(args)
    const url='getEventData?'+params;
    window.location.href = url;
}

function initMap() {
    map = new google.maps.Map(document.getElementById('map'), {
        center: { lat: 54.1, lng: -165.9 },
        zoom: 11,
        mapTypeId: google.maps.MapTypeId.TERRAIN,
        scaleControl: true,
        streetViewControl: false,
        scaleControlOptions: {
            position: google.maps.ControlPosition.BOTTOM_LEFT,
        },
        fullscreenControl: false,
        isFractionalZoomEnabled: true
    });

    // Get a list of stations to create markers for
    $.get('list_stations', map_stations);

    //set the inital map view, if requested
    if (typeof(volcano) !== 'undefined') {
        $('#volcSelect').val(volcano);
    } else {
        $('#volcSelect').change();
    }
}

function initFinal() {
    //matomo analytics tracking
    var _paq = window._paq = window._paq || [];
    /* tracker methods like "setCustomDimension" should be called before "trackPageView" */
    _paq.push(['trackPageView']);
    _paq.push(['enableLinkTracking']);
    (function() {
        var u = "https://apps.avo.alaska.edu/analytics/";
        _paq.push(['setTrackerUrl', u + 'matomo.php']);
        _paq.push(['setSiteId', '3']);
        var d = document,
            g = d.createElement('script'),
            s = d.getElementsByTagName('script')[0];
        g.type = 'text/javascript';
        g.async = true;
        g.src = u + 'matomo.js';
        s.parentNode.insertBefore(g, s);
    })();
}

function registerDoubleClick(event) {
    var context = this;
    setTimeout(function() {
        setZoomRange.call(context, event);
    }, 100);
}

function generateGraphs() {
    //"this" is the button that was clicked
    const dest = $(this).closest('div.chart');
    const optionsDiv = dest.find('div.channelSel');
    //Not neccesarily redundant, as this function could
    //be called() with an arbitrary button as "this"
    const selChannelOpt = optionsDiv.find('.channelOption:checked')
    const station = selChannelOpt.data('station');
    const channel = selChannelOpt.data('channel');

    const entropiesImgs = dest.find('div.anomalies');
    if (channel[channel.length - 1] === 'Z')
        entropiesImgs.removeClass('hidden');
    else
        entropiesImgs.addClass('hidden');

    dest.show();

    let dateFrom,dateTo;
    [dateFrom,dateTo]=getDates(dest);

    var reqParams = {
        'station': station,
        'channel': channel,
        'dateFrom': dateFrom,
        'dateTo': dateTo
    }

    var graphDiv = dest.find('.graphArea')
    Plotly.purge(graphDiv[0]);
    graphDiv.find('.noDataWarning').remove();
    graphDiv.append('<div class="loadingMsg">Loading...<div class="loading"></div></div>');

    $.get('get_graph_data', reqParams)
        .done(function(data) {
            if(dVvStation!==null){
                getdVvData(dest,data,station,dVvStation);
            }
            else{
                graphResults(data, dest);
            }
        })
        .fail(function(a,b,c){
            alert("Unable to load plot data!");
        })
        .always(function() {
            graphDiv.find('.loadingMsg').remove();
        });

}

function setZoomRange(event, params) {
    if (layoutCount > 0) {
        layoutCount -= 1;
        return;
    }

    //see if we have any data about the x axis range being changed
    if (typeof(params) != 'undefined' &&
        !params.hasOwnProperty('xaxis.range') &&
        (!params.hasOwnProperty('xaxis.range[0]') || !params.hasOwnProperty('xaxis.range[1]')))
        return;

    var xaxis_range = this.layout.xaxis.range;

    //Set the input fields to match the xaxis range
    var parentChart = $(this).closest('div.chart');
    var dateFromInput = parentChart.find('input.dateFrom');
    var dateToInput = parentChart.find('input.dateTo');

    parentChart.find('span.dateBtns button').removeClass('active');

    var dates = parseRangeDates(xaxis_range);
    var dateFrom = new Date(dates[0]);
    var dateTo = new Date(dates[1]);
    var dateSpan = dateTo - dateFrom;
    var days = (dateSpan / 1000) / 60 / 60 / 24;
    console.log("New date range:");
    console.log(days);

    //Range and limits here are off by one. Adjust
    dateTo = new Date(dateTo.setDate(dateTo.getDate() - 1));

    var formattedFrom = formatDateString(dateFrom);
    var formattedTo = formatDateString(dateTo);
    dateFromInput.val(formattedFrom);
    dateToInput.val(formattedTo);

    setTitle(parentChart);
    let exactDates = parseRangeDates(xaxis_range, true);
    //rescaleY(parentChart, exactDates[0], exactDates[1]);
}

function setGraphRange() {
    //"this" is the date input that changed. Could be either dateFrom or dateTo.
    var graph = $(this).closest('div.chart');
    var graphDiv = graph.find('div.plotlyPlot:visible')[0]

    if (typeof(graphDiv.data) == 'undefined') {
        return; //no graph (yet, at least);
    }

    var dateFromVal = graph.find('input.dateFrom').val();
    var dateToVal = graph.find('input.dateTo').val();

    var dateFrom = new Date(dateFromVal);
    dateFrom = new Date(dateFrom.setUTCHours(0, 0, 0, 0));

    //dateTo to end of day
    var dateTo = new Date(dateToVal);
    dateTo = new Date(dateTo.setUTCHours(23, 59, 59, 999));

    //var layout = graphDiv.layout;
    var layout_updates = {
        'xaxis2.range': [ //xaxis2 is the "master" that all xaxis match, so we only need to change the one
            formatISODateString(dateFrom),
            formatISODateString(dateTo)
        ]
    }

    //var y_layouts = rescaleY(graph, dateFrom, dateTo, false);

    //layout_updates = Object.assign(layout_updates, y_layouts);
    Plotly.relayout(graphDiv, layout_updates);
}

function setTitle(parentChart) {
    var graphDiv = parentChart.find('div.plotlyPlot:visible')[0]
    if (typeof(graphDiv.data) == 'undefined') {
        return; //no graph (yet, at least);
    }
    var range = parseRangeDates(graphDiv.layout.xaxis.range)

    var dateFrom = formatDateString(range[0]);
    var dateTo = formatDateString(range[1]);
    var plot_title = parentChart.find('.stationName').text();

    plot_title += " - " + dateFrom + " to " + dateTo;

    var filename = "Volcano Seismology ";
    filename += dateFrom.replace(/\//g, '-') + " to ";
    filename += dateTo.replace(/\//g, '-');

    const height=1024;

    var newConfig = {
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': filename,
            'height': height,
            'width': 600,
            'scale': 1.5,
        }
    }

    const label_y=55.296;
    const label_percent=(height-label_y)/height;

    var title_dict = {
        'text': plot_title,
        'x': .06,
        'y': label_percent,
        'xanchor': 'left',
        'yanchor': 'bottom',
        'font': {
            'size': 16,
        },
    }

    layoutCount += 1;
    Plotly.relayout(graphDiv, 'title', title_dict);
    Plotly.react(graphDiv, graphDiv.data, graphDiv.layout, newConfig);
}

function findRangeIndexes(array,min,max) {
    let start=0,stop=array.length-1;
    let foundStart=false;
    for (let i=0;i<array.length;i++) {
        let val=array[i];
        if(!foundStart && min<=val) {
            foundStart=true;
            start=i;
        } else if(val>max) {
            stop=i;
            break;
        }
    }
    return [start,stop];
}

function rescaleY(parentChart, dateFrom, dateTo, run) {
    if (typeof(run) == 'undefined')
        run = true
        //try to figure out the Y axis for each graph
        //decompose the date from and date to into the format of the data
    const dateFromString = formatFakeISODateString(dateFrom);
    const dateToString = formatFakeISODateString(dateTo);

    const graphs = parentChart.find('div.plotlyPlot:visible')[0]
    const graphData=graphs.data;
    var all_yaxis = filtered_keys(graphs.layout, /yaxis?/).sort();

    const axis_lookup={}
    //create a y-axis lookup dict
    for (let axis of all_yaxis){
        axis_lookup[axis]=[Number.MAX_VALUE,-1 * Number.MAX_VALUE];
    }

    let lastDataDate=graphData[0]['x'][graphData[0]['x'].length-1];
    for (let thisData of graphData) {
        if (!thisData.hasOwnProperty('y')) {continue};

        let dates=thisData.x;

        if(dates[dates.length-1]>lastDataDate){lastDataDate=dates[dates.length-1];}

        let start,stop;
        [start,stop]=findRangeIndexes(dates,dateFromString,dateToString);

        let yaxis = typeof(thisData.yaxis)=='undefined'?'':thisData.yaxis.replace('y','');
        yaxis = 'yaxis' + yaxis;

        let yData=thisData.y.slice(start, stop + 1);

        // the following lines *look* cleaner, but I'm thinking the
        // single loop is going to be faster...
        // const maxY = arr.reduce((a, b) => Math.max(a, b), -Infinity);
        // const minY = arr.reduce((a, b) => Math.min(a, b), Infinity);

        // if(maxY>axis_lookup[yaxis][1]) {axis_lookup[yaxis][1]=maxY}
        // if(minY<axis_lookup[yaxis][0]) {axis_lookup[yaxis][0]=yval}

        for(let yval of yData){
            if(yval>axis_lookup[yaxis][1]) {axis_lookup[yaxis][1]=yval}
            if(yval<axis_lookup[yaxis][0]) {axis_lookup[yaxis][0]=yval}
        }
    }

    var layouts = {};
    for (let axis in axis_lookup){
        let max,min;
        [min,max]=axis_lookup[axis];

        const spread=max-min;
        const padding=.05*spread;
        max += padding;

        if (min !=0 ){
            min -= padding;
        }

        if(Math.abs(min)!=Infinity && Math.abs(max)!=Infinity){
            layouts[axis + '.range'] = [min, max];
        }
    }


    // var dateData = graphs.data[0]['x']; //same for all plots, so just use the first

    // var startIdx = 0; //technically should be end of list, since if nothing is greater than the start value, then the start value is past the end,
    // //however we'll go with zero so we don't wind up with an empty list, even if the zoom range is beyond the end of the list.
    // var startValue = dateData.find(function(element) {
    //     return element >= dateFromString;
    // });
    // if (typeof(startValue !== 'undefined'))
    //     startIdx = dateData.indexOf(startValue);

    // var stopIdx = dateData.length - 1; //minus 1 because index is 0 based, length is 1 based
    // //this will return the value or undefined, so I need to find the index of the value and handle undefined.
    // var stopValue = dateData.find(function(element) {
    //     return element >= dateToString
    // });

    // if (typeof(stopValue) !== 'undefined') //no number greater than this in list, this is the greatest!
    //     stopIdx = dateData.indexOf(stopValue); //we know this is in the list, because we found it above.



    // for (var i = 0; i < graphs.data.length; i++) {
    //     //possible that some subgraphs don't have a yAxis (polar, for example)


    //     var yData = graphs.data[i]['y'].slice(startIdx, stopIdx + 1);
    //     var yAxis = all_yaxis.shift();

    //     var max = -1 * Number.MAX_VALUE;
    //     var min = Number.MAX_VALUE;
    //     for (var j = 0; j < yData.length; j++) {
    //         if (yData[j] > max) { max = yData[j]; }
    //         if (yData[j] < min) { min = yData[j]; }
    //     }

    //     var spread = max - min;
    //     var padding = .05 * spread; //5% total value buffer
    //     //leave a small buffer on either side when displaying
    //     max += padding;
    //     if (min != 0)
    //         min -= padding;

    //     layouts[yAxis + '.range'] = [min, max];
    // }

    //See if we should display the "no data" label
    if (lastDataDate < dateFrom) {
        parentChart.find('div.plotlyPlot:visible').addClass("noData");
    } else {
        parentChart.find('div.plotlyPlot:visible').removeClass("noData");
    }

    if (run) {
        layoutCount += 1;
        Plotly.relayout(graphs, layouts);
    } else
        return layouts
}

function filtered_keys(obj, filter) {
    var key, keys = [];
    for (key in obj) {
        if (obj.hasOwnProperty(key) && filter.test(key)) {
            keys.push(key);
        }
    }
    return keys;
}

function changeDateRangeInput() {
    //since we manually changed the date range, de-select the preset buttons
    $('button.active').removeClass('active');

    var context = this;
    setTimeout(function() { generateGraphs.call(context); }, 5);
}

function dateRangeClicked() {
    var endDate = new Date();
    var startDate;
    var type = $(this).data('type');
    var step = $(this).data('step'); //may be null

    var chartDiv = $(this).closest('div.chart');
    var graphDiv = chartDiv.find('.plotlyPlot:visible');

    if (type == 'all') {
        startDate = new Date(graphDiv.data('minDate'));
        endDate = new Date(graphDiv.data('maxDate'));
    } else if (type == 'year') {
        startDate = new Date();
        startDate.setYear(endDate.getFullYear() - Number(step));
    } else if (type == 'month') {
        startDate = new Date();
        startDate.setMonth(endDate.getMonth() - Number(step));
    } else if (type == 'day') {
        startDate = new Date();
        startDate.setDate(endDate.getDate() - Number(step));
    }

    chartDiv.find('input.dateFrom').val(formatDateString(startDate));
    var dateToEntry = chartDiv.find('input.dateTo').val(formatDateString(endDate));

    setTitle(chartDiv);
    var context = this;
    setTimeout(function() { generateGraphs.call(context); }, 5);

    chartDiv.find('span.dateBtns button').removeClass('active');
    $(this).addClass('active');
}

function getDates(dest){
    const dateFromFld = dest.find('input.dateFrom');
    const dateToFld = dest.find('input.dateTo');

    let dateFrom = dateFromFld.val();
    let dateTo = dateToFld.val();

    if (dateFrom == '') {
        dateFrom = new Date(new Date() - 604800000); //one-week
        dateTo = new Date();
        dateFrom = formatDateString(dateFrom);
        dateTo = formatDateString(dateTo);
        dateFromFld.val(dateFrom);
        dateToFld.val(dateTo);
    }
    
    return [dateFrom, dateTo];
}

function getdVvData(dest,data,sta1,sta2,dfrom,dto){
    let dateFrom,dateTo;
    [dateFrom,dateTo]=getDates(dest);
    
    const args={
        sta1:sta1,
        sta2:sta2,
        dFrom:dateFrom,
        dTo:dateTo
    }
    
        
    const future=$.getJSON('getdVvData',args)
    .done(function(dvvData){
        dVvStation=sta2;
        for(const [key,value] of Object.entries(dvvData)){
            data['data'][key]=value
        }
        // data['data']['dvvHeatX']=dvvData['heatX'];
        // data['data']['dvvHeatY']=dvvData['heatY'];
        // data['data']['dvvHeatZ']=dvvData['heatZ'];
        // data['data']['cohX']=dvvData['cohX'];
        // data['data']['cohY']=dvvData['cohY'];
        // data['data']['cohZ']=dvvData['cohZ'];
        graphResults(data, dest);
    })
    .fail(function(a,b,c){alert(b); console.log(a)});

    return future;
}

function showStationGraphs(event,volc) {
    //we clicked on a station. Switch to the station graphs tab
    if(!$('#stationsTab').hasClass('current'))
        $('#stationsTab').click();

    //get the REAL event from the fake GoogleMaps event
    if(typeof(event) != 'undefined'){
        event = Object.values(event)
        .filter(function(property) {
            return property instanceof window.MouseEvent;
        })[0];
    }

    var station = $(this).data('name');
    var channels = $(this).data('channels');
    var site = $(this).data('site');

    if(typeof(volc)!=="undefined"){
        $('#volcSelect').val(volc).change()
    }

    var visible_charts = $('div.chart:visible');
    var dvvSelect = typeof(event)=='undefined' ? false : event.altKey;
    if(dvvSelect){
        visible_charts.each(function(){
            let data=$(this).data('rawData');
            let plotStation=$(this).data('station');
            let plotDiv=$(this);
            getdVvData(plotDiv,data,station,plotStation)
        });
        return;
    }
    else{
        dVvStation=null;
    }

    var available_charts = $('div.chart:hidden')

    var add_chart = typeof(event) == 'undefined' ? false : event.shiftKey;
    if (add_chart && visible_charts.length == 3) {
        alert("Sorry, but due to browser limitations, only three sets of graphs can be shown at once");
        return;
    }

    var chartDiv = null;
    if (!add_chart && (visible_charts.length > 0 || available_charts.length > 0)) { //re-use the existing div
        if (visible_charts.length > 0)
            chartDiv = visible_charts.last();
        else //must have no visible, but one or more hidden.
            chartDiv = available_charts.first();

        chartDiv.find('div.chartHeader').remove();

        chartDiv.prepend(createChartHeader(station, site, channels));
    } else if (add_chart && available_charts.length > 0) {
        chartDiv = available_charts.first();
        chartDiv.find('div.chartHeader').remove();
        chartDiv.prepend(createChartHeader(station, site, channels));
    } else { //create a new div. Not trying to add, and nothing available to replace
        chartDiv = createChartDiv(station, site, channels);
        $('#content').append(chartDiv);
    }
    chartDiv.data('station',station);

    //trigger generation of the default set of graphs
    chartDiv.find('input.channelOption').first().click();
}

function adddVvSelect(data,title){
    title.append(' &nbsp;dv/v Pair:');
    const select=$('<select class="dvvSelect">');
    select.append('<option>');
    for(let opt of data){
        let option=$('<option>').text(opt);
        select.append(option);
    }
    title.append(select);
    select.change();
}

function selectdVvPair(){
    const pair=this.value;

    if(pair=='') {return;}

    const dest=$(this).closest('div.chart');
    const data=dest.data('rawData');
    const station=dest.data('station');
    getdVvData(dest,data,station,pair);
}

function createChartHeader(station, site, channels) {
    divid += 1;
    var chartHeader = $('<div class=chartHeader>');
    //title
    var chartTitle = $('<span class="chartTitle">');
    chartHeader.append(chartTitle);
    chartTitle.append('<b><span class="stationName">' + station +
        "</b>"
    );

    $.getJSON('getdVvPairs',{'station':station})
    .done(function(data){
        adddVvSelect(data,chartTitle);
    })

    //close button
    chartHeader.append("<img src='static/img/RedClose.svg' class='closeBtn noPrint'/>")

    //download button
    chartHeader.append("<button class='downloadData'>Download Data</button>");

    //channel selector title
    var channelSelector = $('<div class="channelSel">');
    var radioButtons = $("<div class='inlineblock channelbuttons'>");
    channelSelector.append(radioButtons)
    radioButtons.append("Channel: ");

    //channel selector options
    //sort the "Z" channel first
    const zIndex = channels.findIndex(function(str) {
        if (str[str.length - 1] === 'Z') {
            return true;
        }
        return false;
    });

    if (zIndex >= 0) {
        const zChan = channels[zIndex];
        channels.splice(zIndex, 1);
        channels.unshift(zChan);
    }

    for (var idx in channels) {
        var channel = channels[idx];
        var button = $('<input type="radio" class="channelOption">');
        button.prop('name', divid + '_channel');
        button.data('station', station);
        button.data('channel', channel);
        radioButtons.append(button);
        radioButtons.append(" ");
        radioButtons.append(channel);
        radioButtons.append(" ");
    }

    channelSelector.find('input[type=radio]:first').prop('checked', true);

    chartHeader.append(channelSelector);

    //date selector
    var dateDiv = $('<div class="dateSelect">')
    dateDiv.append("Date Range: ");
    dateDiv.append("<input type=text size=10 class='dateFrom date' maxlength=10/>")
    dateDiv.append("-");
    dateDiv.append("<input type=text size=10 class='dateTo date' maxlength=10/>");

    //date selector buttons
    var dateBtns = $('<span class="dateBtns">');
    dateBtns.append("<button data-type='all'>all</button>");
    dateBtns.append("<button data-type='year' data-step=1>1y</button>");
    dateBtns.append("<button data-type='month' data-step=6>6m</button>");
    dateBtns.append("<button data-type='month' data-step=1>1m</button>");
    dateBtns.append("<button class='default' data-type='day' data-step=7>1w</button>");
    dateDiv.append(dateBtns);
    dateDiv.append('<div class=loading style="display:none">');

    chartHeader.append(dateDiv);

    return chartHeader;
}

function createEntropiesDiv(volc, station,img,stationID){
    let entropiesTopDiv=$('<div class="anomaliesTop">');

    entropiesTopDiv.append(`<div class=title>${station}</div>`);
    entropiesTopDiv.data('stationID',stationID);
    entropiesTopDiv.on('click',function(event){
        let id=$(this).data('stationID');
        showStationGraphs.call(markerLookup[id],event,volc);
    });

    let entropiesDiv = $('<div class="anomalies short">');
    let entropiesImg = $('<img>');
    entropiesImg.on('error', function() {
            $(this).closest('div.anomaliesTop').addClass('error');
        })
        .on('load', function() {
            $(this).closest('div.anomaliesTop').removeClass('error');
        })
        .attr('src', img);

    entropiesDiv.append(entropiesImg);
    entropiesTopDiv.append(entropiesDiv);

    return entropiesTopDiv;
}

function createChartDiv(station, site, channels) {
    var chartDiv = $('<div class="chart">');

    var chartHeader = createChartHeader(station, site, channels);
    chartDiv.append(chartHeader);


    var graph_wrapper = $('<div class=graphWrapper>');

    graph_wrapper.append('<div class="graphArea plotlyPlot">');

    chartDiv.append(graph_wrapper);

    legend_html = '<div id="legend">\
    Legend:\
    <p>Freq Max10: Median frequency of the ten frequency peaks with higher amplitude.</p>\
    <p>SD Freq Max10: Standard deviation of the frequency (for the ten frequency peaks with higher amplitude).</p>\
    </div>';
    chartDiv.append(legend_html);


    return chartDiv;
}

function download_view() {
    showMessage("PDF Generation Requested.<br>Requested file(s) will download once ready.<br>This may take up to five minutes. Please wait...")
    generateMapImage();
    saveCharts();
}

function saveCharts() {
    $('div.chart:visible .plotlyPlot:visible').each(function(idx, element) {
        var data = element.data;
        var layout = element.layout;

        var args = {
            data: data,
            layout: layout
        }

        dom_post('api/gen_graph', args)
    });
}

function generateMapImage() {
    var mapBounds = map.getBounds().toJSON();
    var params = {
        'map_bounds': mapBounds,
    };

    dom_post('map/download', params);
}

function plotGraph(div, data, layout, config) {
    if (typeof(data) == 'undefined') {
        //nothing provided. Load it from the specified div
        data = div[0].data;
        layout = div[0].layout;
        config = div.data('graph_config');
    } else {
        //values provided. Save them to the specified div
        div.data('graph_config', config)
    }

    Plotly.react(div[0], data, layout, config);

    //check for draw errors, and replot if needed
    var canvas = div.find('canvas')[0];
    var gl = canvas.getContext('webgl');
    var error = gl.getError();
    if (error != 0) {
        Plotly.purge(div[0]);
        Plotly.plot(div[0], data, layout, config);
        console.log("*******" + error + "*******")
    }

    var range = parseRangeDates(div[0].layout.xaxis.range);
    var date_from = new Date(range[0])
    var date_to = new Date(range[1])
    rescaleY(div.closest('.chart'), date_from, date_to)
}

var rdYlGnColorscale = [
    [0.0, 'rgb(165,0,38)'],   // red
    [0.1, 'rgb(215,48,39)'],
    [0.2, 'rgb(244,109,67)'],
    [0.3, 'rgb(253,174,97)'],
    [0.4, 'rgb(254,224,139)'],
    [0.5, 'rgb(255,255,191)'], // yellow
    [0.6, 'rgb(217,239,139)'],
    [0.7, 'rgb(166,217,106)'],
    [0.8, 'rgb(102,189,99)'],
    [0.9, 'rgb(26,152,80)'],
    [1.0, 'rgb(0,104,55)']    // green
];

function graphResults(respData, dest) {
    // save the response data for potential future use
    dest.data('rawData',respData);

    var data = respData.data;
    var factor = respData.factor;
    var graphDiv = dest.find('div.graphArea');

    var channelOpt = dest.find('.channelOption:checked')
    var station = channelOpt.data('station');
    var channel = channelOpt.data('channel');

    graphDiv.data('minDate', data['info']['min_date']);
    graphDiv.data('maxDate', data['info']['max_date']);
    graphDiv.data('factor', factor);

    if (data['dates'].length == 0) {
        //make sure there is no graph in this div
        Plotly.purge(graphDiv[0]);
        graphDiv.parent().show()
        graphDiv.html('<h2 class="noDataWarning">No data found for given parameters</h2>');
        return;
    } else {
        graphDiv.find('.noDataWarning').remove();
    }

    //store the earliest date for future reference
    graphDiv.data('start', data['dates'][0]);

    var freq_max10 = makePlotDataDict(data['dates'], data['freq_max10'])
    var sd_freq_max10 = makePlotDataDict(data['dates'], data['sd_freq_max10'], 2)
    var rsam = makePlotDataDict(data['dates'], data['rsam'], 3)
    const entropies=makePlotDataDict(data['entropy_dates'],data['entropies'], 4)

    const graph_data = [freq_max10, sd_freq_max10, rsam, entropies]
    const graph_labels=['Freq Max10 (Hz)', 'SD Freq Max10 (Hz)', 'RSAM', 'Shannon Entropy'];

    dest.find('div.dVv').remove();
    if(dVvStation!==null){
        dvv_div=$('<div class="dVv"></div>');
        graphDiv.after(dvv_div);

        const chartDiv=graphDiv.closest('.chart');
        const sta1=chartDiv.find('.stationName').text();

        //dVV Heatmap
        const dvv_info=makedVvHeatmap(data['heatX'],data['heatY'],data['heatZ'], 1);
        const dvv=dvv_info[0];
        const layout=dvv_info[1];
        const title=`0.5-5.0 Hz, dv/v, Average, ${sta1}_${dVvStation}`;
        layout['title']=title;

        //coherence heatmap
        const coh_info=makedVvHeatmap(data['cohX'],data['cohY'],data['cohZ'], 2);
        const coh=coh_info[0];
        const coh_layout=coh_info[1];
        const coh_title=`0.5-5.0 Hz, coherence, Average, ${sta1}_${dVvStation}`;
        coh_layout['title']=coh_title;
        coh['zmax']=1;
        coh['colorscale']=rdYlGnColorscale;

        //dvv curves
        const [dvvCurve,dvvCurveLayout]=dvvCurvePlot(data['dvvCurveDates'], data['dvvCurves']);
        const dvvCurveTitle=`${sta1}-${dVvStation} dvv`
        dvvCurveLayout['title']=dvvCurveTitle;

        //Coherence curves
        const [cohCurve,cohCurveLayout]=dvvCurvePlot(data['cohCurveDates'], data['cohCurves']);
        const cohCurveTitle=`${sta1}-${dVvStation} Coherence`
        cohCurveLayout['title']=cohCurveTitle;
        cohCurveLayout['yaxis']['title']='coherence';
        cohCurveLayout['yaxis']['range']=[0,1];

        //dvv vs coherence curve
        const [dvvCohCurve,dvvCohLayout]=dvvCohCurvePlot(data['dvvCurveDates'],data['dvvCurves'],data['cohCurves']);
        const dvvCohTitle=`${sta1}-${dVvStation} dvv and their coherence`;
        dvvCohLayout['title']=dvvCohTitle;

        plotDVV(
            dvv_div,
            [dvv,coh,dvvCurve,cohCurve,dvvCohCurve],
            [layout,coh_layout,dvvCurveLayout,cohCurveLayout,dvvCohLayout]
        );
    }
    else{
        graphDiv.after('<div class="dVv">To plot dv/v, select a station from the pull-down in the title</div>');
    }

    var layout = generateSubgraphLayout(graph_data, graph_labels);

    var annotation = [{
        "xref": 'paper',
        "yref": 'paper',
        "x": -0.01,
        "xanchor": 'left',
        "y": 1.005,
        "yanchor": 'bottom',
        "text": `Channel: ${channel}`,
        "showarrow": false,
        "font": { "size": 12 }
    }]

    layout['annotations'] = annotation;

    var filename = station + " Volcano Seismology ";
    filename += data['dates'][0].replace(/\//g, '-') + " to ";
    filename += data['dates'][data['dates'].length - 1].replace(/\//g, '-');

    plotGraph(graphDiv, graph_data, layout, {
        responsive: true,
        'toImageButtonOptions': {
            format: 'png',
            'height': 800,
            'width': 600,
            'scale': 1.5,
            filename: filename
        }
    });

    //If no dateFrom has been entered, default to one year. Otherwise, use the date range entered.
    var dateFrom = dest.find('input.dateFrom');
    if (dateFrom.val() == "")
        dest.find('span.dateBtns button.default').click();
    else {
        setGraphRange.call(dateFrom[0]);
    }

    Plotly.Plots.resize(graphDiv[0]);
    setTitle(dest);
}

function plotDVV(div,data,layout){
    for(let idx=0;idx<data.length;idx++){
        let dest=$('<div class="dvvPlotDest">');
        div.append(dest);

        let d=data[idx];
        // if it has a length, it's already an array
        if(d.length===undefined){
            d=[d];
        }

        let l=layout[idx];
        Plotly.newPlot(dest.get(0),d,l);
    }
}

function dvvCurvePlot(x,y,idx){
    const datas=[]
    for(const [title,values] of Object.entries(y)){
        let trace={
            x:x,
            y:values,
            type:'scatter',
            mode:'lines',
            name:title,
            connectgaps:false
        }

        datas.push(trace)
    }

    const layout={
        yaxis:{
            title:"dv/v(%)"
        }
    }

    return [datas,layout]
}

function dvvCohCurvePlot(x,y,color,idx){
    const colors = ['blue', 'red','green', 'purple','grey'];
    const datas=[];
    let i=0;
    for(const [title,values] of Object.entries(y)){
        let cval=color[title];
        let trace={
            x:x,
            y:values,
            type:'scatter',
            mode:'markers',
            marker:{
                color: colors[i],
                opacity:cval,
            },
            line:{
                color:colors[i]
            },
            name:title,
            connectgaps:false
        }

        datas.push(trace)
        i++;
    }

    const transparencyValues = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    const barTrace={
        x:[null],
        y:[null],
        mode:'markers',
        marker:{
            color: transparencyValues,
            colorscale:[[0,'rgba(0,0,0,0)'],[1,'rgba(0,0,0,1']],
            cmin:0,
            cmax:1,
            showscale:true,
            colorbar:{
                title:'Coherence Value',
                titleside:'right',
                ticksuffix: '',
                tickvals:transparencyValues,
                thickness:15
            }
        },
        showlegend:false
    }

    datas.push(barTrace);

    const layout={
        yaxis:{
            title:"dv/v(%)",
            showline:true,
            mirror:true,
        },
        xaxis:{
            showline:true,
            mirror:true,
        },
        legend:{
            x:0.02,
            y:0.97,
            bgcolor:'#F3F3F3',
            bordercolor:'#DDDDDD',
            borderwidth:1
        }
    }

    return [datas,layout]
}

function makePlotDataDict(x, y, idx) {
    var trace = {
        x: x,
        y: y,
        type: 'scattergl',
        mode: 'markers',
        marker: {
            size: 2,
            color: 'rgb(55,128,256)'
        },
    }

    if (typeof(idx) !== 'undefined') {
        trace['yaxis'] = `y${idx}`;
        trace['xaxis'] = `x${idx}`;
    }

    return trace;
}

function makedVvHeatmap(x, y, z, idx) {
    const heatmap = {
        x: x,
        y: y,
        z: z,
        type: 'heatmap',
        zmin:0,
        zmax:0.5,
        colorscale:"Bluered",
        connectgaps:false,
        xperiod:1.8e+6,
        hoverongaps:false,
    }

    const heatLayout={
        yaxis:{
            range:[0.5,5]
        },
        plot_bgcolor:"#D3D3D3",
        margin: {
            t: 42, //because, of COURSE it is!
            pad: 0
        },
    }

    return [heatmap, heatLayout];
}

function generateSubgraphLayout(data, titles) {
    var path_idx = window.location.pathname.lastIndexOf("/");
    var path = path_idx > 0 ? window.location.pathname.substring(0, path_idx) : ''
    var img_path = `${window.location.origin}${path}/static/img`;
    //figure the x-axis bounds
    //x-axis is same for all plots
    var x_data = data[0]['x'];
    x_start = new Date(x_data[0])
    var x_range = [x_data[0], x_data[x_data.length - 1]]
    x_range = parseRangeDates(x_range)
    var layout = {
        "paper_bgcolor": 'rgba(255,255,255,1)',
        "plot_bgcolor": 'rgba(255,255,255,1)',
        showlegend: false,
        margin: {
            l: 50,
            r: 25,
            b: 25,
            t: 80,
            pad: 0
        },
        grid: {
            rows: titles.length,
            columns: 1,
            pattern: 'independent',
            'ygap': 0.05,
        },
        'font': { 'size': 12 },
        "images": [{
            "source": `${img_path}/logos.png`,
            "xref": "paper",
            "yref": "paper",
            "x": 1,
            "y": 1.008,
            "sizex": .3762,
            "sizey": .27,
            "xanchor": "right",
            "yanchor": "bottom"
        }],
    }


    for (var i = 1; i <= titles.length; i++) {
        var section_title = titles[i - 1];
        if (section_title == '') {
            continue;
        }

        var y_axis = 'yaxis' + i;
        var x_axis = 'xaxis' + i;

        layout[y_axis] = {
            zeroline: false,
            title: section_title,
            'gridcolor': 'rgba(0,0,0,.3)',
            'showline': true,
            'showgrid': false,
            'linecolor': 'rgba(0,0,0,.5)',
            'mirror': true,
            'ticks': "inside"
        }

        layout[x_axis] = {
            automargin: true,
            autorange: false,
            range: x_range,
            type: 'date',
            tickformat: "%m/%d/%y<br>%H:%M",
            hoverformat: "%m/%d/%Y %H:%M:%S",
            'gridcolor': 'rgba(0,0,0,.3)',
            'showline': true,
            'showgrid': false,
            'mirror': true,
            'linecolor': 'rgba(0,0,0,.5)',
            'ticks': "inside"
        }

        if (i != titles.length) {
            layout[x_axis]['matches'] = `x${titles.length}`;
            layout[x_axis]['showticklabels'] = false;
        }
    }

    return layout;
}

function showMenu() {
    if ($('#locations').is(':visible')) {
        hideMenu();
        return;
    }

    menuMode = true;
    $('#locations').slideDown({
        start: function() {
            $(this).css({
                display: "grid"
            })
        }
    });
}

function hideMenu() {
    $('#locations').slideUp({
        done: function() {
            $(this).css({
                display: ""
            })
        }
    });
    menuMode = false;
}

function getUrlVars() {
    var vars = {};
    window.location.href.replace(/[?&]+([^=&]+)=([^&]*)/gi, function(m, key, value) {
        vars[key] = value;
    });
    return vars;
}

function getTickFormat(timeSpan) {
    var tickformat = '%Y-%m-%d';
    if (timeSpan < 5) {
        tickformat = "%H:%M:%S"
    } else if (timeSpan <= 36 * 60) {
        tickformat = "%-m/%-d %-H:%M"
    }
    return tickformat;
}

function setVisibility() {
    var type = $(this).data('type');
    var show = $(this).is(':checked');
    $(markerGroups[type]).each(function() {
        if (show)
            this.setMap(map);
        else
            this.setMap(null);
    });
}

function parseRangeDates(xaxis_range, exact) {
    if (typeof(exact) === 'undefined') {
        exact = false;
    }
    var dateFrom;
    var dateTo;

    //not sure why I need this, but sometimes this comes back as a date object, and other times as a string.
    if (typeof(xaxis_range[0]) == "string") {
        if (exact) {
            dateFrom = new Date(xaxis_range[0].substring(0, 10) + 'T' + xaxis_range[0].substring(11));
            dateTo = new Date(xaxis_range[1].substring(0, 10) + 'T' + xaxis_range[1].substring(11));
        } else {
            dateFrom = new Date(xaxis_range[0].substring(0, 10));
            dateTo = new Date(xaxis_range[1].substring(0, 10) + 'T23:59:59');
        }
    } else {
        dateFrom = new Date(xaxis_range[0]);
        dateTo = new Date(xaxis_range[1]);

        if (!exact) {
            var dtMonth = dateTo.getMonth()
            var dtYear = dateTo.getFullYear()
            var dtDate = dateTo.getDate()
            dateTo.setUTCHours(23, 59, 59, 999)
            dateTo.setUTCDate(dtDate)
        }
    }

    if (isNaN(dateFrom) || isNaN(dateTo)) {
        console.error('Bad date from/to!');
        return [];
    }

    if (!exact) {
        //go from midnight UTC on the date from to end-of-day on dateTo
        dateFrom = new Date(dateFrom.setUTCHours(0, 0, 0, 0));
        //for date to, set to the last millisecond of the specified date
        dateTo.setUTCHours(23, 59, 59, 999);
    }
    return [dateFrom, dateTo];
}

var layoutCount = 0;

let markerLookup={};
function map_stations(stations) {
    markerLookup={};
    for (var station in stations) {
        var station_data = stations[station];

        var title_string = station;
        var point = new google.maps.LatLng(station_data['lat'], station_data['lng']);

        var text_color = "white";
        var stroke_color = 'white';

        var marker = new google.maps.Marker({
            icon: {
                path: google.maps.SymbolPath.CIRCLE,
                scale: 14,
                fillColor: '#4900F4',
                fillOpacity: .75,
                strokeColor: stroke_color,
                strokeWeight: 3
            },
            label: {
                text: station_data['name'],
                color: text_color,
                fontSize: '.8em'
            },
            position: point,
            clickable: true,
            title: title_string,
            zIndex: 99
        });

        let stationID=station_data['sta_id']

        $(marker).data('id', stationID);
        $(marker).data('name', station_data['name']);
        $(marker).data('channels', station_data['channels'])
        var site_id = station_data['site'].replace(' ', '').toLowerCase();
        $(marker).data('site', site_id);

        stationMarkers.push(marker);
        markerLookup[stationID]=marker;

        marker.setMap(map);

        google.maps.event.addListener(marker, 'click', showStationGraphs);
    }

}

function closeAllGraphs(){
    $('div.chart').find('img.closeBtn').each(closeGraph);
}

function closeGraph() {
    //"this" is the close button
    var graphDiv = $(this).closest('div.chart').hide();
    Plotly.purge(graphDiv.find('div.graphArea')[0]);
    const tiltArea=graphDiv.find('div.tiltArea')
    if(tiltArea.length>0)
        Plotly.purge(tiltArea[0]);
    graphDiv.remove();
}

function dom_post(url, args) {
    iframe_id += 1;
    var target_frame = $(`<iframe id="loadFrame${iframe_id}" name="loadFrame${iframe_id}" style="display:none">`)
    target_frame.on('load', function() { console.log("Frame Loaded"); })
    target_frame.on('error', function() { console.log("Frame ERROR"); })
    $("body").append(target_frame)
    var form = $(`<form method="post" action=${url} target="loadFrame${iframe_id}">`)
    for (var key in args) {
        if (args.hasOwnProperty(key)) {
            var value = args[key]
            if (typeof(value) == "object") {
                value = JSON.stringify(value)
            }
            var field = $(`<input type="hidden" name="${key}">`)
            field.val(value)
            form.append(field)
        }
    }

    $("body").append(form)
    form[0].submit()

    // remove the frame after 10 minutes. If it takes more than 10 minutes to
    // generate and download the item, then this will cause breakage, but I
    // have to choose *some* time...

    setTimeout(function() {
        target_frame.remove();
    }, 600000);

    form.remove()
}

function showMessage(msg, cls) {
    var msgdiv = $('<div class="message" style="display:none">')
    var msgtext = $("<div class='msgtext'>")
    msgdiv.append(msgtext)
    msgdiv.addClass(cls)
    msgtext.html(msg)
    $("body").append(msgdiv);
    msgdiv.slideDown(function() {
        setTimeout(function() {
            msgdiv.slideUp(function() {
                msgdiv.remove();
            });
        }, 5000)
    });
}

var scale_len = null

var old_params;

function compObjects(object1, object2) {
    const keys1 = Object.keys(object1);
    const keys2 = Object.keys(object2);

    if (keys1.length !== keys2.length) {
        return false;
    }

    for (let key of keys1) {
        if (object1[key] !== object2[key]) {
            return false;
        }
    }

    return true;
}

function setMapLocation() {
    if (menuMode) {
        hideMenu();
    }

    $('.extraData').hide();
    $('#content').show();

    var volc = $(this).find('option:selected')
    var center = { lat: volc.data('lat'), lng: volc.data('lon') };
    var zoom = volc.data('zoom');

    //this will de-select the current tab
    map.setCenter(center);
    window.vectorGuard = true;
    map.setZoom(zoom);

    //close any plot divs
    closeAllGraphs();
    global_graph_div = null;

    // $(this).addClass('current');
    // getAnomaliesDebounce();
}

let anomTimer=null;
function getAnomaliesDebounce(){
    if(anomTimer!==null){
        clearTimeout(anomTimer);
    }

    anomTimer=setTimeout(getAnomalies,500);
}

function getBoundsFromLatLng(lat, lng, radiusInKm){
    const lat_change = radiusInKm/111.2;
    const kmperdegree=Math.abs(Math.cos(lat*(Math.PI/180)))*111.320;
    const lon_change=radiusInKm/kmperdegree;
    const bounds = {
        lat_min : lat - lat_change,
        lon_min : lng - lon_change,
        lat_max : lat + lat_change,
        lon_max : lng + lon_change
    };

    const min=new google.maps.LatLng(bounds['lat_min'], bounds['lon_min']);
    const max=new google.maps.LatLng(bounds['lat_max'], bounds['lon_max']);

    const mapBounds=new google.maps.LatLngBounds(min, max)
    map.fitBounds(mapBounds)
}

function getAnomalies(){
    anomTimer=null;
    let entropiesDiv=$('#entropiesPlots').empty();
    const volcs=[]
    $('#volcSelect option').each(function(){
        const volc=this.value;
        volcs.push(volc);
        const args={'volc':volc};
        const volcID=volc.replace(' ','');
        let volcDiv=$(`<div id=${volcID}AnomaliesTop class="volcAnomaliesTop">`);
        let titleDiv=$('<div class=title>')
        titleDiv.append(volc);
        titleDiv.append("<br>");
        titleDiv.append(`<img class="anomalyMap" src="static/img/maps/${volcID}.png">`)
        volcDiv.append(titleDiv)
        volcDiv.append(`<div id=${volcID}Anomalies class="volcAnomalies">`)
        entropiesDiv.append(volcDiv)
        // $.getJSON('listVolcEntropies',args)
        // .done(showAnomalies)
    })

    $.post('listEntropies',{'volcs':volcs})
    .done(showAnomalies);
}

function showAnomalies(data){
    for (let item of data){
        const stations=item['stations']
        volc=item['volc'].replace(' ','');

        const destDiv=$(`#${volc}Anomalies`);
        for(const [station, staData] of Object.entries(stations)){
            let img=staData['img'];
            let id=staData['staid'];

            destDiv.append(createEntropiesDiv(item['volc'],station,img,id));
        }
    }
}

function getEvents() {
    $.getJSON('listEventImages')
    .done(function(data){
        let eventsDiv=$('#eventPlots').empty()
        for(const volc in data){
            const volcID=volc.replace(' ','');
            let volcDiv=$(`<div id=${volcID}EventsTop class="volcAnomaliesTop">`);
            let titleDiv=$('<div class=title>')
            titleDiv.append(volc);
            titleDiv.append("<br>");
            titleDiv.append(`<img class="anomalyMap" src="static/img/maps/${volcID}.png">`)
            titleDiv.append(`<br><button class='downloadEvents' data-volc='${volc}'>Download Data</button>`)
            volcDiv.append(titleDiv)
            const destDiv=$(`<div id=${volcID}Events class="volcAnomalies">`)
            volcDiv.append(destDiv)
            eventsDiv.append(volcDiv)


            for(const i in data[volc]){
                console.log(i)
                const [station,image,stationid]=data[volc][i]
                const eventDiv = createEventsDiv(volc, station,image,stationid)
                destDiv.append(eventDiv)
            }
        }
    })
}

function createEventsDiv(volc, station,img,stationID){
    let eventTopDiv=$('<div class="anomaliesTop">');

    eventTopDiv.append(`<div class=title>${station}</div>`);
    eventTopDiv.data('stationID',stationID);
    eventTopDiv.data('volc',volc)
    eventTopDiv.on('click',function(event){
        const id=$(this).data('stationID');
        const volc=$(this).data('volc')
        showStationGraphs.call(markerLookup[id],event,volc);
    });

    let eventDiv = $('<div class="anomalies">');
    let eventImg = $('<img>');
    eventImg.on('error', function() {
            $(this).closest('div.anomaliesTop').addClass('error');
        })
        .on('load', function() {
            $(this).closest('div.anomaliesTop').removeClass('error');
        })
        .attr('src', img);

    eventDiv.append(eventImg);
    eventTopDiv.append(eventDiv);

    return eventTopDiv;
}

function setSpecialText() {
    if (menuMode) {
        hideMenu();
    }
    $('#content').hide();
    $('.extraData').hide();
    var target = $(this).data('target');
    $(`#${target}`).show();

    $('#locations div .tab').removeClass('current');
    $(this).addClass('current');
}
