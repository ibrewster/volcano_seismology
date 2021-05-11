$(document).ready(function() {
    $(document).on('keypress', '.date', formatDate);
});

function parseISODate(iso_date) {
    var timestamp_parts = iso_date.split('T');
    var date_parts = timestamp_parts[0].split('-');
    var time_parts = timestamp_parts[1].split(':');
    if (time_parts[2].endsWith('Z')) {
        time_parts[2] = time_parts[2].slice(1, -1);
    }
    var date = new Date(date_parts[0], date_parts[1] - 1, date_parts[2],
        time_parts[0], time_parts[1], time_parts[2]);
    return date;
}

function parseLocalDate(local_date) {
    var date_parts = local_date.split('/'); //month, day, year
    var date = new Date(date_parts[2], date_parts[0] - 1, date_parts[1]);
    return date;
}

function formatFakeISODateString(date) {
    //because the system sometimes needs "LOCAL" time to pretend to be UTC...
    if (isNaN(date)) {
        return "";
    }
    var day = date.getDate();
    var month = date.getMonth() + 1;
    var year = date.getFullYear();
    var hour = date.getHours();
    var minute = date.getMinutes();
    var second = date.getSeconds();

    return year + '-' + month + '-' + day + 'T' + hour + ':' + minute + ':' + second + 'Z';
}

function formatISODateString(date) {
    if (isNaN(date)) {
        return "";
    }
    var day = date.getUTCDate();
    if (day < 10) {
        day = '0' + day;
    }

    var month = date.getUTCMonth() + 1;
    if (month < 10) {
        month = '0' + month;
    }
    var year = date.getUTCFullYear();
    var hour = date.getUTCHours();
    if (hour < 10) {
        hour = '0' + hour;
    }
    var minute = date.getUTCMinutes();
    if (minute < 10) {
        minute = '0' + minute;
    }

    var second = date.getUTCSeconds();
    if (second < 10) {
        second = '0' + second;
    }

    return year + '-' + month + '-' + day + 'T' + hour + ':' + minute + ':' + second + 'Z';
}

function formatDateString(date, short, seperator) {
    if (typeof(short) == 'undefined') { short = false }
    if (typeof(seperator) == 'undefined') { seperator = '/' }

    if (isNaN(date)) {
        return "";
    }
    var day = date.getUTCDate();
    var month = date.getUTCMonth() + 1;
    var year = date.getUTCFullYear();
    if (short) {
        year = (year + '').substr(2, 2);
    }

    return month + seperator + day + seperator + year;
}

function numbersOnly(event, filter) {
    if (typeof filter == 'undefined')
        filter = /[^0-9\/\-\.]/;

    var object = $(event.target);
    var keyPressed = (event.keyCode || event.which);
    var valid_key = true;
    if (typeof event.which == "number" && event.which < 32) //delete, tab, etc
        return true;

    var keyChar = String.fromCharCode(keyPressed);
    var curValue = object.val();

    //make sure the key pressed is valid for a number - that is, a digit from 0-9, a negative sign, or a decimal
    //we also allow / so we can use this same filter as a basis for a date filter.
    if (keyChar.match(filter)) {
        valid_key = false;
    }
    //if we got this far, then the character itself is nominally valid. Now let's make sure
    //the usage is valid. For this, we need the current value of the input.
    else if (curValue != null) //if we can't get the value, just stop here to prevent errors. May lead to bad input, but we'll live
    {
        //First, negatives can only go at the beginning.
        if (keyChar == "-" && curValue.length != 0) //already have other characters. Can't type a -
        {
            valid_key = false;
        }
        //check for valid decimal usage
        else if (keyChar == ".") {
            if (curValue.length == 0 || curValue == '-') //can not be the first character, or right after a leading negative. Lead with a zero.
            {
                object.val(object.val() + "0");
            } else if (curValue.indexOf(".") != -1) //can not have more than 1 decimal
                valid_key = false;
        }
    }

    if (!valid_key) {
        if (event.preventDefault) {
            event.preventDefault();
        } else {
            event.returnValue = false;
        }
        return false;
    }
}

function formatDate(event) {
    var field = this;
    //Input Validation - only allow numbers and separators
    var keyMatch = numbersOnly(event);
    if (typeof keyMatch != 'undefined')
        return keyMatch;


    var keyChar = String.fromCharCode((event.keyCode || event.which));

    //formatting - add separators if needed
    //if they typed a separator do some special handling
    if (keyChar == "/") {
        //count the number of slashes already entered
        var count = field.value.match(/\//g);

        //Allow slashes to be typed, except for the following:
        //Two slashes in a row
        //More than two slashes total
        //slash as the first character
        if (field.value.match(/\/$/) ||
            (count && count.length == 2) ||
            field.value.length < 1
        ) {
            if (event.preventDefault) {
                event.preventDefault();
            } else {
                event.returnValue = false;
            }
            return false
        } else
            return true;
    } else if (keyChar == '.' || keyChar == '-') //we allowed these for numbers, but don't allow for dates
    {
        if (event.preventDefault) {
            event.preventDefault();
        } else {
            event.returnValue = false;
        }
        return false
    }

    //check for situations where a slash should be automatically added
    var firstSlashPos = field.value.indexOf("/");
    var len = field.value.length;

    if (!field.value.match(/\/$/)) //don't add a slash if the field ends in one already
    {
        if ((len == 2 && firstSlashPos == -1) || //third number typed, no separator - enter separator
            (len == 4 && firstSlashPos == 1) || //one digit month, two digit day
            (len == 5 && firstSlashPos == 2)
        )
            field.value = field.value + "/";
    }
}