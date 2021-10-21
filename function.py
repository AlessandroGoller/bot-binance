import json

# requires dateparser package
import dateparser
import pytz
from datetime import datetime
from datetime import date
import numpy as np

today = date.today()
def date_to_milliseconds(date_str):
    """Convert UTC date to milliseconds
    If using offset strings add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"
    See dateparse docs for formats http://dateparser.readthedocs.io/en/latest/
    :param date_str: date in readable format, i.e. "January 01, 2018", "11 hours ago UTC", "now UTC"
    :type date_str: str
    """
    # get epoch value in UTC
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    # parse our date string
    d = dateparser.parse(date_str)
    # if the date is not timezone aware apply UTC timezone
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)

    # return the difference in time
    return int((d - epoch).total_seconds() * 1000.0)

def milliseconds_to_date(ms):
    date=datetime.fromtimestamp(ms/1000.0)
    return date

def save_datafile(klines,symbol,interval,start,end=today):
        # open a file with filename including symbol, interval and start and end converted to milliseconds
    with open(
        "Binance_{}_{}_{}-{}.json".format(
            symbol, 
            interval, 
            #date_to_milliseconds(start),
            #date_to_milliseconds(end)
            start,end
        ),
        'w' # set file write mode
    ) as f:
        if type(klines).__module__ == np.__name__:
            klines=klines.tolist()
        f.write(json.dumps(klines))
                
