import json
import datetime
from dateutil import parser


class DataReader:
    def __init__(self, datatype, filepath, interval_minutes=5):
        self.datatype = datatype
        self.filepath = filepath
        self.interval_timedelta = datetime.timedelta(minutes=interval_minutes)

    def read(self):
        if self.datatype == "OH":
            return self.read_OH()

    def read_OH(self):
        with open(self.filepath, 'r') as file:
            data = json.load(file)

        res = []
        
        # Since data is in descending order, reverse it to process sequentially
        data.reverse()

        # Initialize with the first reading
        res.append([data[0]['sgv']])
        
        for i in range(1, len(data)):
            # print(data[i]['dateString'])
            t1 = parser.parse(data[i]['dateString'])
            t0 = parser.parse(data[i-1]['dateString'])

            delt = t1 - t0

            try:
                value = data[i]['sgv']
            except KeyError:
                value = data[i].get('mbg', None)  # If 'mbg' is also not present, it will assign None.

            if delt <= self.interval_timedelta:
                if value is not None:
                    res[-1].append(value)
            else:
                if value is not None:
                    res.append([value])
        return res

