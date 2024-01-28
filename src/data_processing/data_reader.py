import json
import datetime
from dateutil import parser
from dateutil.parser import ParserError
import xml.etree.ElementTree as ET


class DataReader:
    def __init__(self, datatype, filepath, interval_minutes=5):
        self.datatype = datatype
        self.filepath = filepath
        self.interval_timedelta = datetime.timedelta(minutes=interval_minutes)

    def read(self):
        if self.datatype == "OH":
            return self.read_OH()
        elif self.datatype == "OH_24h":
            return self.read_OH_24h()
        elif self.datatype == "ohio":
            return self.read_ohio()
        else:
            raise ValueError("Invalid datatype: {}".format(self.datatype))
        
    def read_ohio(self):
        tree = ET.parse(self.filepath)
        root = tree.getroot()

        res = []
        for item in root.findall("glucose_level"):
            entry0 = item[0].attrib
            res.append([float(entry0["value"])])
            for i in range(1, len(item)):
                last_entry = item[i - 1].attrib
                entry = item[i].attrib
                t1 = datetime.datetime.strptime(entry["ts"], "%d-%m-%Y %H:%M:%S")
                t0 = datetime.datetime.strptime(last_entry["ts"], "%d-%m-%Y %H:%M:%S")
                delt = t1 - t0
                if delt <= self.interval_timedelta:
                    res[-1].append(float(entry["value"]))
                else:
                    res.append([float(entry["value"])])
        return res

    def read_OH(self):
        with open(self.filepath, "r") as file:
            data = json.load(file)

        res = []

        # Since data is in descending order, reverse it to process sequentially
        data.reverse()
        try:
            value_0 = data[0]["sgv"]
        except KeyError:
            value_0 = data[0].get(
                    "mbg", None
                )
        except IndexError:
            value_0 = None

        if value_0 is not None:
            # Initialize with the first reading
            res.append([value_0])

            for i in range(1, len(data)):
                # print(data[i]['dateString'])
                try:
                    t1 = parser.parse(data[i]["dateString"]).replace(tzinfo=None)
                    t0 = parser.parse(data[i - 1]["dateString"]).replace(tzinfo=None)
                except (KeyError, ParserError):
                    # skip one reading if 'dateString' is not present or in wrong format
                    continue
                
                delt = t1 - t0

                try:
                    value = data[i]["sgv"]
                except KeyError:
                    value = data[i].get(
                        "mbg", None
                    )  # If 'mbg' is also not present, it will assign None.
                
                # Check if value is an empty string and set it to None
                if value == '':
                    value = None
                    
                if delt <= self.interval_timedelta:
                    if value is not None:
                        res[-1].append(value)
                else:
                    if value is not None:
                        res.append([value])

        return res
    
    def read_OH_24h(self):
        """ a similar method to read_OH,
        but this time we are reading 24 hours of data in each sequence.

        gets t0 and when t1 - t0 > 24 hours, we start a new sequence.
        """

        with open(self.filepath, "r") as file:
            data = json.load(file)

        res = []

        # Since data is in descending order, reverse it to process sequentially
        data.reverse()
        try:
            value_0 = data[0]["sgv"]
            
        except KeyError:
            value_0 = data[0].get(
                    "mbg", None
                )
        except IndexError:
            value_0 = None

        try:
            t0 = parser.parse(data[0]["dateString"]).replace(tzinfo=None)
        except (IndexError, KeyError, ParserError):
            value_0 = None

        if value_0 is not None:
            # Initialize with the first reading
            # cast t0 to  a float of the seconds of the day
            t0_to_save = t0 # t0.hour * 3600 + t0.minute * 60 + t0.second
            res.append([(value_0, t0_to_save)])
            # set t0 to the beginning of the day
            t0 = t0.replace(hour=0, minute=0, second=0, microsecond=0)


            for i in range(1, len(data)):
                # print(data[i]['dateString'])
                try:
                    t1 = parser.parse(data[i]["dateString"]).replace(tzinfo=None)
                except (KeyError, ParserError, IndexError):
                    # skip one reading if 'dateString' is not present or in wrong format
                    continue
                
                delt = t1 - t0

                try:
                    value = data[i]["sgv"]
                except KeyError:
                    value = data[i].get(
                        "mbg", None
                    )
                except IndexError:
                    value = None
                
                # Check if value is an empty string and set it to None
                if value == '':
                    value = None
                
                if delt <= datetime.timedelta(hours=24):
                    if value is not None:
                        t1_to_save = t1.hour * 3600 + t1.minute * 60 + t1.second
                        res[-1].append((value, t1_to_save))
                else:
                    if value is not None:
                        t1_to_save = t1 # t1.hour * 3600 + t1.minute * 60 + t1.second
                        res.append([(value, t1_to_save)])
                        # update t0
                        t0 = t1
                        t0 = t0.replace(hour=0, minute=0, second=0, microsecond=0)

        return res
