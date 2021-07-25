import inspect
from datetime import datetime

"""
a log would be like:
    date Time|function|log_Type|logText|
    09-12-2021 02:43|testFunc|ERROR|Could not find some value|
    09-12-2021 02:43|testFunc|LOG|Found value at some position|
"""

class Logger:
    def __init__(self, fname):
        self.fname = fname
    
    def get_current_ts(self):
        cts=datetime.isoformat(datetime.now())
        return cts[:-7]

    def log(self, logType, logText):
        f=open(self.fname, "a")
        cts = self.get_current_ts()
        f.write(cts+"|"+inspect.stack()[1][3]+"|"+str(logType)+"|"+str(logText)+"|\n")
        f.close()

