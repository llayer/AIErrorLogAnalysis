import re
import datetime
import calendar
import time
import os
import subprocess



def dateformat(value):
    
    PAT_YYYY = re.compile(r'^20[0-9][0-9]$')
    PAT_YYYYMMDD = re.compile(r'^20[0-9][0-9][0-1][0-9][0-3][0-9]$')
    
    """Return seconds since epoch for provided YYYYMMDD or number with suffix 'd' for days"""
    msg  = 'Unacceptable date format, value=%s, type=%s,' \
            % (value, type(value))
    msg += " supported format is YYYYMMDD or number with suffix 'd' for days"
    value = str(value).lower()
    if  PAT_YYYYMMDD.match(value): # we accept YYYYMMDD
        if  len(value) == 8: # YYYYMMDD
            year = value[0:4]
            if  not PAT_YYYY.match(year):
                raise Exception(msg + ', fail to parse the year part, %s' % year)
            month = value[4:6]
            date = value[6:8]
            ddd = datetime.date(int(year), int(month), int(date))
        else:
            raise Exception(msg)
        return calendar.timegm((ddd.timetuple()))
    elif value.endswith('d'):
        try:
            days = int(value[:-1])
        except ValueError:
            raise Exception(msg)
        return time.time()-days*24*60*60
    else:
        raise Exception(msg)
        
def hdate(date):
    "Transform given YYYYMMDD date into HDFS dir structure YYYY/MM/DD"
    date = str(date)
    return '%s/%s/%s' % (date[0:4], date[4:6], date[6:8])

def range_dates(trange):
    "Provides dates range in HDFS format from given list"
    out = [hdate(str(trange[0]))]
    if  trange[0] == trange[1]:
        return out
    tst = dateformat(trange[0])
    while True:
        tst += 24*60*60
        tdate = time.strftime("%Y%m%d", time.gmtime(tst))
        out.append(hdate(tdate))
        if  str(tdate) == str(trange[1]):
            break
    return out


def run_cmd(args_list):
        """
        Run a shell command
        """
        proc = subprocess.Popen(args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        s_output, s_err = proc.communicate()
        s_return =  proc.returncode
        return s_return, s_output, s_err
    

    
bad_dir = [
    # CHECK too large paths
    'hdfs:///cms/wmarchive/avro/fwjr/2017/06/22',
    
    # Empty paths
    'hdfs:///cms/wmarchive/avro/fwjr/2017/02/05',
    'hdfs:///cms/wmarchive/avro/fwjr/2017/02/06',
    'hdfs:///cms/wmarchive/avro/fwjr/2017/05/20',
    'hdfs:///cms/wmarchive/avro/fwjr/2017/05/21',
    'hdfs:///cms/wmarchive/avro/fwjr/2017/05/22',
    'hdfs:///cms/wmarchive/avro/fwjr/2017/07/01',
    'hdfs:///cms/wmarchive/avro/fwjr/2017/07/02',
    'hdfs:///cms/wmarchive/avro/fwjr/2017/07/03',
    'hdfs:///cms/wmarchive/avro/fwjr/2017/07/04',
    'hdfs:///cms/wmarchive/avro/fwjr/2017/07/05',
    'hdfs:///cms/wmarchive/avro/fwjr/2017/09/06',
    'hdfs:///cms/wmarchive/avro/fwjr/2017/09/29',
    'hdfs:///cms/wmarchive/avro/fwjr/2017/09/30',
    'hdfs:///cms/wmarchive/avro/fwjr/2017/10/01',
    'hdfs:///cms/wmarchive/avro/fwjr/2017/10/02',
    'hdfs:///cms/wmarchive/avro/fwjr/2017/10/08',
    'hdfs:///cms/wmarchive/avro/fwjr/2017/11/03',
    'hdfs:///cms/wmarchive/avro/fwjr/2017/11/04',
    'hdfs:///cms/wmarchive/avro/fwjr/2017/11/05',
    'hdfs:///cms/wmarchive/avro/fwjr/2017/11/11',
    'hdfs:///cms/wmarchive/avro/fwjr/2017/11/12',
    'hdfs:///cms/wmarchive/avro/fwjr/2018/08/04',
    'hdfs:///cms/wmarchive/avro/fwjr/2018/08/05',
    'hdfs:///cms/wmarchive/avro/fwjr/2018/08/17',
    'hdfs:///cms/wmarchive/avro/fwjr/2018/08/18',
    'hdfs:///cms/wmarchive/avro/fwjr/2018/08/19',
    'hdfs:///cms/wmarchive/avro/fwjr/2018/08/21'  
]

def getDirs( timerange ):
    
    hdir = 'hdfs:///cms/wmarchive/avro/fwjr/'

    pat=re.compile(".*/20[0-9][0-9].*")
    if  len(hdir.split()) == 1 and not pat.match(hdir):
        hdir = hdir.split()[0] 
        hdirs = []
        for tval in range_dates(timerange):
            if  hdir.find(tval) == -1:
                hdfs_file_path = os.path.join(hdir, tval)
                # check whether the hdfs path exists
                #cmd = ['hdfs', 'dfs', '-test', '-d', hdfs_file_path]
                #ret, out, err = run_cmd(cmd)
                #if ret == 0:
                #    hdirs.append(hdfs_file_path)
                #else:
                #    print "Path does not exist:", hdfs_file_path
                if not hdfs_file_path in bad_dir:
                    hdirs.append(hdfs_file_path)
        hdir = hdirs
        return hdir
