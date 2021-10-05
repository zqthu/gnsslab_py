import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

useprms=('STAX','STAY','STAZ','VELX','VELY','VELZ')
nprm=6

recordre=re.compile(r'''^
     \s([\s\d]{5})  # param id
     \s([\s\w]{6})  # param type
     \s([\s\w]{4}|[\s\-]{4})  # point id
     \s([\s\w]{2}|[\s\-]{2})  # point code
     \s([\s\w]{4}|[\s\-]{4})  # solution id
     \s(\d\d\:\d\d\d\:\d\d\d\d\d) #parameter epoch
     \s([\s\w\/]{4})  # param units
     \s([\s\w]{1})  # param constraints
     \s([\s\dE\+\-\.]{21})  # param value
     \s([\s\dE\+\-\.]{11})  # param stddev
     \s*$''', re.IGNORECASE | re.VERBOSE )
    
epochre=re.compile(r'^(\d\d)\:(\d\d\d)\:(\d\d\d\d\d)$')


class Sta():
    def __init__(self, staid, epochstr):
        self.staid = staid
        self.refresh(epochstr)
    
    def refresh(self, epochstr):
        self.epochstr = epochstr
        self.epoch = sinexEpoch(epochstr)
        self.value_array = [0,0,0,0,0,0] # x,y,z,vx,vy,vz

def sinexEpoch(epochstr):
    match=epochre.match(epochstr)
    if not match:
        print("Epoch Error")
    y,d,s=(int(d) for d in match.groups())
    if y==0 and d==0 and s==0:
        return None
    year=y+1900 if y > 50 else y+2000
    return datetime(year,1,1)+timedelta(days=d-1,seconds=s)

def read_sinex_posvel(sinex_file):
    sinex_f = open(sinex_file)
    sinex_data = sinex_f.readlines()
    sinex_f.close()
    
    li_sta = []
    li_staid = []
    
    for i in range(len(sinex_data)):
        line = sinex_data[i]
        match=recordre.match(line)
        if "-SOLUTION/ESTIMATE" in line:
            break
        if match is None:
            continue
        prmid,prmtype,ptid,ptcode,solnid,epochstr,units,constraint,value,stddev=(x.strip() for x in match.groups())

        try:
            prmno=useprms.index(prmtype)
        except:
            print("%s not in tuple" % prmtype)
            continue
        prmid=int(prmid)

        if ptid not in li_staid:
            li_staid.append(ptid)
            new_sta = Sta(ptid, epochstr)
            li_sta.append(new_sta)
        idx = li_staid.index(ptid)
        sta = li_sta[idx]
        if sinexEpoch(epochstr) > sta.epoch:
            sta.refresh(epochstr) # latest epoch
        sta.value_array[prmno] = float(value)

    # parsed = pd.DataFrame(columns=["staid","epochstr","epoch","x","y","z","vx","vy","vz"])
    parsed = pd.DataFrame(columns=["staid","epoch","x","y","z","vx","vy","vz"])
    for i in range(len(li_staid)):
        sta = li_sta[i]
        # data_line = [sta.staid,sta.epochstr,sta.epoch]+sta.value_array    
        # new_df = pd.DataFrame([data_line], columns=["staid","epochstr","epoch","x","y","z","vx","vy","vz"])
        data_line = [sta.staid,sta.epoch]+sta.value_array    
        new_df = pd.DataFrame([data_line], columns=["staid","epoch","x","y","z","vx","vy","vz"])
        parsed = pd.concat([parsed, new_df])
    parsed.reset_index(inplace=True, drop=True)
    return parsed