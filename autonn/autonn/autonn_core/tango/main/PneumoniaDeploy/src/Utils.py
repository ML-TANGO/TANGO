# -*- coding: utf-8 -*-

from pathlib import Path
import os, platform, sys, time, json, base64, datetime, glob, hashlib
import re, shutil

def printJson(jsonData):
    ''' Function  printJson'''
    print(json.dumps(jsonData, indent=4))

def isFileExist(_file):
    return os.path.exists(_file)

def copyFile(src, dst):
    ''' Function  copyFile'''
    shutil.copyfile(src, dst)

"""
def getFilePath(_fullPath)

파일이 저장된 경로만 리턴해준다.
"""
def getFilePath(_fullPath):
    return os.path.split(_fullPath)[0]

def checkAndDeleteFile(_file):
    tryCount = 0
    while True:
        if isFileExist(_file):
            # logPrint("파일 삭제 : {}".format(_file))
            try:
                os.remove(_file)
                return
            except:
                logPrint("({}) 파일 삭제 중 오류 발생: {}".format(tryCount+1, _file))
                tryCount += 1  # 재시도
                if tryCount == 10:
                    logPrint("삭제 실패: {}".format(_file))
                    return



def changeExt(_file, oldExt, newExt):
    return _file.replace(oldExt, newExt)

"""
def getFileName(_fullPath)

파일의 이름만 리턴해준다.
"""
def getFileName(_fullPath):
    return os.path.split(_fullPath)[1]

def getFileNameWithoutExt(_filaname):
    return os.path.splitext(_filaname)[0]


def makeDirectory(path):
    ''' Function  makeDirectory'''
    if os.path.isdir(path) == False:
        os.mkdir(path)
        return True
    return False

"""
def getJsonDataFromFile(_fullPath)

JSON 파일의 데이터를 리턴해준다.
"""
def readJsonFile(_fullPath):
    try:
        jsonData = dict()
        if not Path(_fullPath).is_file():
            return None
        else:
            with open(_fullPath, encoding="UTF-8") as infile:
                fileData = infile.read()
                jsonData = json.loads(fileData)
        
        return jsonData
    except:
        return None

def getJsonDataFromFile(_fullPath):
    # Utils.logPrint(infoFile)
    jsonData = None
    retryCount = 0
    while jsonData == None:
        if isFileExist(_fullPath):
            # print("File existed")
            jsonData = readJsonFile(_fullPath)
        if jsonData == None:
            logPrint("================ JSON ERROR : {}".format(_fullPath))
            time.sleep(0.01)
            if retryCount == 10:
                break
            retryCount += 1
        else:
            break
    return jsonData

"""
def getJsonDataFromPathAndFile(_path, _file)

지정된 경로내의 지정된 JSON 파일의 데이터를 리턴해준다.
"""
def getJsonDataFromPathAndFile(_path, _file):
    if hasPathSeparator(_path) == True:
        jsonPath = _path + _file
    else:
        jsonPath = _path + getSystemPathSeparator() + _file
    jsonData = dict()
    if not Path(jsonPath).is_file():
        return None
    else:
        with open(jsonPath, encoding="UTF-8") as infile:
            fileData = infile.read()
            jsonData = json.loads(fileData)
    
    return jsonData

def getSystemPathSeparator():
    if isWindows():
        return "\\"
    else:
        return "/"

# 경로 끝에 \나 /가 있는 지 검사
def hasPathSeparator(_path):
    l = len(_path)
    if _path[l-1] == "\\" or _path[l-1] == "/":
        return True
    return False

def isWindows():
    if platform.system() == "Windows":
        return True
    else:
        return False

def saveJsonData(filePath, jsonData):
    with open(filePath, 'w', encoding="UTF-8") as outfile:
        json.dump(jsonData, outfile, ensure_ascii=False, indent=4)

def mapFloatToString(value):
    # print(value)
    strMsg = ""

    if value < 0:
        strMsg = "-"
        value = -value
    else:
        strMsg = ""

    DegInt = int(value)
    Min = (value - DegInt) * 60
    MinInt = int(Min)
    Sec = (Min - MinInt) * 60
    SecInt = round(Sec)

    strMsg = "%d°%02d'%02d\"" % (DegInt, MinInt, SecInt)
    return strMsg

def longitudeToString(longitude):
    strMsg = ""
    if longitude < 0:
        # if(hencIsNationalLanguage())
        strMsg = strMsg + "서경 "
        # else strMsg += "W ";
        longitude = -longitude
    else:
        #if(hencIsNationalLanguage())
        strMsg = strMsg + "동경 "
        # else strMsg += "E ";

    strMsg = strMsg + mapFloatToString(longitude) + " "
    
    return strMsg

def latitudeToString(latitude):
    strMsg = ""
    if latitude < 0:
        #if(hencIsNationalLanguage())
        strMsg = strMsg + "남위 "
        # else strMsg += "S ";
        latitude = -latitude
    else:
        #if(hencIsNationalLanguage())
        strMsg = strMsg + "북위 "
        #else strMsg += "N ";

    strMsg = strMsg + mapFloatToString(latitude)

    return strMsg

def ByteArrayToHex( byteStr ):
    """
    Convert a byte string to it's hex string representation e.g. for output.
    """
    
    # Uses list comprehension which is a fractionally faster implementation than
    # the alternative, more readable, implementation below
    #   
    #    hex = []
    #    for aChar in byteStr:
    #        hex.append( "%02X " % ord( aChar ) )
    #
    #    return ''.join( hex ).strip()        

    return ''.join( [ "%02X " % x for x in byteStr ] ).strip()

def getBase64Str(s):
    bytesStr = s.encode('UTF-8')
    utf64Str = base64.b64encode(bytesStr)
    bstr = utf64Str.decode(encoding='UTF-8')
    return bstr

def getStrFromBase64Str(bs):
    bstr = bs.encode(encoding='UTF-8')
    byteStr = base64.b64decode(bstr)
    s = byteStr.decode('UTF-8')
    return s

def getCurrentDatetimeStrForPngName():
    return datetime.datetime.today().strftime('%Y%m%d_%H%M%S')

def getCurrentDatetimeStr():
    return datetime.datetime.today().strftime('%Y/%m/%d %H:%M:%S')

def getCurrentTxDatetimeStr():
    return datetime.datetime.today().strftime('%Y%m%d_%H%M%S')

def getFileList(path):
    return glob.glob("{}/*".format(path))

def getCurrentDateTime():
    now = datetime.datetime.now()
    Y = now.year
    M = now.month
    D = now.day
    h = now.hour
    m = now.minute
    s = now.second

    return [Y,M,D,h,m,s]

def logPrint(log, logWriter="이상호"):
    if type(log).__name__ == "dict" or type(log).__name__ == "list":
        print(json.dumps(log, indent=4))
    else:
        print("[{}] - [{}] {}".format(logWriter, getCurrentDatetimeStr(), log))
    return

# 오늘 날짜의 유닉스 타임 구하기, 시분초가 0이므로 자정에 대한 값이 된다.
def getUnixDateOfToday():
    now = datetime.datetime.now()
    today = datetime.datetime(now.year, now.month, now.day)
    cud = int(time.mktime(today.timetuple()))
    return cud

# 오늘 날짜 현재 시간의 유닉스 타임 구하기
def getUnixDatetimeOfToday():
    now = datetime.datetime.now()
    today = datetime.datetime(now.year, now.month, now.day, now.hour, now.minute, now.second)
    cudt = int(time.mktime(today.timetuple()))
    return cudt

# 특정 날짜의 유닉스 타임 구하기
def getUnixDateOfSomeday(year, month, day):
    thatday = datetime.datetime(year, month, day)
    tud = int(time.mktime(thatday.timetuple()))
    return tud

# 특정 날짜/시간/분의 유닉스 타임 구하기
def getUnixDateOfSometime(year, month, day, hour, minute):
    thatday = datetime.datetime(year, month, day, hour, minute)
    tud = int(time.mktime(thatday.timetuple()))
    return tud

# Unixdatetime으로부터 datetime 구하기
def getDatetimeFromUnixDatetime(ud):
    return datetime.datetime.fromtimestamp(ud)

# Unixdatetime으로부터 YYYY-MM-DD 문자열 구하기
def getDateStrFromUnixDatetime(ud):
    dt = getDatetimeFromUnixDatetime(ud)
    return "%04d-%02d-%02d" % (dt.year, dt.month, dt.day)

# Unixdatetime으로부터 YYYYMMDDhhmmss 딕셔너리 구하기
def getYMDHMSFromUnixDatetime(ud):
    dt = getDatetimeFromUnixDatetime(ud)
    return { "year": dt.year, "month": dt.month, "day": dt.day, "hour": dt.hour, "minute": dt.minute, "second": dt.second }

# YYYYMMDDhhmmss to YYYY/MM/DD hh:mm:ss
def YMD_to_str(ymd):
    return "{}/{}/{} {}:{}:{}".format(ymd[0:4], ymd[4:6], ymd[6:8], ymd[8:10], ymd[10:12], ymd[12:14])

def getSHA256(string):
    hash_object = hashlib.sha256(string.encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig

def isValidMp4(filePath):
    f = open(filePath, "rb")
    headerData = f.read(12)
    print(headerData)
    f.seek(-13, 2)
    tailData = f.read(14)
    print(tailData)
    f.close()
    if headerData == b"\x00\x00\x00 ftypisom" and tailData == b"Lavf60.10.100":
        return True
    return False

def renameFile(dst, src):
    os.rename(src, dst)

def is_valid_id(id):
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
    return bool(re.match(pattern, id))