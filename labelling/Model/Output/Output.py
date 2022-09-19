import requests
import json


# send msg for node Server
def sendMsg(url, msg):
    headers = {"Content-Type": "application/json; charset=utf-8"}
    res = requests.post(url, headers=headers, data=json.dumps(msg))
    return res
