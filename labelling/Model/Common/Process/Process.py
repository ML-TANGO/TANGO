# -*- coding: utf-8 -*-
'''
    각종 유틸 function 을 모아놓은 스크립트
'''
import sys
import os


def prcErrorData(file, msg):
    pid = os.getpid()
    msg = msg.replace("\n", " ")
    print(f'#_%{pid}&G&{os.path.basename(file)}&G&{True}&G&{msg}&G&0')
    sys.stdout.flush()


def prcSendData(file, msg):
    pid = os.getpid()
    print(f'#_%{pid}&G&{os.path.basename(file)}&G&{False}&G&{msg}&G&1')
    sys.stdout.flush()


def prcLogData(msg):
    pid = os.getpid()
    print(f'#_%{pid}&G&{os.path.basename(__file__)}&G&{False}&G&{msg}&G&0')
    sys.stdout.flush()


def prcGetArgs(pid):
    data = ''
    while True:
        char = sys.stdin.read(1024)
        if char != '':
            data = data + char
        else:
            break
    # prcSendData(pid, False, f'{data}, "의 작업을 시작합니다."')
    return data


def prcClose(file):
    prcSendData(file, 'END')
