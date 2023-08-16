from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Thread
import requests
import time

# prj_mng port
def_send_port = 8085
# def_send_port = 8089
def_send_url = "%s%4d" % ("http://0.0.0.0:", def_send_port)

# my port
def_recv_port = 8888
# def_recv_port = 8085
my_url = "%s%4d" % ("http://0.0.0.0:", def_recv_port)


def send_thr():
    """
    send_thr()
    """

    method_name = 'GET'

    for i in range(20):
        print("0= ready_request, 1=start, 2=stop, 3=state_request")
        x = input()
        if x[0] == '0':
            url = "%s%s" % (def_send_url, '/status_request')
            data = 'user_id=""&project_id=""'
        elif x[0] == '1':
            url = "%s%s" % (def_send_url, '/start')
            data = 'user_id=kyunghee&project_id=1'
        elif x[0] == '2':
            url = "%s%s" % (def_send_url, '/stop')
            data = 'user_id=kyunghee&project_id=1'
        elif x[0] == '3':
            url = "%s%s" % (def_send_url, '/status_request')
            data = 'user_id=kyunghee&project_id=1'
        else:
            continue

        headers = {
            'Host': "%s%4d" % ("0.0.0.0:", def_send_port),
            'Origin': my_url,
            'Accept': "application/json, text/plain",
            'Access-Control_Allow_Origin': '*',
            'Access-Control-Allow-Credentials': "true"
        }

        response = requests.get(url=url, headers=headers, params=data)
        print(response)

        dict_meta = {'status_code': response.status_code, 'ok': response.ok,
                     'encoding': response.encoding,
                     'Content-Type': response.headers['Content-Type'],
                     'text': response.text}

        if dict_meta['ok']:
            print("RESPONSE SUCCESS")
            print(dict_meta['text'])
            # 성공 응답 시 액션
        else:
            print("RESPONSE ERROR")
            # 실패 응답 시 액션


class MyHandler(SimpleHTTPRequestHandler):
    """
    MyHandler
    """

    def send_cors_headers(self):
        """
        send header for CORS

        Args: None
        Returns: None
        """
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header('Referrer-Policy', 'same-origin')
        self.send_header("Access-Control-Allow-Methods", "GET, OPTION")
        self.send_header("Access-Control-Allow-Credentials", "true")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Origin, Accept, token")

    def do_GET(self):
        """
        process HTTP GET command

        Args: None
        Returns: None
        """
        print(self.path)
        pathlist = self.path.split('/')[1].split('?')
        cnt = len(pathlist)
        if cnt < 2:
            cmd = "unknown"
        else:
            cmd = pathlist[0]
            print('cmd = %s' % cmd)
            conid = ''
            userid = ''
            prjid = ''
            res = ''

            ctmp = pathlist[1].split('&')
            mycnt = len(ctmp)
            for i in range(mycnt):
                tval = ctmp[i].split('=')[0]
                if tval == 'container_id':
                    conid = ctmp[i].split('=')[1]
                elif tval == 'user_id':
                    userid = ctmp[i].split('=')[1]
                elif tval == 'project_id':
                    prjid = ctmp[i].split('=')[1]
                elif tval == 'result':
                    res = ctmp[i].split('=')[1]
                else:
                    print('unkown path')

        print("cmd =", cmd)
        print("conid =", conid)
        print("userid =", userid)
        print("prjid =", prjid)
        print("result =", res)
        buf = "OK"
        self.send_response(200, 'ok')
        self.send_cors_headers()
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(buf.encode())
        return


####################################################################
####################################################################
if __name__ == '__main__':
    th1 = Thread(target=send_thr)
    th1.start()

    server = HTTPServer(('', def_recv_port), MyHandler)
    print("Started WebServer on Port 8085")
    print("Press ^C to quit WebServer")
    try:
        server.serve_forever()
    except KeyboardInterrupt as e:
        time.sleep(1)
        server.socket.close()
        print("Project Manager Module End", e)
