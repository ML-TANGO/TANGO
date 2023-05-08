import time
import socket
# for web service
import requests
from http.server import HTTPServer, SimpleHTTPRequestHandler
import detect
import os
def_port = 8901

##################################################################
# code for neural network
##################################################################
def call_this():
    ## code here
    res = detect.run(weights=os.environ['MODEL'], data= 'data/'+os.environ['ANN'], source='images')
    return res


####################################################################
# class for code generationpython cors header request origin
####################################################################
class YoloService():

    ####################################################################
    def __init__(self):
        return

    ####################################################################
    def run(self):
        # call neural net function
        res = call_this() 
        return res

####################################################################
# class for HTTP server
####################################################################
class MyHandler(SimpleHTTPRequestHandler):
    """Web Server definition """
    m_deploy_obj = YoloService()
    allowed_list = ('0,0,0,0', '127.0.0.1')
    
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
        if self.path[1] == '?':
            t_path = "%s%s" % ('/', self.path[2:])
        else:
            t_path = self.path
        pathlist = t_path.split('/')[1].split('?')
        print(pathlist)
        cmd = pathlist[0]
        print("cmd =", cmd)

        if cmd == "run":
            buf = self.m_deploy_obj.run()
            self.send_response(200, 'ok')
            self.send_cors_headers()
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(buf.encode())
            
        elif cmd == "stop":
            buf = "end"
            self.send_response(200, 'ok')
            self.send_cors_headers()
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(buf.encode())
            time.sleep(1)
            raise KeyboardInterrupt
        return

####################################################################
####################################################################
if __name__ == '__main__':
    server = HTTPServer(('', def_port), MyHandler)
    print("Started neural net service...")
    print("Press ^C to quit WebServer")

    try:
        server.serve_forever()
    except KeyboardInterrupt as e:
        time.sleep(1)
        server.socket.close()
        print("neural net service  End", e)

