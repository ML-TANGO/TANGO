from http.server import HTTPServer, BaseHTTPRequestHandler
import requests


class MyHandler(BaseHTTPRequestHandler):
  
    def do_GET(self):
        """
        process HTTP GET command

        Args: None
        Returns: None
        """
        pathlist = self.path.split('/')[1].split('?')
        cnt = len(pathlist)
        if cnt < 2:
            cmd = "unknown"
        else:
            ctmp = pathlist[1].split('&')
            mycnt = len(ctmp)
            cmd = pathlist[0]
            conid  = ctmp[0].split('container_id')[1].split('=')[1]
            userid = ctmp[1].split('user_id')[1].split('=')[1]
            prjid = ctmp[2].split('project_id')[1].split('=')[1]
            res = ctmp[3].split('result')[1].split('=')[1]
        print("cmd =", cmd)
        print("conid =", conid)
        print("userid =", userid)
        print("prjid =", prjid)
        print("result =", res)
        buf = "OK"
        self.send_response_only(200, 'OK')
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(buf.encode())
        return

 
####################################################################
####################################################################
if __name__ == '__main__':
    server = HTTPServer(('', 8085), MyHandler)
    print("Started WebServer on Port 8085")
    print("Press ^C to quit WebServer")
    try:
        server.serve_forever()
    except KeyboardInterrupt as e:
        time.sleep(1)
        server.socket.close()
        print("Project Manager Module End", e)

