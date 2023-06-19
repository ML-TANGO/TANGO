"""
k8s_deploy.py
This module deploys neural network program to k8s
"""
# k8s_deploy.py
import deployment
import yaml
import shutil
import time

# for system calls
import os
import socket


# for web service
import requests
from http.server import HTTPServer, SimpleHTTPRequestHandler


# for docker and project manager
# "." for test /tango for docker container
#def_top_folder = "./tango/common"  # for test
def_top_folder = "/tango/common"    # for docker

def_deployinfo_file = "deployment.yaml"
def_zip_file = "nn_model.zip"
def_code_folder_name = "nn_model_2"

def_deploy_port = 8902


####################################################################
# class for code generationpython cors header request origin
####################################################################
class K8SDeploy:
    """Class Definition for K8SDeploy """
    # for docker & project manager
    m_current_file_path = def_top_folder
    m_current_userid = ""
    m_current_projectid = ""
    m_current_code_folder = "."
    m_nfs_path = ""
    m_nfs_ip = ""
    m_deployinfo_file = def_deployinfo_file
    m_arch_type = 'x86'
    m_acc_type = 'cpu'
    m_os_type = 'ubuntu'
    m_engine_type = 'pytorch'
    m_libs = []
    m_apt = []
    m_papi = []
    m_image_name ='python:3.9'
    m_dep_type = 'k8s'
    m_dep_work_dir = ""
    m_dep_entrypoint = []
    m_dep_hostip = ""
    m_dep_hostport = ""

    m_nn_file = "test.py"
    m_weight_file = 'test.pt'
    m_annotation_file = "coco.dat"
    m_model_file= ""
    m_execution_tool=""
    ####################################################################
    def __init__(self):
        """
        Initialize function

        Args: None
        Returns: None
        """
        self.m_last_run_state = 0
        return

    ####################################################################
    def set_folder(self, uid, pid):
        """
        Set user id, project id and file path
        Args:
            uid : user id(string)
            pid : project id (string)
        Returns: int
            0 : success
            -1 : file error
        """
        self.m_current_userid = uid
        self.m_current_projectid = pid
        self.m_current_file_path = "%s%s%s%s%s%s" % (def_top_folder, "/", uid, "/", pid, "/")
        self.m_current_code_folder = "%s%s" % (self.m_current_file_path, def_code_folder_name)
        return 0

    ####################################################################
    def get_real_filepath(self, filename):
        """
        Get absolute file path

        Args:
            filename : filename
        Returns:
            full file path (string)
        """
        ret = "%s%s" % (self.m_current_file_path, filename)
        return ret

    ####################################################################
    def get_code_filepath(self, filename):
        """
        Get absolute file path for code generation

        Args:
            filename : filename
        Returns:
            full file path (string)
        """
        # if no m_current_code_folder, then make the folder
        try:
            if not os.path.exists(self.m_current_code_folder):
                os.makedirs(self.m_current_code_folder)
        except OSError:
            print('Error: Creating directory ' + self.m_current_code_folder)

        ret = "%s%s%s" % (self.m_current_code_folder, "/", filename)
        return ret

    ####################################################################
    def run(self):
        """
        Read Deploy information, zip files

        Args: None
        Returns: None
        """
        # KPST modify to get information for k8s deploy
        ret = self.parse_deployinfo_file()
        
        if ret == 0:
            # #############################
            # KPST add code here for k8s_deploy
            # neural net applications are located in nn_model folder
            # use the following variables
            # self.m_dep_entrypoint = the file that should be called
            # self.m_dep_hostip = target system ip
            # self.m_dep_hostport = target system port number
            # self.m_nn_file = pytorch python file
            # self.m_weight_file = weight file
            # self.m_annotation_file = annotation info file
            # to be added
            #print(self.m_current_projectid)
            
            
            run_kubernetes=deployment.KubeJob(job_name=self.m_current_userid, input_data=self.m_dep_work_dir, output_data=self.m_current_code_folder, nn_file=self.m_nn_file,
                                            weight_file=self.m_weight_file, annotation_file=self.m_annotation_file, prj_path=self.m_current_code_folder, model_file=self.m_model_file,
                                            nfs_ip=self.m_nfs_ip, nfs_path=self.m_nfs_path, image_name=self.m_image_name, svc_port=self.m_dep_hostport)

            state=run_kubernetes.run_deploy()
            print(state)
            self.m_last_run_state = 0
        else:
            ret = -1
            self.m_last_run_state = -1
        return ret 
    def del_run(self):
        kube_del=deployment.Delete_kube(name=self.m_current_userid)
        kube_del.delete_job_pv_pvc()

    ####################################################################
    def parse_deployinfo_file(self):
        """
        Read Information file generated from CodeGen

        Args: None
        Returns: int
            0 : success
            -1 : file error
        """
        # KPST modify to get information for k8s deploy
        try:
            f = open(self.get_real_filepath(self.m_deployinfo_file), encoding='UTF8')
        except IOError as err:
            print("Deploy Info file Read Error", err)
            return -1

        dep_info = yaml.load(f, Loader=yaml.FullLoader)
        # parse deployment yaml file
        for key, value in sorted(dep_info.items()):
            if key == 'build':
                for subkey, subvalue in sorted(value.items()):
                    if subkey == 'architecture':
                        self.m_arch_type = subvalue
                    elif subkey == 'accelerator':
                        self.m_acc_type = subvalue
                    elif subkey == 'os':
                        self.m_os_type = subvalue
                    elif subkey == 'target_name':
                        self.m_image_name= subvalue
                    elif subkey == 'components':
                        for third_key, third_value in sorted(subvalue.items()):
                            if third_key == 'engine':
                                self.m_engine_type = third_value
                            elif third_key == 'libs':
                                self.m_libs = third_value
                            elif third_key == 'custom_packages':
                                for forth_key, forth_value in sorted(third_value.items()):
                                    if forth_key == 'apt':
                                        self.m_apt = forth_value
                                    if forth_key == 'papi':
                                        self.m_papi = forth_value

            elif key == 'deploy':
                for subkey, subvalue in sorted(value.items()):
                    # if subkey == 'type':
                    #     self.m_dep_type = subvalue
                    # elif subkey == 'work_dir':
                    #     self.m_dep_work_dir = subvalue
                    if subkey == 'entrypoint':
                        self.m_dep_entrypoint = subvalue
                    elif subkey =='k8s':
                        print(subvalue.items())
                        for thirdkey, thirdvalue in sorted(subvalue.items()):
                            if thirdkey == 'nfsip':        #K8S
                                self.m_nfs_ip = str(thirdvalue)
                                print(self.m_nfs_ip)
                            elif thirdkey == 'nfspath':      #K8S
                                self.m_nfs_path = str(thirdvalue)
                    elif subkey == 'network':
                        for thirdkey, thirdvalue in sorted(subvalue.items()):
                            if thirdkey == 'service_host_ip':
                                self.m_dep_hostip = thirdvalue
                            elif thirdkey == 'service_host_port':
                                self.m_dep_hostport = int(thirdvalue)

            elif key == 'optional':
                for subkey, subvalue in sorted(value.items()):
                    if subkey == 'nn_file':
                        self.m_nn_file = subvalue
                    elif subkey == 'weight_file':
                        self.m_weight_file = subvalue
                    elif subkey == 'annotation_file':
                        self.m_annotation_file = subvalue
                    elif subkey == 'model_file':
                        self.m_model_file = subvalue
        f.close()
        return 0

    ####################################################################
    def response(self):
        """
        Send Success Message to Project Manager 

        Args: None
        Returns: None 
        """
        host = ''
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            host = s.getsockname()[0]
        except socket.error as err:
            print(err)
        prj_url = "%s%s%s" % ('http://', host, ':8085/status_report')
        print(prj_url)
        prj_data = 'container_id=k8s_deploy'
        prj_data = "%s%s%s%s%s" % (prj_data, '&user_id=', self.m_current_userid,
                                   '&project_id=', self.m_current_projectid)
        # add result code
        if self.m_last_run_state == 0:  # success
            prj_data = "%s%s" % (prj_data, '&result=success')
        else:
            prj_data = "%s%s" % (prj_data, '&result=failed')

        headers = {
            'Host': '0.0.0.0:8085',
            'Origin': 'http://0.0.0.0:8901',
            'Accept': "application/json, text/plain",
            'Access-Control_Allow_Origin': '*',
            'Access-Control-Allow-Credentials': "true"
            }

        try:
            requests.get(url=prj_url, headers=headers, params=prj_data)
        except requests.exceptions.HTTPError as err:
            print("HTTPError:", err)
        except requests.exceptions.ConnectionError as err:
            print("ConnectionError:", err)
        except requests.exceptions.Timeout as err:
            print("Timeout:", err)
        except requests.exceptions.RequestException as err:
            print("RequestException:", err)
        return


####################################################################
# class for HTTP server
####################################################################
class MyHandler(SimpleHTTPRequestHandler):
    """Web Server definition """
    m_deploy_obj = K8SDeploy()
    m_flag = 1
    m_stop = 0
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
        """
        Process HTTP GET command

        Args: None
        Returns: None
        """
        print(self.path)
        if self.path[1] == '?':
            t_path = "%s%s" % ('/', self.path[2:])
        else:
            t_path = self.path
        pathlist = t_path.split('/')[1].split('?')
        cnt = len(pathlist)
        if cnt < 2:
            print(t_path)
            cmd = "unknown"
        else:
            ctmp = pathlist[1].split('&')
            mycnt = len(ctmp)
            if mycnt == 0:
                cmd = pathlist[0]
            elif mycnt == 1:
                cmd = pathlist[0]
                userid = ctmp[0].split('user_id')[1].split('=')[1]
                if userid == '""' or userid == '%22%22':
                    self.m_deploy_obj.set_folder("", "")
                else:
                    self.m_deploy_obj.set_folder(userid, "")
            else:  # mycnt == 2:
                cmd = pathlist[0]
                userid = ctmp[0].split('user_id')[1].split('=')[1]
                prjid = ctmp[1].split('project_id')[1].split('=')[1]
                if userid == '""' or userid == '%22%22':
                    userid = ""
                if prjid == '""' or prjid == '%22%22':
                    prjid = ""
                self.m_deploy_obj.set_folder(userid, prjid)
        print("cmd =", cmd)

        if cmd == "start":
            buf = 'starting'
            self.send_response(200, 'ok')
            self.send_cors_headers()
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(buf.encode())
            if self.m_flag == 1:
                self.m_deploy_obj.run()
            # send notice to project manager
            self.m_deploy_obj.response()
        elif cmd == 'stop':
            buf = "finished"
            self.send_response(200, 'ok')
            self.send_cors_headers()
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(buf.encode())
            self.m_stop = 1
            
        elif cmd == "pause":
            self.m_flag = 0
            buf = "OK"
            self.send_response(200, 'ok')
            self.send_cors_headers()
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(buf.encode())
        elif cmd == 'resume':
            self.m_flag = 1
            buf = "OK"
            self.send_response_only(200, 'OK')
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(buf.encode())

        elif cmd == 'status_request':
            buf = "error"
            if self.m_deploy_obj.m_current_userid == "":
                buf = "ready"
            else:
                if self.m_flag == 0:
                    buf = "stopped"
                elif self.m_flag == 1:
                    buf = "completed"
            self.send_response(200, 'ok')
            self.send_cors_headers()
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(buf.encode())
        else:
            buf = ""
            self.send_response(200, 'ok')
            self.send_cors_headers()
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(buf.encode())

        if self.m_stop == 1:
            
            self.m_deploy_obj.del_run()
            #time.sleep(1)
            
            #raise KeyboardInterrupt
        return

    def do_OPTIONS(self):
        """
        send header for CORS
        
        Args: None
        Returns: None
        """
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()    

    '''
    def do_POST(self):
        self.send_response(200)
        self.send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        dataLength = int(self.headers["Content-Length"])
        data = self.rfile.read(dataLength)
        print(data)
        response = {"status": "OK"}
        self.send_dict_response(response)
    '''


####################################################################
####################################################################
if __name__ == '__main__':
    server = HTTPServer(('', def_deploy_port), MyHandler)
    print("Started K8S Deployment Server....")
    print("Press ^C to quit WebServer")

    try:
        server.serve_forever()
    except KeyboardInterrupt as e:
        time.sleep(1)
        server.socket.close()
        print("K8S Deploy Module End", e)
