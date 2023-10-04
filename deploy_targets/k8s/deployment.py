# -*- coding: utf-8 -*-

import datetime
import configparser
import time
from datetime import datetime
from kubernetes import client, config, watch
from kubernetes.stream import stream
from kubernetes.client.rest import ApiException
from kubernetes.client.api import core_v1_api
import logging
import logging.handlers

config.load_kube_config()
v1 = client.CoreV1Api() 
apps_v1 = client.AppsV1Api() 
batch_v1 = client.BatchV1Api() 

#IMAGE="busan_deid:source"
NAMESPACE="default"
# NFS_IP = '192.168.0.189'    # for test

try:
    real_time=datetime.today().strftime('%Y-%m-%d')
    logger = logging.getLogger('log')
    logger.setLevel(logging.INFO)
    logfile_H=logging.handlers.TimedRotatingFileHandler(filename='./log/'+real_time+"-log.txt" ,when= 'midnight', interval =1, encoding='utf-8',backupCount=2)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile_H.setFormatter(formatter)
    logfile_H.suffix = "%Y-%m-%d"
    logger.addHandler(logfile_H)
except Exception as err:
    pass
   
   #job_name=self.m_acc_type, input_data=self.m_current_file_path, output_data=self.m_current_code_folder, weight_file=self.m_weight_file, anotation_file=self.m_annotation_file
class KubeJob:
    def __init__(self, job_name, input_data, output_data, weight_file, annotation_file, prj_path, 
                model_file, nfs_path, nfs_ip, nn_file, image_name, svc_port, service_host_ip):
        self.job_name = job_name
        self.input_data = input_data
        self.output_data = output_data
        self.annotation_file = annotation_file
        self.weight_file = weight_file
        self.prj_path = prj_path
        self.model_file = model_file
        self.nfs_path = nfs_path
        self.nn_file = nn_file
        self.nfs_ip = nfs_ip
        self.image_name = image_name
        self.svc_port = svc_port
        self.service_host_ip=service_host_ip
    

    def create_job_object(self):
        #output_data=self.output_data.split('./')
       # self.output_data=output_data[1]
        #if 'train' in str(self.nn_file):
        prj_path_split=str(self.prj_path).split('/tango')
            
        container = client.V1Container(
            name= self.job_name,
            image= self.image_name,   # for test
            # image_pull_policy= 'IfNotPresent',
            image_pull_policy= 'Always',
            command=["bash", "-c"],
            volume_mounts=[client.V1VolumeMount(name=self.job_name, mount_path='/mnt')],
            env=[
            client.V1EnvVar(name="NAME", value=str(self.job_name)),
            client.V1EnvVar(name="INPUT_DATA", value=str(self.input_data)),
            client.V1EnvVar(name="OUTPUT_DATA", value=str(self.output_data)),
            client.V1EnvVar(name="WEIGHT", value=str(self.weight_file)),
            client.V1EnvVar(name="CLASSES", value=str(self.output_data)),
            client.V1EnvVar(name="PRJ_PATH", value=str(prj_path_split[1])),
            client.V1EnvVar(name="ANN", value=str(self.annotation_file)),
            client.V1EnvVar(name="MODEL", value=str(self.model_file)),
            client.V1EnvVar(name="NN_FILE", value=str(self.nn_file))
            ],
            #if 'train' in str(self.nn_file):
            #    args=["cd /app/ && cd $(PRJ_PATH) && python3 $(NN_FILE) --cfg   --weights $(MODEL) --data models/$(ANN) --name $(NAME)"]
            
            #args=["cd /app/ && cd $(PRJ_PATH) && python3 $(NN_FILE)  --weights $(MODEL) --data data/$(ANN) "]) # --source $(INPUT_DATA) -w $(WEIGHT) -c $(ANN) --device cpu
            # args=["cd $(PRJ_PATH) && python3 output.py"])
            # args=["sleep 10000;"])
            args=["cd /mnt$(PRJ_PATH)/fileset-main/yolov5s/ && python3 output.py"])
        volume = client.V1Volume(name=self.job_name, persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name=self.job_name))

        template = client.V1PodTemplateSpec(spec=client.V1PodSpec(node_selector={"kubernetes.io/hostip" : self.service_host_ip}, restart_policy="Never", containers=[container], volumes=[volume]))  #,node_selector={"kubernetes.io/hostname" : "etri-3"}

        spec = client.V1JobSpec(
            template=template,
            backoff_limit=0,
            selector={"app": self.job_name}
            )
        
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(name=self.job_name, labels={"app": self.job_name}),
            spec=spec)
        return job
    
    def create_service(self):
        body = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name='svc-'+self.job_name,
                labels={"app": self.job_name}
            ),
            spec=client.V1ServiceSpec(
                selector={"job-name": self.job_name},
                type="NodePort",
                ports=[client.V1ServicePort(
                    port=8901,
                    target_port=8901,
                    node_port=int(self.svc_port)) 
                    ]
            )
        )
        v1.create_namespaced_service(namespace="default",body=body)

    def create_job(self):
        api_response=None
        api=None
        read_job=None
        try:
            read_job=batch_v1.read_namespaced_job(name=self.job_name,namespace=NAMESPACE)
        except:
            pass
        print(read_job)
        if read_job is None:
            job=self.create_job_object()
            api_response = batch_v1.create_namespaced_job(
                body=job,
                namespace=NAMESPACE)
            print("Job created. status='%s'" % str(api_response.status))
        

    def create_pv(self):
        resp=None
        try:
            resp = v1.read_persistent_volume(name=self.job_name)
        except ApiException as e:
            if e.status != 404:
                print("Unknown error: %s" % e)
                pass

        if resp is None:
            #print("PersistentVolumeClaim %s does not exist. Creating it..." % name)       ############### NFS
            body={'apiVersion': 'v1', 
            'kind': 'PersistentVolume', 
            'metadata': {'name': self.job_name, 'labels': {'name': self.job_name}}, 
            'spec': {'capacity': {'storage': '50Gi'}, 'StorageClassName': "",
            'accessModes': ['ReadWriteMany'],
            'nfs': {'server': self.nfs_ip, 'path': self.nfs_path}}}
            v1.create_persistent_volume(body=body)

    def run_deploy(self):
        self.create_pv()
        self.create_pvc()
        time.sleep(1)
        try:
            pvc_get=v1.read_namespaced_persistent_volume_claim(name=self.job_name,namespace=NAMESPACE)
            p_status=pvc_get.status.phase
        except:
            pass
        if p_status:
            for loopCount in range(0, 10, 1):
                pvc_get=v1.read_namespaced_persistent_volume_claim(name=self.job_name,namespace=NAMESPACE)
                p_status=pvc_get.status.phase
                if p_status=='Bound':
                    self.create_job()
                    self.create_service()
                    return True
                else:
                    time.sleep(1)
                    
            #return False
        
    def create_pvc(self):
        resp=None
        try:
            resp = v1.read_namespaced_persistent_volume_claim(name=self.job_name,namespace=NAMESPACE)
        except ApiException as e:
            if e.status != 404:
                print("Unknown error: %s" % e)
                exit(1)
    
        if resp is None:
            #print("PersistentVolumeClaim %s does not exist. Creating it..." % name)
            body={'kind': 'PersistentVolumeClaim', 
            'apiVersion': 'v1', 
            'metadata': {'name': self.job_name}, 
            'spec': {'accessModes': ['ReadWriteMany'], 'StorageClassName': "",
            'resources': {'requests': {'storage': '50Gi'}}, 
            'selector': {'matchLabels': {'name': self.job_name}}}}
            v1.create_namespaced_persistent_volume_claim(namespace=NAMESPACE, body=body)
        
class Pod_Amount:
    def __init__(self, node_name):
        self.node_name= node_name
        self.count=0
        self.namespace=NAMESPACE
        
    def pods_status_running_amount(self):   
        status = v1.list_namespaced_pod(namespace=self.namespace)
        for i in status.items:
            if i.spec.node_name==self.node_name:
                self.count+=1
        return self.count

    def pods_status_complete_amount(self):   
        status = v1.list_namespaced_pod(namespace=self.namespace)
        for i in status.items:
            if i.spec.node_name==self.node_name:
                if i.status.phase=='Succeeded':
                    self.count+=1
        return self.count

    def pods_status_pending_amount(self):   
        status = v1.list_namespaced_pod(namespace=self.namespace)
        for i in status.items:
            if i.spec.node_name==self.node_name:
                if i.status.phase!='Running' and i.status.phase!='Succeeded':
                    self.count+=1
        return self.count

class KubeGet:
    def __init__(self,name):
        self.name=name
    
    def job_pod_name(self):
        par='-'
        pods=v1.list_namespaced_pod(namespace=NAMESPACE)
        for re in pods.items:
            ho = re.metadata.generate_name
            if ho==self.name+par:                       
                pod_name = re.metadata.name
                return pod_name
    def pod_ip(self):
        pod_name=self.job_pod_name()
        pod_json=v1.read_namespaced_pod(name=self.name, namespace=NAMESPACE)
        pod_ip=pod_json.status.pod_ip

        return pod_ip
            
class Delete_kube:
    def __init__(self,name):
        self.name=name
    def job_pod_name(self):
        p='-'
        pods=v1.list_namespaced_pod(namespace=NAMESPACE)
        for re in pods.items:
            ho = re.metadata.generate_name
            if ho==self.name+p:                       
                pod_name = re.metadata.name
                #pods_create_update(name=pod_name,namespace=namespace) 
                return pod_name
    def delete_pv_pvc(self):
        resp = None
        pv_status = None
        try:
            v1.delete_namespaced_persistent_volume_claim(namespace=NAMESPACE, name=self.name, grace_period_seconds=0) ##pvc삭제
            v1.delete_persistent_volume(name=self.name, grace_period_seconds=0)
        except:
            pass
        
        try:
            resp = v1.read_namespaced_persistent_volume_claim(namespace=NAMESPACE,name=self.name)
        except ApiException as e:
            #print("Unknown error: %s" % e)
            pass
        if resp is not None:
            pv=resp.spec.volume_name            ##pvc랑 바운드 된 pv이름 받아옴
            #road = v1.read_namespaced_persistent_volume_claim(namespace=namespace, name=name)
            v1.delete_namespaced_persistent_volume_claim(namespace=NAMESPACE, name=self.name, grace_period_seconds=0) ##pvc삭제
        try:
            pv_status= v1.read_persistent_volume(name=self.name)
        except ApiException as e:
            #print(e)
            pass
        if pv_status is not None:                               ## pv아직 존재시 pv도 삭제
            v1.delete_persistent_volume(name=self.name, grace_period_seconds=0)

        if not resp:
            #print("%s\t%s not exist"%(namespace,name))
            pass
    def delete_job_pod(self):            #job에서 pod가 하나일때를 가정함
        resp = None
        pod_name=None

        try:
            resp = batch_v1.read_namespaced_job(namespace=NAMESPACE,name=self.name)
            #print(resp)
        except ApiException as e:
            if e.status != 404:
                #print("Unknown error: %s" % e)
                exit(1)

        if resp is not None:
            try:
                pod_name=self.job_pod_name()
                print(pod_name)
                batch_v1.delete_namespaced_job(name=self.name,namespace=NAMESPACE, grace_period_seconds=0)
                res = v1.delete_namespaced_pod(name=pod_name, namespace=NAMESPACE,grace_period_seconds=0)

                #print('Pod Delete res : ',res)
            except Exception as e:
                pass

        elif resp is None:
            try:
                res = v1.delete_namespaced_pod(name=pod_name, namespace=NAMESPACE, grace_period_seconds=0)
                #print('Pod Delete res : ',res)
            except:
                #v1.delete_namespaced_pod(name=pod_name, namespace=NAMESPACE,grace_period_seconds=0)
                pass
            #print("%s\t%s not exist"%(namespace,name))

    def delete_svc(self):
        resp=None
        print(self.name)
        try:
            resp =v1.read_namespaced_service(name='svc-'+self.name, namespace=NAMESPACE)
            
        except Exception as e:
            pass
        if resp is not None:
            resp = v1.delete_namespaced_service(name='svc-'+self.name,namespace=NAMESPACE, grace_period_seconds=0)

    def delete_job_pv_pvc(self):              
        try:
            self.delete_job_pod()
    
        except ApiException as e:
            #print(e)
            pass
        try:
            self.delete_svc()
        except:
            pass
        try:
            self.delete_pv_pvc()
        except ApiException as e:
            #print(e)
            pass


