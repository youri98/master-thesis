import torch
import multiprocessing
import platform,socket,re,uuid,json,psutil,logging

def getSystemInfo():
    info={}
    info['platform']=platform.system()
    # info['platform-release']=platform.release()
    # info['platform-version']=platform.version()
    # info['architecture']=platform.machine()
    # info['hostname']=socket.gethostname()
    # info['ip-address']=socket.gethostbyname(socket.gethostname())
    # info['mac-address']=':'.join(re.findall('..', '%012x' % uuid.getnode()))
    info['processor']=platform.processor()
    info['ram']=str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB"
    info['n cores'] = multiprocessing.cpu_count()
    info['gpu'] = torch.cuda.is_available()

    return json.dumps(info)


print(json.loads(getSystemInfo()))