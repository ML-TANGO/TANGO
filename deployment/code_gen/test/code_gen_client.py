import requests

method_name = 'GET' 

print("0= ready_request, 1=start, 2=stop, 3=state_request")
for i in range(12): 
    x = input()
    if x[0] == '0':
        url  = 'http://127.0.0.1:8888/status_request' 
        data = 'user_id=""&project_id=""'        
    elif x[0] == '1':
        url  = 'http://127.0.0.1:8888/start' 
        data = 'user_id=jammanbo&project_id=221018'        
    elif x[0] == '2':
        url  = 'http://127.0.0.1:8888/stop' 
        data = 'user_id=jammanbo&project_id=221018'        
    elif x[0] == '3':
        url  = 'http://127.0.0.1:8888/status_request' 
        data = 'user_id=jammanbo&project_id=221018'    
    else:
        continue

    response = requests.get(url=url, params=data)
    print(response)

    dict_meta = {'status_code':response.status_code, 'ok':response.ok, 
            'encoding':response.encoding, 
            'Content-Type': response.headers['Content-Type'],
            'text': response.text}

    if dict_meta['ok'] == True:
        print("RESPONSE SUCCESS")
        print(dict_meta['text'])
        # 성공 응답 시 액션
    else:
        print("RESPONSE ERROR")
        # 실패 응답 시 액션
