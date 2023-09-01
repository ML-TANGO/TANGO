#!/command/execlineb -P
/venvs/cloud_manger/bin/python3 -m http.server 8088  # for testing
# cd /source
# /venvs/cloud_manager/bin/python3 -m uvicorn cloud_manager.server:app --port 8088