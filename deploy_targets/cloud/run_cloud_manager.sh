#!/command/with-contenv sh
cd /source
/venvs/cloud_manager/bin/python3 \
    -m uvicorn cloud_manager.server:app \
    --host 0.0.0.0 \
    --port 8890 \
    --reload
