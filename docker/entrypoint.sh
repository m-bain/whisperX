#!/bin/bash
# Set READ_IMPLIES_EXEC personality to allow executable stacks
# This is required for CTranslate2 on Docker Desktop
setarch $(uname -m) -R /bin/bash -c "exec uvicorn api:app --host 0.0.0.0 --port 8000"
