#!/bin/bash
cd /home/ubuntu/aa/layout_detection && \
git pull origin && \
source env/bin/activate && \
pip install -r requirements.txt && \
sudo supervisorctl -c /home/ubuntu/aa/layout_detection/config/aa_supervisor.conf signal SIGHUP aa_api