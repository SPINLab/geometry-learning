#!/usr/bin/env bash
set -x
"${SLACK_API_TOKEN:?You need to provide a SLACK_TOKEN_API environment parameter}"
for script in '../model/*.py'
do
    python3 $1
    EC=$?
    if [ ${EC} -eq 0 ]
    then
        echo "${1} completed successfully."
    else
        echo "${1} failed, sending notification..."
        python3 ./slack_notify.py "Failed at executing ${1}"
    fi
done