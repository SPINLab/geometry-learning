import http
import os
from datetime import datetime

import boto3
import requests

from slackclient import SlackClient

SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
# Set this to the appropriate region
REGION_NAME = 'eu-west-1'

# Get environment variables
# Slack is required. We need to know if something is wrong
slack_token = os.environ['SLACK_API_TOKEN']
slack_channel = os.environ['SLACK_CHANNEL']
# We are also going to require Amazon credentials, set as environment variables
amazon_id = os.environ['AWS_ACCESS_KEY_ID']
amazon_key = os.environ['AWS_SECRET_ACCESS_KEY']

# Initialize frameworks
ec2 = boto3.client('ec2', region_name=REGION_NAME)
sc = SlackClient(slack_token)


# Slack notification function
def notify(signature, message):
    sc.api_call("chat.postMessage", channel=slack_channel,
                text="Script " + signature + " notification: " + str(message))


# Get build queue length
queue = "http://teamcity:8111/guestAuth/app/rest/buildQueue"
headers = {
    'Accept': "application/json",
    'Cache-Control': "no-cache",
}
queue_res = requests.get(queue, headers=headers)
queue_status = queue_res.json()
queue_length = queue_status['count']

# Get instance id for this machine
# https://stackoverflow.com/questions/33301880/how-to-obtain-current-instance-id-from-boto3#33307704
try:
    instance_metadata = requests.get('http://169.254.169.254/latest/meta-data/instance-id')
except ConnectionError as e:
    notify(SCRIPT_NAME, 'ERROR getting instance id, cannot issue commands')
    raise ConnectionError(e)

instance_id = instance_metadata.text

if queue_length == 0:
    print('build server reports empty queue, shutting down.')
    shutdown_res = ec2.stop_instances(InstanceIds=[instance_id])
    http_status_code = shutdown_res['ResponseMetadata']['HTTPStatusCode']
    http_status = http.HTTPStatus(http_status_code).name

    if http_status_code == 200:
        print('Stop instances:', http_status)
        notify(SCRIPT_NAME, 'successful shutdown of {} with response {}'.format(instance_id, http_status))
    else:
        notify(SCRIPT_NAME, 'ERROR shutting down instance id: {}'.format(http_status))
else:
    notify(SCRIPT_NAME, 'job finished, build server reports non-empty queue, continuing.')


