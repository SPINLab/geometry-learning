import os

import sys
from slackclient import SlackClient

slack_token = os.environ.get("SLACK_API_TOKEN")


if slack_token:
    sc = SlackClient(slack_token)
    sc.api_call(
      "chat.postMessage",
      channel="#machinelearning",
      text="Notification: " + ', '.join(sys.argv[1:]))
else:
    print('No slack notification: no slack API token environment variable "SLACK_API_TOKEN" set.')
