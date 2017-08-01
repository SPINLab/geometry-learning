#!/usr/bin/env bash
sudo apt-get update

# For Ubuntu version 14.04 (Trusty)
eval $(cat /etc/*release | grep VERSION_ID)

if [ VERSION_ID = "14.04" ]; then
  sudo apt-get -y install linux-image-extra-$(uname -r) linux-image-extra-virtual
fi

sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get -y install docker-ce
sudo pip install docker-compose
sudo usermod -aG docker ${USER}
echo "Please log out and in again to make use of the added user permissions on executing Docker commands"