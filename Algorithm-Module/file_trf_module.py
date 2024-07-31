## USING SFTP

import paramiko

def transfer_file_sftp(hostname, port, username, password, remote_file_path, local_file_path):
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, port, username, password)
    
    sftp = ssh.open_sftp()
    sftp.get(remote_file_path, local_file_path)
    sftp.close()
    ssh.close()

# Replace these variables with your details
hostname = 'remote_host'
port = 22
username = 'username'
password = 'password'
remote_file_path = '/path/to/remote/file'
local_file_path = '/path/to/local/destination'

transfer_file_sftp(hostname, port, username, password, remote_file_path, local_file_path)


## USING SCP

import paramiko
from scp import SCPClient

def create_ssh_client(hostname, port, username, password):
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, port, username, password)
    return ssh

def transfer_file_scp(hostname, port, username, password, remote_file_path, local_file_path):
    ssh = create_ssh_client(hostname, port, username, password)
    scp = SCPClient(ssh.get_transport())
    scp.get(remote_file_path, local_file_path)
    scp.close()

# Replace these variables with your details
hostname = 'remote_host'
port = 22
username = 'username'
password = 'password'
remote_file_path = '/path/to/remote/file'
local_file_path = '/path/to/local/destination'

transfer_file_scp(hostname, port, username, password, remote_file_path, local_file_path)
