from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer
import os

def run_ftp_server():
    # Instantiate a dummy authorizer for managing 'virtual' users
    authorizer = DummyAuthorizer()
    
    # Define the FTP Server folder home directory paths
    USER_DIR = os.path.join(os.path.dirname(__file__), "SCCS FTP Server")
    # USER_DIR = r"\\wsl.localhost\Ubuntu-22.04\home\ck\MLOps\RIFT FTP Server"
    
    # Ensure the directories exist
    if not os.path.isdir(USER_DIR):
        os.makedirs(USER_DIR)
    
    # Define a new user having full r/w permissions and anonymous user
    authorizer.add_user("user", "12345", USER_DIR, perm="elradfmw")
    authorizer.add_anonymous(USER_DIR)

    # Instantiate an FTP handler class
    handler = FTPHandler
    handler.authorizer = authorizer

    # Define a customized banner (string returned when client connects)
    handler.banner = "pyftpdlib based ftpd ready."

    # Specify a masquerade address and the range of ports to use for
    # passive connections. Useful when behind a NAT.
    # handler.masquerade_address = "172.20.10.3"
    # handler.passive_ports = range(60000, 65535)

    # Instantiate an FTP server class and listen on 0.0.0.0:2121
    address = ("0.0.0.0", 2121)
    server = FTPServer(address, handler)

    # Set a limit for connections
    server.max_cons = 256
    server.max_cons_per_ip = 5

    # Start ftp server
    try:
        print("Starting FTP server...")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down the server.")
        server.close_all()

if __name__ == "__main__":
    run_ftp_server()
