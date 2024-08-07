import ftplib
import os
import time

def upload_files(ftp, local_directory, server_directory, file_extension):
    ftp.cwd('/')
    if server_directory not in ftp.nlst():
        ftp.mkd(server_directory)
    ftp.cwd(server_directory)

    for filename in os.listdir(local_directory):
        if filename.endswith(file_extension):
            local_path = os.path.join(local_directory, filename)
            with open(local_path, 'rb') as file:
                ftp.storbinary(f'STOR {filename}', file)
                print(f"Uploaded: {filename} to {server_directory}")

def connect_and_upload(server, port, username, password, local_png_dir, local_xml_dir):
    try:
        ftp = ftplib.FTP()
        ftp.connect(server, port)
        ftp.login(user=username, passwd=password)
        print(f"Connected to FTP server: {server}")

        start_time = time.time()
        upload_files(ftp, local_png_dir, 'png_files', '.png')
        upload_files(ftp, local_xml_dir, 'xml_files', '.xml')
        end_time = time.time()
        elapsed_time = end_time - start_time

        # List all files in the directory
        xml_files = ftp.nlst(".")
        print(xml_files)
        png_files = ftp.nlst("../png_files")
        print(png_files)

        total_size = 0

        # Iterate through the files and sum the sizes of PNG files
        ftp.voidcmd('TYPE I')
        for file in png_files:
            if file.lower().endswith('.png'):
                try:
                    file_size = ftp.size(file)
                    total_size += file_size
                except ftplib.error_perm as e:
                    print(f"Could not get size for {file}: {e}")
        
        for file in xml_files:
            if file.lower().endswith('.xml'):
                try:
                    file_size = ftp.size(file)
                    total_size += file_size
                except ftplib.error_perm as e:
                    print(f"Could not get size for {file}: {e}")

        total_size_mb = total_size / (1024 * 1024)
        bandwidth = total_size_mb / elapsed_time  # MBps

        print(f"Total size of files: {total_size_mb:.2f} MB")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        print(f"Bandwidth: {bandwidth:.2f} MBps")

        ftp.quit()
        print("FTP connection closed")

    except ftplib.all_errors as e:
        print(f"FTP error: {e}")

if __name__ == "__main__":
    # FTP server details
    FTP_SERVER = "127.0.0.1"  # Change to your server address
    FTP_PORT = 2121  # Change to your server port
    FTP_USER = "user"  # Change to your username
    FTP_PW = "12345"  # Change to your password

    # Local directory to scan for files
    # os.path.join(os.path.dirname(__file__), "C3 System/images")
    PNG_DIR = os.path.join(os.path.dirname(__file__), "C3 System/images")
    XML_DIR = os.path.join(os.path.dirname(__file__), "C3 System/annotations")

    connect_and_upload(FTP_SERVER, FTP_PORT, FTP_USER, FTP_PW, PNG_DIR, XML_DIR)
