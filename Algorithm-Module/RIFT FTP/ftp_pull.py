from ftplib import FTP
import os
import time

def get_newest_so_file(ftp):
    Trained_Model = os.path.join(os.path.dirname(__file__), "SCCS FTP Server/Trained Model")
    ftp.cwd('/Trained Model')
    # List files on the server
    files = ftp.nlst()
    
    # Filter out only .so files
    so_files = [f for f in files if f.endswith('.so')]

    # Get the newest .so file based on modification time
    newest_file = None
    newest_time = -1
    
    for file in so_files:
        # Get modification time of the file
        modified_time = ftp.sendcmd(f"MDTM {file}")
        
        # Parse the modification time to an integer (timestamp)
        # Example MDTM response: '213 20240801123045' (YYYYMMDDHHMMSS)
        modified_time = int(modified_time.split()[1])
        
        if modified_time > newest_time:
            newest_time = modified_time
            newest_file = file

    return newest_file


def download_file(ftp, filename, download_dir):
    # Ensure the directory exists
    os.makedirs(download_dir, exist_ok=True)
    
    # Define the local file path
    local_filename = os.path.join(download_dir, os.path.basename(filename))
    
    with open(local_filename, "wb") as file:
        # Download the file
        ftp.retrbinary(f"RETR {filename}", file.write)

    print(f"Downloaded {local_filename}")


def monitor_and_download(download_dir):
    # Set to store downloaded files to prevent re-downloading
    downloaded_files = set()

    while True:
        try:
            start_time = time.time()
            # Connect to the FTP server
            ftp = FTP()
            ftp.connect("127.0.0.1", 2121)
            ftp.login("user", "12345")

            # Get the newest .so file
            newest_file = get_newest_so_file(ftp)
            
            # Check if the newest file has already been downloaded
            if newest_file and newest_file not in downloaded_files:
                download_file(ftp, newest_file, download_dir)
                downloaded_files.add(newest_file)
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Get size of file
            ftp.voidcmd('TYPE I')
            file_size = ftp.size(newest_file)

            # Convert size to megabytes
            file_size_mb = file_size / (1024 * 1024)
            bandwidth = file_size_mb / elapsed_time  # MBps

            print(f"File size: {file_size_mb:.2f} MB")
            print(f"Time taken: {elapsed_time:.2f} seconds")
            print(f"Bandwidth: {bandwidth:.2f} MBps")

            # Close the FTP connection
            ftp.quit()
            print("FTP connection closed")
            
        except Exception as e:
            print(f"Error: {e}")

        # Wait for a while before checking again (e.g., 10 seconds)
        time.sleep(10)

if __name__ == "__main__":
    
    download_directory = os.path.join(os.path.dirname(__file__), "Algo Module/CV model")        # Algo module local dir

    if not os.path.exists(download_directory):
        os.makedirs(download_directory)
        print(f"Created directory: {download_directory}")
    
    monitor_and_download(download_directory)