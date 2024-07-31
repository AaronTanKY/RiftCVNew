import sys
import subprocess
import klvdata

def main():
    video_file = 'Truck.ts'
    
    ffmpeg_command = [
        'ffmpeg',
        '-i', video_file,
        '-map', '0:2',
        '-codec', 'copy',
        '-f', 'data',
        '-'
    ]
    
    process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    count = 0
    for packet in klvdata.StreamParser(process.stdout.read()):
        metadata = packet.MetadataList()
        for key, value in metadata.items():
            print(key, value)
        
        # packet.structure()
        # count = count + 1
        # print(count)
        


if __name__ == "__main__":
    main()