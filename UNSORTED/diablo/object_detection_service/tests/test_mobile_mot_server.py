import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the current script
sys.path.append(os.path.join(dir_path, "../../../main_server/protobufs"))
sys.path.append(os.path.join(dir_path, "../"))

from mobile_od_server import main

sys.path.pop()
sys.path.pop()


def test_mobile_od_server():
    main()
