import os
from pathlib import Path


########################
#### Configurations ####
########################
class Processors:
    def __init__(self, albumentations, visualization, preprocess, postprocess):
        self.use_albumentations_library = albumentations # True: use artifact, False: use custom
        if self.use_albumentations_library:
            self.preprocess_library_torch = False # artifact uses albumentations, this configuration ignored 
        else:
            self.preprocess_library_torch = preprocess # True: pytorch, False: cv2
        if not self.preprocess_library_torch:
            self.visualization_library_cv2 = True
        else:
            self.visualization_library_cv2 = visualization # True: cv2, False: pil
        self.postprocess_library_torch = postprocess # True: pytorch, False: cv2

    def display_config(self):
        print(f"Use albumentations library: {self.use_albumentations_library}")
        print(f"Use cv2 as visualization library: {self.visualization_library_cv2}")
        print(f"Use torch as preprocess library: {self.preprocess_library_torch}")
        print(f"Use torch as postprocess library: {self.postprocess_library_torch}")

def set_processor_configs(albumentations_path):
    albumentations_path = Path(albumentations_path) / "processors" / "af_preprocessor.json"
    if os.path.exists(albumentations_path):
        config = Processors(True, True, False, True) # use artifact, cv2, none, torch
    else:
        config = Processors(False, False, True, True) # no artifact, pil, torch, torch
    return config


########################
##### Data loaders #####
########################
def load_labels(path):
    with open(path, "r") as f:
        return f.read().strip().split("\n")

def load_image(path, config):
    if config.visualization_library_cv2:
        import cv2
        #  image = cv2.imread(path, )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # cv reads in BGR
        # cv2.imwrite('/workspace/example-applications/sample_images/cv2_im.jpg',image)
        return image
    else:
        from PIL import Image
        image = Image.open(path)
        return image


########################
### Metadata readers ###
########################
def get_layout_dims(layout_list, shape_list):
    if len(layout_list) != len(shape_list):
        raise ValueError("Both input lists should have the same number of elements.")
    
    result = []
    
    for i in range(len(layout_list)):
        layout_str = layout_list[i]
        shape_tuple = shape_list[i]
        
        if len(layout_str) != len(shape_tuple):
            raise ValueError(f"Length of layout string does not match the number of elements in the shape tuple for input {i}.")
        
        layout_dict = {letter: number for letter, number in zip(layout_str, shape_tuple)}
        result.append(layout_dict)
    
    return result


########################
##### Box plotters #####
########################
def plot_one_box(box, img, color, label=None, line_thickness=None):
    import cv2
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

    # list of COLORS
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

    return img


def plot_boxes(image, output, labels):
    import numpy as np 
    output_image = np.array(image)
    for bb in output:
        for i in range(0,len(bb)):
            box = bb[i][0:4]
            label = labels[int(bb[i][5])]
            output_image = plot_one_box(
                box,
                output_image,
                color=(0, 0, 255),
                label=label,
            )
    
    return output_image


def save_image(output_image, image_path, config):
    import datetime
    p = os.path.splitext(image_path)
    output_filename = f"{p[0]}-{datetime.datetime.now()}{p[1]}"

    if config.visualization_library_cv2:
        import cv2
        cv2.imwrite(output_filename, output_image)
    else:
        import torchvision.transforms as transforms
        pil_to_transform = transforms.ToPILImage()
        output_image = pil_to_transform(output_image)
        output_image.save(output_filename)
    
    return output_filename

########################
#####    Timer     #####
########################

import time
import math

class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed_times = []
        self.ms_factor = 0.000001

    def start(self):
        self.start_time = time.time()

    def stop(self):
        end_time = time.time()
        elapsed_ms = (end_time - self.start_time) * 1000  # Convert to milliseconds
        self.elapsed_times.append(elapsed_ms)

    def averageElapsedMilliseconds(self):
        if not self.elapsed_times:
            return 0.0
        elif len(self.elapsed_times) < 2:
            return self.elapsed_times[0]
        else:
            return sum(self.elapsed_times[1:]) / (len(self.elapsed_times) - 1)

    def standardDeviationMilliseconds(self):
        if len(self.elapsed_times) < 2:
            return 0.0

        mean = self.averageElapsedMilliseconds()
        accum = sum((d - mean) ** 2 for d in self.elapsed_times[1:])
        return math.sqrt(accum / (len(self.elapsed_times) - 2))

# Function to round a float to a specified number of decimal places
def roundToDecimalPlaces(value, decimal_places):
    factor = 10.0 ** decimal_places
    return round(value * factor) / factor

