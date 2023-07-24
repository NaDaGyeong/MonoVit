import numpy as np
import PIL.Image as pil
import os

from .mono_dataset import MonoDataset

# FOV
FOV = 50

# image size
img_width = 512
img_height = 352

image_size = [img_width, img_height]

class INFRADataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(INFRADataset, self).__init__(*args, **kwargs)

        focal = img_width / (2 * np.tan(FOV * np.pi / 360))

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array([[focal / img_width, 0, 0.5, 0],
                          [0, focal / img_height, 0.5, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (img_width, img_height)

    def check_depth(self):
        return False

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        """
        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        return folder, frame_index, side

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}.jpg".format(frame_index-1 )
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        
        return color