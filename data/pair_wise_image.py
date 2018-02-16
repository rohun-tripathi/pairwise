from torch.utils import data
from global_constants import DATA_PATH, SPLIT_PATH
import os, json
from PIL import Image
import numpy as np

label_map = {'1': 0, '2': 1, 'E': 2}

# Paper uses image width as 63x63
crop_width = 31


# mode belongs to {train, test}
class IIWDataset(data.Dataset):
    def __init__(self, mode, transforms=None, resize_transform=None):
        self.mode = mode
        self.data_items = self.make_data_set(mode)
        self.transforms = transforms
        self.resize_transform = resize_transform

    @staticmethod
    def make_data_set(mode='train'):
        data_items = []

        split_list = np.load(open(os.path.join(SPLIT_PATH, 'iiw_' + mode + '_ids.npy'), "rb"))

        images = set(filter(lambda x: ".png" in x, os.listdir(DATA_PATH)))
        judgements = filter(lambda x: ".json" in x, os.listdir(DATA_PATH))

        for judgement_file in judgements:
            # check if the image_id in data set split.
            image_id = judgement_file.rstrip(".json")
            if int(image_id) not in split_list:
                continue

            # check if the image file exists
            image_for_judgement = image_id + ".png"
            if image_for_judgement not in images:
                continue

            labels = json.load(open(os.path.join(DATA_PATH, judgement_file)))
            comparisons = labels['intrinsic_comparisons']
            points = labels['intrinsic_points']
            id_to_points = {p['id']: p for p in points}

            for c in comparisons:
                darker = c['darker']
                if darker not in ('1', '2', 'E'):
                    continue

                weight = c['darker_score']
                if weight <= 0.0 or weight is None:
                    continue

                point1 = id_to_points[c['point1']]
                point2 = id_to_points[c['point2']]

                if not point1['opaque'] or not point2['opaque']:
                    continue

                # Add all the items for this iteration
                data_items.append([os.path.join(DATA_PATH, image_for_judgement), point1, point2, label_map[darker]])

        print("Dataset Debug. Mode - ", mode, "items length - ", len(data_items))
        return data_items

    @staticmethod
    def get_cropped_region(image, point):
        rows, cols = image.size
        x, y = int(point['y'] * rows), int(point['x'] * cols)
        old_im = image.crop((x - crop_width, y - crop_width, x + crop_width + 1, y + crop_width + 1))

        # If we want cropped image in the corner in the center
        # left_top = max(0, x - crop_width), max(0, y - crop_width)
        # right_bottom = min(rows, x + crop_width), min(cols, y + crop_width)
        # old_im = image.crop((left_top[0], left_top[1], right_bottom[0] + 1, right_bottom[1] + 1))
        # old_size = old_im.size
        #
        # if old_size[0] < crop_width * 2 + 1 or old_size[1] < crop_width * 2 + 1:
        #     new_im = Image.new("RGB", (63, 63))  ## luckily, this is already black!
        #     new_im.paste(old_im, (int((new_size[0] - old_size[0]) / 2), int((new_size[1] - old_size[1]) / 2)))
        # else:
        #     new_im = old_im

        return old_im

    def __getitem__(self, index):
        image_path, point1, point2, label = self.data_items[index]
        image = Image.open(image_path)

        point_1_img, point_2_img = self.get_cropped_region(image, point1), self.get_cropped_region(image, point2)

        if self.resize_transform is not None:
            image = self.resize_transform(image)

        if self.transforms is not None:
            image = self.transforms(image)
            point_1_img = self.transforms(point_1_img)
            point_2_img = self.transforms(point_2_img)

        return image, point_1_img, point_2_img, label

    def __len__(self):
        return len(self.data_items)
