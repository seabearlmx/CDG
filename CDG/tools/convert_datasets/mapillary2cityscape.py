# Obtained from: https://github.com/openseg-group/openseg.pytorch.git
# Aiming to convert the annotation format to be TrainId and match the classes in Cityscapes dataset

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import mmengine
import PIL.Image as Image
import cv2
import numpy as np

LABEL_DIR = "label"


def convert_to_train_id(trans_idx, train_mask_folder, train_label_dir, filename):
    if filename.endswith(".png"):
        maskpath = os.path.join(train_mask_folder, filename)
        if os.path.isfile(maskpath):
            mask = np.asarray(Image.open(maskpath))
            mask = trans_idx[mask]
            cv2.imwrite(
                os.path.join(train_label_dir, filename),
                mask.astype(np.uint8),
            )
        else:
            print("cannot find the mask:", maskpath)


class MapillaryGenerator(object):
    def __init__(self, args, label_dir=LABEL_DIR):
        self.args = args
        self.version = args.version
        self.train_label_dir = os.path.join(self.args.save_dir, "train", label_dir)
        self.val_label_dir = os.path.join(self.args.save_dir, "val", label_dir)
        if not os.path.exists(self.train_label_dir):
            os.makedirs(self.train_label_dir)

        if not os.path.exists(self.val_label_dir):
            os.makedirs(self.val_label_dir)

    def generate_label(self):
        trans_idx = self.get_trans_idx()

        # train_img_folder = os.path.join(self.args.ori_root_dir, 'images/training')
        train_mask_folder = os.path.join(
            self.args.ori_root_dir, f"training/labels"
        )

        # val_img_folder = os.path.join(self.args.ori_root_dir, 'images/validation')
        val_mask_folder = os.path.join(
            self.args.ori_root_dir, f"validation/labels"
        )
        for filename in os.listdir(train_mask_folder):
            print(filename)
            convert_to_train_id(
                trans_idx, train_mask_folder, self.train_label_dir, filename
            )
        for filename in os.listdir(val_mask_folder):
            print(filename)
            convert_to_train_id(
                trans_idx, val_mask_folder, self.val_label_dir, filename
            )

    def get_trans_idx(self):
        # class name and index of cityscapes dataset
        # [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33],
        if not self.args.train_id:
            class_name_dict = {
                7: "road",
                8: "sidewalk",
                11: "building",
                12: "wall",
                13: "fence",
                17: "pole",
                19: "trafficlight",
                20: "trafficsign",
                21: "vegetation",
                22: "terrain",
                23: "sky",
                24: "person",
                25: "rider",
                26: "car",
                27: "truck",
                28: "bus",
                31: "train",
                32: "motorcycle",
                33: "bicycle",
            }
        else:
            class_name_dict = {
                0: "road",
                1: "sidewalk",
                2: "building",
                3: "wall",
                4: "fence",
                5: "pole",
                6: "trafficlight",
                7: "trafficsign",
                8: "vegetation",
                9: "terrain",
                10: "sky",
                11: "person",
                12: "rider",
                13: "car",
                14: "truck",
                15: "bus",
                16: "train",
                17: "motorcycle",
                18: "bicycle",
            }
        class_name_dict = {v: k for k, v in class_name_dict.items()}

        # class name and index of mapillary dataset
        with open(
                os.path.join(self.args.ori_root_dir, f"config_{self.version}.json")
        ) as config_file:
            labels = json.load(config_file)["labels"]

        print("Following classes are mapped to corresponding classes in cityscapes:")
        mapillary2city = [255] * len(labels)
        ignored = []

        for label_id, label in enumerate(labels):
            name = label["readable"].lower().replace(" ", "").replace("-", "")
            if name in class_name_dict.keys():
                mapillary2city[label_id] = class_name_dict[name]
                print(
                    "{} => {}: {} => {}".format(
                        name, name, label_id, class_name_dict[name]
                    )
                )
            elif "trafficsign" in name or "front" in name or "back" in name:
                mapillary2city[label_id] = class_name_dict["trafficsign"]
                print(
                    "{} => {}: {} => {}".format(
                        name, "traffic sign", label_id, class_name_dict["trafficsign"]
                    )
                )
            elif "onrail" in name:
                mapillary2city[label_id] = class_name_dict["train"]
                print(
                    "{} => {}: {} => {}".format(
                        name, "train", label_id, class_name_dict["train"]
                    )
                )
            elif "cyclist" in name or "rider" in name:
                mapillary2city[label_id] = class_name_dict["rider"]
                print(
                    "{} => {}: {} => {}".format(
                        name, "rider", label_id, class_name_dict["rider"]
                    )
                )
            elif "pole" in name or "streetlight" in name:
                mapillary2city[label_id] = class_name_dict["pole"]
                print(
                    "{} => {}: {} => {}".format(
                        name, "pole", label_id, class_name_dict["pole"]
                    )
                )
            elif "curb" in name or "pedestrianarea" in name:
                mapillary2city[label_id] = class_name_dict["sidewalk"]
                print(
                    "{} => {}: {} => {}".format(
                        name, "sidewalk", label_id, class_name_dict["sidewalk"]
                    )
                )
            elif (
                    "crosswalkplain" in name
                    or "parking" in name
                    or "bikelane" in name
                    or "servicelane" in name
                    or "lanemarking" in name
            ):
                mapillary2city[label_id] = class_name_dict["road"]
                print(
                    "{} => {}: {} => {}".format(
                        name, "road", label_id, class_name_dict["road"]
                    )
                )
            else:
                ignored.append(name)

        print("\nFollowing classes are mapped to void class:")
        print(ignored)
        return np.asarray(mapillary2city, dtype=np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ori_root_dir",
        type=str,
        help="The directory of the cityscapes data.",
    )
    parser.add_argument(
        "save_dir",
        type=str,
        help="The directory to save the data.",
    )
    parser.add_argument(
        "--version",
        default="v1.2",
        type=str,
    )
    parser.add_argument(
        "--train_id",
        action="store_true",
    )

    args = parser.parse_args()

    Mapillary_generator = MapillaryGenerator(args)
    Mapillary_generator.generate_label()
