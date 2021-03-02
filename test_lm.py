import os
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import json
from oriented_features.full_pipeline.bop_dataset import BopDataset
from argparse import Namespace
from feature_graph.utils.vis import plotImages
from tqdm import tqdm

os.chdir("/home/qiaog/src/DTOID/")
from wrapper import DTOIDWrapper

def main():
    model = DTOIDWrapper()

    args = Namespace()
    args.bop_root = "/home/qiaog/datasets/bop/"
    args.dataset_name = "lm"
    args.model_type = None
    args.split_name = "bop_test"
    args.split = "test"
    args.split_type = None
    args.skip = 1

    dataset = BopDataset(args)

    obj_ids_str = ["%02d" % _  for _ in dataset.obj_ids]
    print("len(dataset):", len(dataset))
    print("obj_ids_str:", obj_ids_str)

    results = []

    try:
        last_id = None
        for d in tqdm(dataset):
            img_numpy = d['img']
            mask_gt = d['mask_gt']
            mask_gt_visib = d['mask_gt_visib']
            obj_id = d['obj_id']
            scene_id = d['scene_id']
            im_id = d['im_id']
            obj_id_str = "%02d" % obj_id

            if last_id is None:
                last_id = obj_id_str
                
            if last_id != obj_id_str:
                model.clearCache()
            model.getTemplates(obj_id_str)

            out = model(img_numpy, obj_id_str)
            pred_seg = out['pred_seg_np'][0]
            pred_bbox_np = out['pred_bbox_np'][0]
            pred_scores_np = out['pred_scores_np'][0]
            network_w, network_h, img_w, img_h = out['network_w'], out['network_h'], out['img_w'], out['img_h']

            # x1, y1, x2, y2 = pred_bbox_np
            # temp_score = pred_scores_np

            # x1 = int(x1 / network_w * img_w)
            # x2 = int(x2 / network_w * img_w)
            # y1 = int(y1 / network_h * img_h)
            # y2 = int(y2 / network_h * img_h)

            # img_temp = img_numpy.copy()
            # rec_color = (0, 255, 255)
            # cv2.rectangle(img_temp,
            #             (x1, y1),
            #             (x2, y2),
            #             rec_color,2)
            # plotImages(
            #     [img_temp, mask_gt_visib, pred_seg, pred_seg > 0],
            #     ['pred bbox', 'GT mask', "pred seg map", "pred seg mask"]
            # )
            # plt.show()

            IoU = np.logical_and(mask_gt > 0, pred_seg > 0).sum() / np.logical_or(mask_gt > 0, pred_seg > 0).sum()
            IoU_visible = np.logical_and(mask_gt_visib > 0, pred_seg > 0).sum() / np.logical_or(mask_gt_visib > 0, pred_seg > 0).sum()

            results.append({
                "obj_id": obj_id, 
                "scene_id": scene_id,
                "im_id": im_id,
                "IoU": IoU,
                "IoU_visible": IoU_visible
            })
        
    except KeyboardInterrupt:
        print("Warning: End early")
        pass

    print("Average IoU:", np.mean([_['IoU'] for _ in results]))
    print("Average IoU_visible:", np.mean([_['IoU_visible'] for _ in results]))
        
    json.dump(results, open("test_result_%s_%s.json" % (args.dataset_name, args.split_name), "w"), indent="  ")

if __name__ == "__main__":
    main()