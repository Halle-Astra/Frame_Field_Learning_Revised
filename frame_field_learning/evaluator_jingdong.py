import os
import csv

from tqdm import tqdm
from multiprocess import Pool, Process, Queue
from functools import partial
import time

import torch
import torch.utils.data
# from pytorch_memlab import profile, profile_every

from . import inference, save_utils, polygonize
from . import local_utils
from . import measures

from lydorn_utils import run_utils
from lydorn_utils import python_utils
from lydorn_utils import print_utils
from lydorn_utils import async_utils
import numpy as np

debug_halle = False
debug_halle_saving = True# True will save files, False will not.
debug_halle_final = False

class Evaluator:
    def __init__(self, gpu: int, config: dict, shared_dict, barrier, model, run_dirpath):
        self.gpu = gpu
        self.config = config
        assert 0 < self.config["eval_params"]["batch_size_mult"], \
            "batch_size_mult in polygonize_params should be at least 1."

        self.shared_dict = shared_dict
        self.barrier = barrier
        self.model = model

        self.checkpoints_dirpath = run_utils.setup_run_subdir(run_dirpath,
                                                              config["optim_params"]["checkpoints_dirname"])

        self.eval_dirpath = os.path.join(config["data_root_dir"], "eval_runs", os.path.split(run_dirpath)[-1])
        if self.gpu == 0:
            os.makedirs(self.eval_dirpath, exist_ok=True)
            print_utils.print_info("Saving eval outputs to {}".format(self.eval_dirpath))

    # @profile
    def evaluate(self, split_name: str, ds: torch.utils.data.DataLoader):
        print(f"FILE:{os.path.split(__file__)[-1]}, Info: evaluate is running by this file.")
        # Prepare data saving:
        flag_filepath_format = os.path.join(self.eval_dirpath, split_name, "{}.flag")

        # Loading model
        self.load_checkpoint()
        self.model.eval()

        # Create pool for multiprocessing
        pool = None
        if not self.config["eval_params"]["patch_size"]:
            # If single image is not being split up, then a pool to process each sample in the batch makes sense
            if self.config['num_workers']:
                pool = Pool(processes=self.config["num_workers"])

        compute_polygonization = self.config["eval_params"]["save_individual_outputs"]["poly_shapefile"] or \
                                 self.config["eval_params"]["save_individual_outputs"]["poly_geojson"] or \
                                 self.config["eval_params"]["save_individual_outputs"]["poly_viz"] or \
                                 self.config["eval_params"]["save_aggregated_outputs"]["poly_coco"]
        # if debug_halle:
        pool = None# added by jingdong, I don't want to use multiprocessing.
        # Saving individual outputs to disk:
        save_individual_outputs = True in self.config["eval_params"]["save_individual_outputs"].values()
        saver_async = None
        if save_individual_outputs:
            save_outputs_partial = partial(save_utils.save_outputs, config=self.config, eval_dirpath=self.eval_dirpath,
                                           split_name=split_name, flag_filepath_format=flag_filepath_format)
            saver_async = async_utils.Async(save_outputs_partial)# bug 根源
            saver_async.start()

        # Saving aggregated outputs
        save_aggregated_outputs = True in self.config["eval_params"]["save_aggregated_outputs"].values()

        tile_data_list = []

        if self.gpu == 0:
            tile_iterator = tqdm(ds, desc="Eval {}: ".format(split_name), leave=True)# ds
        else:
            tile_iterator = ds
        # if True:
        for tile_i, tile_data in enumerate(tile_iterator):
            # tile_i = 0
            # print(f"FILE:{os.path.abspath('.')}, INFO: path checking...")
            # tile_data = np.load('./tile_data_0902.npy',allow_pickle = True).tolist() # the relative path is relative to the path of this file ?
            if debug_halle:
                if tile_i <900:#or tile_i == len(tile_iterator)-1:
                    if tile_i == len(tile_iterator)-1:
                        print(f"FILE:{os.path.split(__file__)[-1]}, var inspecting, tile_data:{tile_data}.")
                        import numpy as np
                        np.save('tile_data_0902',tile_data)
                    continue
            # print(f"FILE:{os.path.split(__file__)[-1]}, INFO: It is {tile_i}th iteration in evaluating.")
            # --- Inference, add result to tile_data_list
            if self.config["eval_params"]["patch_size"] is not None:
                # Cut image into patches for inference
                inference.inference_with_patching(self.config, self.model, tile_data)
            else:
                # Feed images as-is to the model
                inference.inference_no_patching(self.config, self.model, tile_data)

            tile_data_list.append(tile_data)

            # --- Accumulate batches into tile_data_list until capacity is reached (or this is the last batch)
            if self.config["eval_params"]["batch_size_mult"] <= len(tile_data_list)\
                    or tile_i == len(tile_iterator) - 1 or debug_halle_final:
                # Concat tensors of tile_data_list
                accumulated_tile_data = {}
                for key in tile_data_list[0].keys():
                    if isinstance(tile_data_list[0][key], list):
                        accumulated_tile_data[key] = [item for _tile_data in tile_data_list for item in _tile_data[key]]
                    elif isinstance(tile_data_list[0][key], torch.Tensor):
                        accumulated_tile_data[key] = torch.cat([_tile_data[key] for _tile_data in tile_data_list], dim=0)
                    else:
                        raise TypeError(f"Type {type(tile_data_list[0][key])} is not handled!")
                tile_data_list = []  # Empty tile_data_list
            else:
                # tile_data_list is not full yet, continue running inference...
                continue

            # --- Polygonize
            if compute_polygonization:
                crossfield = accumulated_tile_data["crossfield"] if "crossfield" in accumulated_tile_data else None
                accumulated_tile_data["polygons"], accumulated_tile_data["polygon_probs"] = polygonize.polygonize(
                    self.config["polygonize_params"], accumulated_tile_data["seg"],
                    crossfield_batch=crossfield,
                    pool=pool)
            if debug_halle_saving:
                print(f"FILE:{os.path.split(__file__)[-1]}, INFO: Saving will be started.")
                # --- Save output
                if self.config["eval_params"]["save_individual_outputs"]["seg_mask"] or \
                        self.config["eval_params"]["save_aggregated_outputs"]["seg_coco"]:
                    # Take seg_interior:
                    seg_pred_mask = self.config["eval_params"]["seg_threshold"] < accumulated_tile_data["seg"][:, 0, ...]
                    accumulated_tile_data["seg_mask"] = seg_pred_mask

                accumulated_tile_data = local_utils.batch_to_cpu(accumulated_tile_data)
                # import numpy as np
                # print(f"FILE:{os.path.split(__file__)[-1]}, var inspecting, polygon's shape:{accumulated_tile_data['polygons'].shape},"
                #       f"image_id:{accumulated_tile_data['image_id'][0]}")
                sample_list = local_utils.split_batch(accumulated_tile_data)

                # Save individual outputs:
                if save_individual_outputs:
                    for sample in sample_list:
                        saver_async.add_work(sample)

                # Store aggregated outputs:
                if save_aggregated_outputs:
                    self.shared_dict["name_list"].extend(accumulated_tile_data["name"])
                    if self.config["eval_params"]["save_aggregated_outputs"]["stats"]:
                        y_pred = accumulated_tile_data["seg"][:, 0, ...].cpu()
                        if "gt_mask" in accumulated_tile_data:
                            y_true = accumulated_tile_data["gt_mask"][:, 0, ...]
                        elif "gt_polygons_image" in accumulated_tile_data:
                            y_true = accumulated_tile_data["gt_polygons_image"][:, 0, ...]
                        else:
                            raise ValueError("Either gt_mask or gt_polygons_image should be in accumulated_tile_data")
                        iou = measures.iou(y_pred.reshape(y_pred.shape[0], -1), y_true.reshape(y_true.shape[0], -1),
                                           threshold=self.config["eval_params"]["seg_threshold"])
                        self.shared_dict["iou_list"].extend(iou.cpu().numpy())
                    if self.config["eval_params"]["save_aggregated_outputs"]["seg_coco"]:
                        for sample in sample_list:
                            annotations = save_utils.seg_coco(sample)
                            self.shared_dict["seg_coco_list"].extend(annotations)
                    if self.config["eval_params"]["save_aggregated_outputs"]["poly_coco"]:
                        for sample in sample_list:
                            annotations = save_utils.poly_coco(sample["polygons"], sample["polygon_probs"], sample["image_id"].item())
                            self.shared_dict["poly_coco_list"].append(annotations)  # annotations could be a dict, or a list
        # END of loop over samples
        # 总之结论：bug重点在这之前
        print(f"FILE:{os.path.split(__file__)[-1]}, INFO: Testing...Program will be terminated.")
        self.barrier.abort()
        print(f"FILE:{os.path.split(__file__)[-1]}, INFO: Barrier has been aborted.")
        # exit('Can I exit this process?')#return
        # Save aggregated results
        if save_aggregated_outputs:
            self.barrier.wait()  # Wait on all processes so that shared_dict is synchronized.
            if self.gpu == 0:
                if self.config["eval_params"]["save_aggregated_outputs"]["stats"]:
                    print("Start saving stats:")
                    # Save sample_stats in CSV:
                    t1 = time.time()
                    stats_filepath = os.path.join(self.eval_dirpath, "{}.stats.csv".format(split_name))
                    stats_file = open(stats_filepath, "w")
                    fnames = ["name", "iou"]
                    writer = csv.DictWriter(stats_file, fieldnames=fnames)
                    writer.writeheader()
                    for name, iou in sorted(zip(self.shared_dict["name_list"], self.shared_dict["iou_list"]), key=lambda pair: pair[0]):
                        writer.writerow({
                            "name": name,
                            "iou": iou
                        })
                    stats_file.close()
                    print(f"Finished in {time.time() - t1:02}s")

                if self.config["eval_params"]["save_aggregated_outputs"]["seg_coco"]:
                    print("Start saving seg_coco:")
                    t1 = time.time()
                    seg_coco_filepath = os.path.join(self.eval_dirpath, "{}.annotation.seg.json".format(split_name))
                    python_utils.save_json(seg_coco_filepath, list(self.shared_dict["seg_coco_list"]))
                    print(f"Finished in {time.time() - t1:02}s")

                if self.config["eval_params"]["save_aggregated_outputs"]["poly_coco"]:
                    print("Start saving poly_coco:")
                    poly_coco_base_filepath = os.path.join(self.eval_dirpath, f"{split_name}.annotation.poly")
                    t1 = time.time()
                    save_utils.save_poly_coco(self.shared_dict["poly_coco_list"], poly_coco_base_filepath)
                    print(f"Finished in {time.time() - t1:02}s")

        # Sync point of individual outputs
        if save_individual_outputs:
            print_utils.print_info(f"GPU {self.gpu} -> INFO: Finishing saving individual outputs.")
            saver_async.join()
            self.barrier.wait()  # Wait on all processes so that all saver_asyncs are finished
            saver_async.cancel_join()
            print('saver aync terminate joining...')

    def load_checkpoint(self):
        """
        Loads best val checkpoint in checkpoints_dirpath
        """
        filepaths = python_utils.get_filepaths(self.checkpoints_dirpath, startswith_str="checkpoint.best_val.",
                                               endswith_str=".tar")
        if len(filepaths):
            filepaths = sorted(filepaths)
            filepath = filepaths[-1]  # Last best val checkpoint filepath in case there is more than one
            if self.gpu == 0:
                print_utils.print_info("Loading best val checkpoint: {}".format(filepath))
        else:
            # No best val checkpoint fount: find last checkpoint:
            filepaths = python_utils.get_filepaths(self.checkpoints_dirpath, endswith_str=".tar",
                                                   startswith_str="checkpoint.")
            if len(filepaths) == 0:
                raise FileNotFoundError("No checkpoint could be found at that location.")
            filepaths = sorted(filepaths)
            filepath = filepaths[-1]  # Last checkpoint
            if self.gpu == 0:
                print_utils.print_info("Loading last checkpoint: {}".format(filepath))
        # map_location is used to load on current device:
        checkpoint = torch.load(filepath, map_location="cuda:{}".format(self.gpu))

        self.model.module.load_state_dict(checkpoint['model_state_dict'])
