import sys
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import os
import pickle

from metric_evaluation.data_loader import DataLoader
from metric_evaluation.metrics import db_eval_boundary, db_eval_iou
from metric_evaluation import utils


class VOSEvaluation(object):
    def __init__(self, frames_root, gt_root, res_root, sequences='all', re_evaluate=False):
        """
        Class to evaluate DAVIS sequences from a certain set and for a certain task
        :param davis_root: Path to the DAVIS folder that contains JPEGImages, Annotations, etc. folders.
        :param task: Task to compute the evaluation, chose between semi-supervised or unsupervised.
        :param gt_set: Set to compute the evaluation
        :param sequences: Sequences to consider for the evaluation, 'all' to use all the sequences in a set.
        """
        self.frames_root = frames_root
        self.dataloader = DataLoader(frames_root=frames_root, gt_root=gt_root, res_root=res_root, sequences=sequences)
        self.metrics_res = None
        self.re_evaluate = re_evaluate

    @staticmethod
    def _evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metric):
        if all_res_masks.shape[0] > all_gt_masks.shape[0]:
            sys.stdout.write("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
            sys.exit()
        elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
            zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
            all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
        j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
        for ii in range(all_gt_masks.shape[0]):
            if 'J' in metric:
                j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
            if 'F' in metric:
                f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
        return j_metrics_res, f_metrics_res

    def _read_results(self, eval_path):
        if not os.path.exists(eval_path):
            print(f"Results are no available in the specified path, evaluating model")
            return False
        else:
            print(f"Reading evaluation from specified path")
            with open(eval_path, 'rb') as handle:
                self.metrics_res = pickle.load(handle)
            return True

    def evaluate(self, eval_path, metric=('J', 'F')):
        if (not self.re_evaluate) and self._read_results(eval_path):
            return

        metric = metric if isinstance(metric, tuple) or isinstance(metric, list) else [metric]
        if 'T' in metric:
            raise ValueError('Temporal metric not supported!')
        if 'J' not in metric and 'F' not in metric:
            raise ValueError('Metric possible values are J for IoU or F for Boundary')

        # Containers
        metrics_res = {}
        if 'J' in metric:
            metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}, "Per_seq_object_frame": {}}
        if 'F' in metric:
            metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}, "Per_seq_object_frame": {}}

        # Sweep all sequences
        for seq in list(self.dataloader.get_sequences()):
            all_gt_masks, _, all_masks_id = self.dataloader.get_all_masks(seq, True)
            all_gt_masks, all_masks_id = all_gt_masks[:, 1:, :, :], all_masks_id[1:]

            all_pr_masks, _, all_preds_id = self.dataloader.get_all_predictions(seq, True)
            all_pr_masks, all_preds_id = all_pr_masks[:, 1:, :, :], all_preds_id[1:]

            j_metrics_res, f_metrics_res = self._evaluate_semisupervised(all_gt_masks, all_pr_masks, None, metric)

            metrics_res['J']["M_per_object"][seq] = {}
            metrics_res['J']["Per_seq_object_frame"][seq] = {}
            metrics_res['F']["M_per_object"][seq] = {}
            metrics_res['F']["Per_seq_object_frame"][seq] = {}
            for ii in range(all_gt_masks.shape[0]):
                if 'J' in metric:
                    [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
                    metrics_res['J']["M"].append(JM)
                    metrics_res['J']["R"].append(JR)
                    metrics_res['J']["D"].append(JD)
                    metrics_res['J']["M_per_object"][seq][ii+1] = JM
                    metrics_res['J']["Per_seq_object_frame"][seq][ii+1] = j_metrics_res[ii]
                if 'F' in metric:
                    [FM, FR, FD] = utils.db_statistics(f_metrics_res[ii])
                    metrics_res['F']["M"].append(FM)
                    metrics_res['F']["R"].append(FR)
                    metrics_res['F']["D"].append(FD)
                    metrics_res['F']["M_per_object"][seq][ii+1] = FM
                    metrics_res['F']["Per_seq_object_frame"][seq][ii+1] = f_metrics_res[ii]

        self.metrics_res = metrics_res
        print("Saving evaluation results to specified path")
        with open(eval_path, 'wb') as handle:
            pickle.dump(metrics_res, handle)
