from operator import gt
import os
from glob import glob
from collections import defaultdict
import numpy as np
from PIL import Image


class DataLoader(object):
    SUBSET_OPTIONS = ['train', 'val', 'test-dev', 'test-challenge']
    TASKS = ['semi-supervised', 'unsupervised']
    DATASET_WEB = 'https://davischallenge.org/davis2017/code.html'
    VOID_LABEL = 255

    def __init__(self, frames_root, gt_root, res_root, sequences='all'):
        """
        Class to read the DAVIS dataset
        :param root: Path to the DAVIS folder that contains JPEGImages, Annotations, etc. folders.
        :param task: Task to load the annotations, choose between semi-supervised or unsupervised.
        :param subset: Set to load the annotations
        :param sequences: Sequences to consider, 'all' to use all the sequences in a set.
        :param resolution: Specify the resolution to use the dataset, choose between '480' and 'Full-Resolution'
        """
        
        self.img_path = frames_root
        self.mask_path = gt_root
        self.pr_path = res_root

        if sequences == 'all':
            seq_folders = glob(os.path.join(self.pr_root,'*'))
            seqs = [os.path.split(seq_folder)[1] for seq_folder in seq_folders]
            sequences_names = [x.strip() for x in seqs]
        else:
            sequences_names = sequences if isinstance(sequences, list) else [sequences]
        self.sequences = defaultdict(dict)

        for seq in sequences_names:
            images = np.sort(glob(os.path.join(self.img_path, seq, '*.jpg'))).tolist()
            self.sequences[seq]['images'] = images
            masks = np.sort(glob(os.path.join(self.mask_path, seq, '*.png'))).tolist()
            masks.extend([-1] * (len(images) - len(masks)))
            self.sequences[seq]['masks'] = masks
            predictions = np.sort(glob(os.path.join(self.pr_path, seq, '*.png'))).tolist()
            predictions.extend([-1] * (len(images) - len(predictions)))
            self.sequences[seq]['predictions'] = predictions

    def get_contents(self, sequence):
        images = []
        masks = []
        preds = []
        for img, msk, pr in zip(self.sequences[sequence]['images'], 
                                self.sequences[sequence]['masks'],
                                self.sequences[sequence]['predictions']):
            image = np.array(Image.open(img))
            mask = None if msk is None else np.array(Image.open(msk))
            prediction = None if pr is None else np.array(Image.open(pr))
            images.append(image)
            masks.append(mask)
            preds.append(prediction)
        return images, masks, preds

    def _get_all_elements(self, sequence, obj_type):
        obj = np.array(Image.open(self.sequences[sequence][obj_type][0]))
        all_objs = np.zeros((len(self.sequences[sequence][obj_type]), *obj.shape))
        obj_id = []
        for i, obj in enumerate(self.sequences[sequence][obj_type]):
            all_objs[i, ...] = np.array(Image.open(obj))
            obj_id.append(os.path.splitext(os.path.split(obj)[-1])[0])
        return all_objs, obj_id

    def get_all_images(self, sequence):
        return self._get_all_elements(sequence, 'images')

    def get_all_masks(self, sequence, separate_objects_masks=False):
        masks, masks_id = self._get_all_elements(sequence, 'masks')
        masks_void = np.zeros_like(masks)

        # Separate void and object masks
        for i in range(masks.shape[0]):
            masks_void[i, ...] = masks[i, ...] == 255
            masks[i, masks[i, ...] == 255] = 0

        if separate_objects_masks:
            num_objects = int(np.max(masks[0, ...]))
            tmp = np.ones((num_objects, *masks.shape))
            tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
            masks = (tmp == masks[None, ...])
            masks = masks > 0
        return masks, masks_void, masks_id

    def get_all_predictions(self, sequence, separate_objects_masks=False):
        preds, preds_id = self._get_all_elements(sequence, 'predictions')
        preds_void = np.zeros_like(preds)

        # Separate void and object masks
        for i in range(preds.shape[0]):
            preds_void[i, ...] = preds[i, ...] == 255
            preds[i, preds[i, ...] == 255] = 0

        if separate_objects_masks:
            num_objects = int(np.max(preds[0, ...]))
            tmp = np.ones((num_objects, *preds.shape))
            tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
            preds = (tmp == preds[None, ...])
            preds = preds > 0
        return preds, preds_void, preds_id

    def get_sequences(self):
        for seq in self.sequences:
            yield seq

