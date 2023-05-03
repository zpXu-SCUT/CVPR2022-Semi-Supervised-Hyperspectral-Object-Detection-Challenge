import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadMaskedImageFromFile:

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
            
        img_bytes = self.file_client.get(filename  + '.png')
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)
            
        if results['mask_prefix'] is not None:
            maskname = osp.join(results['mask_prefix'],
                                results['img_info']['filename'])
        else:
            maskname = results['img_info']['maskname']

        mask_bytes = self.file_client.get(maskname + '_mask.png')
        mask = mmcv.imfrombytes(mask_bytes, flag=self.color_type)
        if self.to_float32:
            mask = mask.astype(np.float32)

        mask[mask == 255] = 1
        img = img * mask

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str
    
@PIPELINES.register_module()
class LoadMaskedHSIImageFromFile:

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
            
        # img_bytes = self.file_client.get(filename  + '_rd.npy')
        # img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        img = np.load(filename  + '_rd.npy')
        if self.to_float32:
            img = img.astype(np.float32)
            
        img = img / 1000.0
        img = img.astype(np.float32)
            
        if results['mask_prefix'] is not None:
            maskname = osp.join(results['mask_prefix'],
                                results['img_info']['filename'])
        else:
            maskname = results['img_info']['maskname']

        mask_bytes = self.file_client.get(maskname + '_mask.png')
        mask = mmcv.imfrombytes(mask_bytes, flag=self.color_type)
        if self.to_float32:
            mask = mask.astype(np.float32)

        mask[mask == 255] = 1
        mask = np.repeat(mask, 17, axis = 2)
        img = img * mask

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        results['img_norm_cfg'] = dict(
            mean=np.zeros(51, dtype=np.float32),
            std=np.ones(51, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str
