import nibabel as nib
import numpy as np
import cv2

class ImageLoader:
    @staticmethod
    def rearrange_dimensions(data):
        if data.shape[0] > data.shape[-1]:
            return np.transpose(data, (2, 0, 1))
        return data

    @staticmethod
    def load_nii_central_slice_lazy(file_path, dtype=np.float32):
        img = nib.load(file_path)
        data = img.dataobj  # acesso preguiçoso
        data = ImageLoader.rearrange_dimensions(data)
        idx = data.shape[0] // 2
        slice_ = np.asarray(data[idx], dtype=dtype)
        return slice_

    @staticmethod
    def normalize_image(img):
        img = img - np.min(img)
        img = img / (np.max(img) + 1e-8)
        return (img * 255).astype(np.uint8)

    @staticmethod
    def load_image(file_path):
        slice_image = ImageLoader.load_nii_central_slice_lazy(file_path)
        slice_image = np.rot90(slice_image)
        norm_image = ImageLoader.normalize_image(slice_image)
        return cv2.cvtColor(norm_image, cv2.COLOR_GRAY2RGB)
