import nibabel as nib
import pydicom
import tempfile
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt
import gc
import os
import json
import cv2

class ImageOperations:
    def __init__(self):
        pass

    @staticmethod
    def rearrange_dimensions(data):
        if data.shape[0] > data.shape[-1]:
            return np.transpose(data, (2, 0, 1))
        else:
            return data

    @staticmethod
    def load_nii_central_slice_lazy(file_path, dtype=np.float32):
        img = nib.load(file_path)

        affine = img.affine
        data = img.dataobj  # lazy loading
        data = ImageOperations.rearrange_dimensions(data)
        
        nb_of_slices = data.shape[0]
        idx = nb_of_slices // 2
        slice_ = np.asarray(data[idx], dtype=dtype)
        return slice_, affine, nb_of_slices

    @staticmethod
    def load_nii_central_slice(file_path, dtype=np.float32, flip=False):
        nii = nib.load(file_path).get_fdata()
        nii = ImageOperations.rearrange_dimensions(nii)
        idx = nii.shape[0] // 2
        central_slice = nii[idx].astype(dtype)
        central_slice = np.rot90(central_slice)  # Rotate the slice for vertical orientation
        if flip:
            central_slice = np.flip(central_slice, axis=0)
        return central_slice

    @staticmethod
    def normalize_image(img):
        img = img - np.min(img)
        img = img / (np.max(img) + 1e-8)
        return (img * 255).astype(np.uint8)


    @staticmethod
    def load_image(file_path):
        if file_path.lower().endswith('.nii') or file_path.lower().endswith('.nii.gz'):
            slice_image, affine, nb_of_slices = ImageOperations.load_nii_central_slice_lazy(file_path)
        else:
            slice_image, affine, nb_of_slices = ImageOperations.load_dicom_central_slice_lazy(file_path)

        norm_image = ImageOperations.normalize_image(slice_image)

        return cv2.cvtColor(norm_image, cv2.COLOR_GRAY2RGB), affine, nb_of_slices

    @staticmethod
    def load_dicom_central_slice_lazy(folder_path, dtype=np.float32, flip=False):
        """
        Carrega o slice central de uma série DICOM armazenada em uma pasta.
        Retorna (slice_normalizado_RGB, affine_fake, nb_of_slices).
        """
        # Lista de arquivos DICOM
        dicom_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if f.lower().endswith(('.dcm', '.dicom'))]

        if len(dicom_files) == 0:
            raise FileNotFoundError(f"No dicom found files in {folder_path}")

        # Ordering by InstanceNumber if it exists
        dicoms = [pydicom.dcmread(f) for f in dicom_files]
        dicoms.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0)))

        nb_of_slices = len(dicoms)
        central_index = nb_of_slices // 2
        central_slice = dicoms[central_index].pixel_array.astype(dtype)
        central_slice = np.rot90(central_slice, k=-1)  # Rotate the slice for vertical orientation
        if flip:
            central_slice = np.flip(central_slice, axis=0)

        return central_slice, np.eye(4), nb_of_slices
    

    @staticmethod
    def load_central_slice_any(file_path, dtype=np.float32, flip=False):
        if os.path.isdir(file_path):
            central_slice, _, _ = ImageOperations.load_dicom_central_slice_lazy(file_path, dtype=dtype, flip=flip)
            return np.rot90(central_slice)
        elif file_path.lower().endswith('.nii') or file_path.lower().endswith('.nii.gz'):
            return ImageOperations.load_nii_central_slice(file_path, dtype=dtype, flip=flip)
        else:
            raise ValueError(f"Not supported file format: {file_path}. Expected .nii or DICOM folder.")
    
    @staticmethod
    def load_dicom_slice(file_path, slice_index, dtype=np.float32):
        # found the slice index in the dicom folder
        if not os.path.isdir(file_path):
            raise ValueError(f"Expected a directory for DICOM files, got: {file_path}")
        dicom_files = [os.path.join(file_path, f) for f in os.listdir(file_path)
                       if f.lower().endswith(('.dcm', '.dicom'))]
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in {file_path}")
        
        dicoms = [pydicom.dcmread(f) for f in dicom_files]
        dicoms.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0)))
        if slice_index < 0 or slice_index >= len(dicoms):
            raise IndexError(f"Slice index {slice_index} out of range for {len(dicoms)} slices.")
        slice_data = dicoms[slice_index].pixel_array.astype(dtype)
        slice_data = np.rot90(slice_data, k=-1)  # Rotate the slice
        return slice_data
        
    @staticmethod
    def load_nii_slice(file_path, slice_index, dtype=np.float32):
        nii_obj = nib.load(file_path)
        nii_data = nii_obj.get_fdata()
        nii_data = ImageOperations.rearrange_dimensions(nii_data)
        slice_data = nii_data[slice_index, :, :]
        del nii_data, nii_obj
        gc.collect()
        slice_data = slice_data.astype(dtype)
        return slice_data
    
    @staticmethod
    def load_any_slice(file_path, slice_index, dtype=np.float32):
        if file_path.lower().endswith('.nii') or file_path.lower().endswith('.nii.gz'):
            return ImageOperations.load_nii_slice(file_path, slice_index, dtype=dtype)
        elif os.path.isdir(file_path):
            return ImageOperations.load_dicom_slice(file_path, slice_index, dtype=dtype)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    @staticmethod
    def display_thresholded_slice(img, mask, threshold):
        bin_mask = ThresholdOperations.threshold_image(img, mask, threshold)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img, cmap='gray')
        ax.imshow(bin_mask, cmap='jet', alpha=0.2)
        ax.text(0.5, 0.05, f'Threshold: {threshold:.2f}',
               ha='center', va='center',
               transform=ax.transAxes,
               color='white', fontsize=16)
        ax.axis('off')
        return fig

class MaskOperations:
    def __init__(self):
        pass

    @staticmethod
    def create_mask(image, points, reduction_scale=1.0):
        if len(points) < 3:
            raise ValueError("At least 3 points are required to create a polygon mask.")
        
        # Adjust the points to the original image scale
        scale = 1 / reduction_scale
        scaled_points = [(int(x * scale), int(y * scale)) for (x, y) in points]
        
        mask = np.zeros_like(image[:,:,0], dtype=np.uint8)
        pts = np.array([scaled_points], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)

        # Use only one channel for bitwise operations 
        result = cv2.bitwise_and(image, image, mask=mask)
        return result, mask

    @staticmethod
    def create_combined_mask(image, polygons, reduction_scale=1.0):
        combined_mask = np.zeros_like(image[:,:,0], dtype=np.uint8)  # ensure single channel
        for poly in polygons:
            if len(poly) >= 3:
                _, mask = MaskOperations.create_mask(image, poly, reduction_scale=reduction_scale)
                combined_mask = np.maximum(combined_mask, mask)  # union of all polygons
        result = cv2.bitwise_and(image, image, mask=combined_mask)
        return result, combined_mask
    
    @staticmethod 
    def create_mask_nifti(folder_path, original_affine):
        mask_path = os.path.join(folder_path, 'mask.nii')
        images = [np.array(Image.open(os.path.join(folder_path, f)).convert('L'))
                  for f in sorted(os.listdir(folder_path), 
                  key=lambda x: int(x.split('_')[1])) if f.endswith('.png')]
        volume = np.stack(images, axis=0)
        volume = np.transpose(volume, (0, 2, 1))  # Transpose to match NIfTI format
        volume = np.rot90(volume, k=2, axes=(1,2))  # Rotate to correct orientation
        nib.save(nib.Nifti1Image(volume, original_affine), mask_path)
        return mask_path

    @staticmethod
    def measure_mask_area(mask):
        return np.count_nonzero(mask)

    @staticmethod
    def save_nii_mask(mask, affine, nb_of_slices, file_path, file_name="dense.nii"):
        # transform in a mask of 1's
        rotated_mask = np.flip(np.rot90(mask), axis=0)
        mask_3d = np.where(rotated_mask > 0, 1, 0).astype(np.uint8)
        mask_3d = np.repeat(mask_3d[np.newaxis, :, :], nb_of_slices, axis=0)
        mask_3d = np.transpose(mask_3d, (0, 2, 1))
        mask_nii = nib.Nifti1Image(mask_3d, affine)
        nib.save(mask_nii, os.path.join(file_path, file_name))

    @staticmethod
    def save_mask_json(points, scale, file_path):
        os.makedirs(file_path, exist_ok=True)
        json_data = {"points": points,
                     "scale": scale}
        with open(os.path.join(file_path, "mask.json"), 'w') as f:
            json.dump(json_data, f)

    @staticmethod
    def save_mask(mask, affine, nb_of_slices, file_path, file_name="dense.nii", points=None, scale=1.0):
        os.makedirs(file_path, exist_ok=True)
        MaskOperations.save_nii_mask(mask, affine, nb_of_slices, file_path, file_name)
        assert points is not None, "Points must be provided to save mask JSON."
        MaskOperations.save_mask_json(points, scale, file_path)
    
if __name__ == "__main__":
    # Example usage
    image_name = "21991_40.nii"
    file_path = os.path.join("media", image_name)

    # load the image and the affine transformation
    image, affine, nb_of_slices = ImageOperations.load_image(file_path)
    
    # Normally the points would be obtained from streamlit canvas
    # Here we use a hardcoded example for demonstration
    points = [(491, 223), (393, 363), (582, 369)]

    # Create the mask and save it
    # The reduction scale is also obtained from the streamlit session
    # It depends from the original image size and the canvas size
    # It was necessary to better presentation of the image in the canvas
    result, mask = MaskOperations.create_mask(image, points, reduction_scale=0.3256)
    output_path = os.path.join(os.getcwd(), 'output', image_name.split('.')[0])
    MaskOperations.save_mask(mask, affine, nb_of_slices, file_path=output_path,
                              points=points, scale=0.3256)
    print("Mask created and saved successfully.")


class ThresholdOperations:
    def __init__(self):
        pass

    @staticmethod
    def normalize_data(data):
        mn, mx = data.min(), data.max()
        return (data - mn) / (mx - mn)
    
    @staticmethod
    def apply_threshold(image, mask, threshold):
        norm_image = ImageOperations.normalize_image(image)
        return (norm_image > threshold) & (mask > 0)

    @staticmethod
    def display_thresholded_slice(img, mask, threshold):
        bin_mask = ThresholdOperations.apply_threshold(img, mask, threshold)
        return ImageOperations.display_thresholded_slice(img, bin_mask, threshold)
    
    @staticmethod
    def threshold_image(img_slice, mask_slice, threshold):
        norm = ThresholdOperations.normalize_data(img_slice)
        return (norm > threshold) & (mask_slice > 0)


    @staticmethod
    def adjust_threshold(image_slice, mask_slice, target_area, slice_index):
        threshold = 0.8
        step = 0.01
        best_threshold = threshold
        best_diff = float('inf')

        while threshold >= 0:
            thresholded_image = ThresholdOperations.threshold_image(image_slice, mask_slice, threshold)
            area = MaskOperations.measure_mask_area(thresholded_image)
            diff = abs(area - target_area)

            if diff < best_diff:
                best_diff = diff
                best_threshold = threshold

            threshold -= step

        return best_threshold, ThresholdOperations.threshold_image(image_slice, mask_slice, best_threshold)