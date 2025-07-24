import nibabel as nib
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
        return data

    @staticmethod
    def load_nii_central_slice_lazy(file_path, dtype=np.float32):
        img = nib.load(file_path)

        affine = img.affine
        data = img.dataobj  # acesso preguiçoso
        data = ImageOperations.rearrange_dimensions(data)
        nb_of_slices = data.shape[0]
        idx = nb_of_slices // 2
        slice_ = np.asarray(data[idx], dtype=dtype)
        return slice_, affine, nb_of_slices

    @staticmethod
    def load_nii_central_slice(file_path, dtype=np.float32):
        nii = nib.load(file_path).get_fdata()
        nii = ImageOperations.rearrange_dimensions(nii)
        idx = nii.shape[0] // 2
        return nii[idx].astype(dtype)

    @staticmethod
    def normalize_image(img):
        img = img - np.min(img)
        img = img / (np.max(img) + 1e-8)
        return (img * 255).astype(np.uint8)


    @staticmethod
    def load_image(file_path):
        slice_image, affine, nb_of_slices = ImageOperations.load_nii_central_slice_lazy(file_path)
        slice_image = np.rot90(slice_image)
        norm_image = ImageOperations.normalize_image(slice_image)
        return cv2.cvtColor(norm_image, cv2.COLOR_GRAY2RGB), affine, nb_of_slices

    @staticmethod
    def load_nii_slice(file_path, slice_index, dtype=np.float32):
        nii_obj = nib.load(file_path)
        nii_data = nii_obj.get_fdata()
        slice_data = nii_data[slice_index, :, :]
        del nii_data, nii_obj
        gc.collect()
        return slice_data.astype(dtype)

    @staticmethod
    def display_thresholded_slice(img, mask, threshold):
        bin_mask = ThresholdOperations.threshold_image(img, mask, threshold)

        rotated_img = np.rot90(img, k=1)
        rotated_mask = np.rot90(bin_mask, k=1)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rotated_img, cmap='gray')
        ax.imshow(rotated_mask, cmap='jet', alpha=0.2)
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
        
        # Ajuste os pontos para o tamanho original da imagem
        scale = 1 / reduction_scale
        scaled_points = [(int(x * scale), int(y * scale)) for (x, y) in points]
        
        mask = np.zeros_like(image[:,:,0], dtype=np.uint8)
        pts = np.array([scaled_points], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)

        # Use only one channel for bitwise operations 
        result = cv2.bitwise_and(image, image, mask=mask)
        return result, mask
    
    #GIVING AN ERROR IN THE END OF THE FILE PROCESSING
    @staticmethod 
    def create_mask_nifti(folder_path, original_affine):
        images = [np.array(Image.open(os.path.join(folder_path, f)).convert('L')) 
                  for f in sorted(os.listdir(folder_path), 
                  key=lambda x: int(x.split('_')[1])) if f.endswith('.png')]
        volume = np.stack(images, axis=0)
        transposed_flipped_volume = np.flip(np.transpose(volume, (0, 2, 1)), axis=2)
        nib.save(nib.Nifti1Image(transposed_flipped_volume, original_affine), 
                 os.path.join(folder_path, 'mask.nii'))
        return os.path.join(folder_path, 'mask.nii')

    #@staticmethod
    #def create_mask_nifti(folder_path, original_affine):
    #    def valid_png(f):
    #        parts = f.split('_')
    #        return f.endswith('.png') and len(parts) > 1 and parts[1].split('.')[0].isdigit()
    #    images = [
    #        np.array(Image.open(os.path.join(folder_path, f)).convert('L'))
    #        for f in sorted(
    #            filter(valid_png, os.listdir(folder_path)),
    #            key=lambda x: int(x.split('_')[1].split('.')[0])
    #        )
    #    ]
    #    volume = np.stack(images, axis=0)
    #    transposed_flipped_volume = np.flip(np.transpose(volume, (0, 2, 1)), axis=2)
    #    nib.save(nib.Nifti1Image(transposed_flipped_volume, original_affine),
    #             os.path.join(folder_path, 'mask.nii'))
    #    return os.path.join(folder_path, 'mask.nii')

    @staticmethod
    def measure_mask_area(mask):
        return np.count_nonzero(mask)

    @staticmethod
    def save_png_mask(mask, file_path, file_name="dense.png"):
        os.makedirs(file_path, exist_ok=True)
        cv2.imwrite(os.path.join(file_path, file_name), mask)

    @staticmethod
    def save_nii_mask(mask, affine, nb_of_slices, file_path, file_name="dense.nii"):
        # transform in a mask of 1's
        mask_3d = np.where(mask > 0, 1, 0).astype(np.uint8)
        mask_3d = np.repeat(mask_3d[np.newaxis, :, :], nb_of_slices, axis=0)
        mask_3d = np.transpose(mask_3d, (0, 2, 1))  # Ensure shape is (Y, X, Z)
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