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
        # Garante que as dimensões estejam no formato esperado (slice, Y, X)
        # Se o primeiro eixo for maior que o último, provavelmente precisa transpor
        if data.shape[0] > data.shape[-1]:
            return np.transpose(data, (2, 0, 1))
        return data

    @staticmethod
    def load_nii_central_slice_lazy(file_path, dtype=np.float32):
        img = nib.load(file_path)
        affine = img.affine
        data = img.dataobj  # acesso preguiçoso
        nb_of_slices = data.shape[0]
        idx = nb_of_slices // 2
        slice_ = np.asarray(data[idx], dtype=dtype)
        return slice_, affine, nb_of_slices
    
    @staticmethod
    def load_nii_central_slice(file_path, dtype=np.float32):
        nii = nib.load(file_path).get_fdata()  # shape (Z, Y, X)
        nii = ImageOperations.rearrange_dimensions(nii)  # aplica rearranjo para consistência
        idx = nii.shape[0] // 2  # slice central
        return nii[idx].astype(dtype)  # retorna slice 2D (Y, X)

    @staticmethod
    def normalize_image(img):
        img = img - np.min(img)
        img = img / (np.max(img) + 1e-8)
        return (img * 255).astype(np.uint8)

    @staticmethod
    def load_image(file_path):
        slice_image, affine, nb_of_slices = ImageOperations.load_nii_central_slice_lazy(file_path)
        slice_image = np.rot90(slice_image)  # rotaciona para visualização consistente
        norm_image = ImageOperations.normalize_image(slice_image)
        return cv2.cvtColor(norm_image, cv2.COLOR_GRAY2RGB), affine, nb_of_slices

    @staticmethod
    def load_nii_slice(file_path, slice_index, dtype=np.float32):
        nii_obj = nib.load(file_path)
        nii_data = nii_obj.get_fdata()
        
        # Aplica rearranjo para garantir que o shape seja (Z, Y, X)
        #nii_data = ImageOperations.rearrange_dimensions(nii_data)
        
        slice_data = nii_data[slice_index, :, :]
        
        # DEBUG - Descomente para ver shapes carregados
        # print(f"[DEBUG] load_nii_slice - file: {os.path.basename(file_path)}, slice_index: {slice_index}, shape: {slice_data.shape}")
        
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
        scale = 1 / reduction_scale
        scaled_points = [(int(x * scale), int(y * scale)) for (x, y) in points]
        mask = np.zeros_like(image[:,:,0], dtype=np.uint8)
        pts = np.array([scaled_points], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)
        result = cv2.bitwise_and(image, image, mask=mask)
        return result, mask

    @staticmethod
    def create_mask_nifti(folder_path, original_affine):
        images = [np.array(Image.open(os.path.join(folder_path, f)).convert('L')) 
                for f in sorted(os.listdir(folder_path), 
                key=lambda x: int(x.split('_')[1])) if f.endswith('.png')]
        volume = np.stack(images, axis=0)  # shape (Z, Y, X)
        # Salva diretamente, sem transpor nem flipar
        nib.save(nib.Nifti1Image(volume, original_affine), 
                os.path.join(folder_path, 'mask.nii'))
        return os.path.join(folder_path, 'mask.nii')

    @staticmethod
    def measure_mask_area(mask):
        return np.count_nonzero(mask)

    @staticmethod
    def save_png_mask(mask, file_path, file_name="dense.png"):
        os.makedirs(file_path, exist_ok=True)
        cv2.imwrite(os.path.join(file_path, file_name), mask)

    @staticmethod
    def save_nii_mask(mask, affine, nb_of_slices, file_path, file_name="dense.nii"):
        # Garante máscara binária 2D
        mask_2d = np.where(mask > 0, 1, 0).astype(np.uint8)
        # Cria volume 3D com a máscara na fatia central, sem transposição
        mask_3d = np.zeros((nb_of_slices, *mask_2d.shape), dtype=np.uint8)
        center_slice = nb_of_slices // 2
        mask_3d[center_slice] = mask_2d
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


class ThresholdOperations:
    def __init__(self):
        pass

    @staticmethod
    def normalize_data(data):
        mn, mx = data.min(), data.max()
        return (data - mn) / (mx - mn + 1e-8)
    
    @staticmethod
    def apply_threshold(image, mask, threshold):
        norm_image = ImageOperations.normalize_image(image)
        return (norm_image > threshold * 255) & (mask > 0)

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
