import nibabel as nib
import numpy as np
import os
import json
import cv2

class ImageLoader:
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
        data = ImageLoader.rearrange_dimensions(data)
        idx = data.shape[0] // 2
        slice_ = np.asarray(data[idx], dtype=dtype)
        return slice_, affine

    @staticmethod
    def normalize_image(img):
        img = img - np.min(img)
        img = img / (np.max(img) + 1e-8)
        return (img * 255).astype(np.uint8)

    @staticmethod
    def load_image(file_path):
        slice_image, affine = ImageLoader.load_nii_central_slice_lazy(file_path)
        slice_image = np.rot90(slice_image)
        norm_image = ImageLoader.normalize_image(slice_image)
        return cv2.cvtColor(norm_image, cv2.COLOR_GRAY2RGB), affine

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

    @staticmethod
    def save_png_mask(mask, file_path, file_name="mask.png"):
        os.makedirs(file_path, exist_ok=True)
        cv2.imwrite(os.path.join(file_path, file_name), mask)

    @staticmethod
    def save_nii_mask(mask, affine, file_path, file_name="mask.nii"):
        
        mask_nii = nib.Nifti1Image(mask, affine)
        nib.save(mask_nii, os.path.join(file_path, file_name))

    @staticmethod
    def save_mask_json(points, scale, file_path):
        os.makedirs(file_path, exist_ok=True)
        json_data = {"points": points,
                     "scale": scale}
        with open(os.path.join(file_path, "mask.json"), 'w') as f:
            json.dump(json_data, f)

    @staticmethod
    def save_mask(mask, affine, file_path, file_name="mask.png", points=None, scale=1.0):
        os.makedirs(file_path, exist_ok=True)
        MaskOperations.save_png_mask(mask, file_path, file_name)
        MaskOperations.save_nii_mask(mask, affine, file_path, file_name.replace('.png', '.nii'))
        assert points is not None, "Points must be provided to save mask JSON."
        MaskOperations.save_mask_json(points, scale, file_path)
    
if __name__ == "__main__":
    # Example usage
    image_name = "21991_40.nii"
    file_path = os.path.join("media", image_name)

    # load the image and the affine transformation
    image, affine = ImageLoader.load_image(file_path)
    
    # Normally the points would be obtained from streamlit canvas
    # Here we use a hardcoded example for demonstration
    points = [(491, 223), (393, 363), (582, 369)]

    # Create the mask and save it
    # The reduction scale is also obtained from the streamlit session
    # It depends from the original image size and the canvas size
    # It was necessary to better presentation of the image in the canvas
    result, mask = MaskOperations.create_mask(image, points, reduction_scale=0.3256)
    output_path = os.path.join(os.getcwd(), 'output', image_name.split('.')[0])
    MaskOperations.save_mask(mask, affine, file_path=output_path, file_name="mask.png", points=points, scale=0.3256)
    print("Mask created and saved successfully.")   