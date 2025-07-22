import nibabel as nib
import numpy as np
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
        print(pts)
        cv2.fillPoly(mask, pts, 255)

        # Use only one channel for bitwise operations 
        result = cv2.bitwise_and(image, image, mask=mask)
        return result, mask

    @staticmethod
    def save_mask(mask, file_path):
        cv2.imwrite(file_path, mask)
        return file_path

    #NOT FINISHED!
    def save_nii_mask(mask, file_path):
        mask_nii = nib.Nifti1Image(mask, np.eye(4))
        nib.save(mask_nii, file_path)
        return file_path
    
if __name__ == "__main__":
    # Example usage
    file_path = "media/21991_40.nii"
    image = ImageLoader.load_image(file_path)
    points = [(491, 223), (393, 363), (582, 369)]
    mask = MaskOperations.create_mask(image, points, reduction_scale=0.3256)
    #MaskOperations.save_mask(mask, "path/to/save/mask.png")
    print("Mask created and saved successfully.")   