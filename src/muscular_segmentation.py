import onnxruntime as ort
from tqdm import tqdm
import SimpleITK as sitk
import skimage.measure as measure
from scipy.ndimage import zoom
import cv2
import numpy as np
import json
import os

import matplotlib.pyplot as plt


class Model:
    def __init__(self, model_path):
        self.model_path = model_path
        path_model_0 = os.path.join(model_path, 'fold_0', 'checkpoint_final.onnx')
        path_model_1 = os.path.join(model_path, 'fold_1', 'checkpoint_final.onnx')
        path_model_2 = os.path.join(model_path, 'fold_2', 'checkpoint_final.onnx')
        path_model_3 = os.path.join(model_path, 'fold_3', 'checkpoint_final.onnx')
        path_model_4 = os.path.join(model_path, 'fold_4', 'checkpoint_final.onnx')

        # loading models
        self.session_0 = ort.InferenceSession(path_model_0, providers=["CUDAExecutionProvider"])
        self.session_1 = ort.InferenceSession(path_model_1, providers=["CUDAExecutionProvider"])
        self.session_2 = ort.InferenceSession(path_model_2, providers=["CUDAExecutionProvider"])
        self.session_3 = ort.InferenceSession(path_model_3, providers=["CUDAExecutionProvider"])
        self.session_4 = ort.InferenceSession(path_model_4, providers=["CUDAExecutionProvider"])

        # load model config
        config_path = os.path.join(os.path.dirname(model_path), 'config.json')
        self.config = json.load(open(config_path, 'r'))
        self.patch_size = self.config["model_parameters"]["patch_size"]  # exemple: [64, 256, 128] 
        self.stride = [max(1, p // 2) for p in self.patch_size] # defining stride as half of patch_size
        
        # save pectoral side
        self.pectoral_side = None
        self.extra_factor = 2  # Factor to increase sampling density on the pectoral side

    def check_image_shape(self, data):
        if len(data.shape) != 3:
            raise ValueError(f"Input data must be a 3D array, got shape {data.shape}.")
        assert data.shape[0] - data.shape[1] < 0 and \
              data.shape[0] - data.shape[2] < 0, \
            f"Image loaded is in incorrect shape {data.shape}.\nIt should be in (Z, X, Y) format. where Z is the number of slices"


    def preprocess_data(self, data):
        data = data.astype("float32")
        mask = data > 0

        # Z-score normalization
        if np.any(mask):
            mean = np.mean(data[mask])
            std = np.std(data[mask])
        else:
            mean = np.mean(data)
            std = np.std(data)
        data = (data - mean) / (std + 1e-8)

        self.pectoral_side = self.detect_pectoral_side(data)
        if self.pectoral_side == 'left':
            data = np.flip(data, axis=2)

        data = self.pad_to_patch_size(data)
        return data
    

    def pad_to_patch_size(self, data):
        self.original_shape = data.shape
        self.flags = [0, 0] # upper/lower, left/right padding flags
        middle_slice = data[self.original_shape[0] // 2,:,:]
        # verify if the upper or lower part of the image is background
        if np.sum(middle_slice[0,:]) == 0:
            self.flags[0] = 1  # upper part is background

        # verify if the left or right part of the image is background
        if np.sum(middle_slice[:,0]) == 0:
            self.flags[2] = 1

        pad_width = []
        for axis, (dim, p) in enumerate(zip(data.shape, self.patch_size)):
            if dim < p: 
                pad = p - dim
                
                # Z axis
                if axis == 0:  
                    pad_width.append((0, pad))
                    continue  # default pad at the end
                else:
                    flag = self.flags[axis - 1]  # flags[0] for X, flags[1] for Y
                    # X and Y axis
                    if flag:
                        pad_width.append((0,pad))
                    else:
                        pad_width.append((pad, 0))
            else:
                # no padding needed
                pad_width.append((0, 0))
 
        return np.pad(data, pad_width, mode='minimum')

    def remove_padding(self, data):
        slices = []
        for axis, (dim, p) in enumerate(zip(self.original_shape, self.patch_size)):
            pad = p - dim if dim < p else 0
            if pad > 0:
                # Z axis (sempre pad no final)
                if axis == 0:
                    slices.append(slice(0, dim))
                # X and Y axis
                else:
                    flag = self.flags[axis - 1]  # flags[0] for X, flags[1] for Y
                    if flag:
                        # Pad foi no início
                        slices.append(slice(0, dim))
                    else:
                        # Pad foi no final
                        slices.append(slice(pad, pad + dim))
            else:
                slices.append(slice(0, dim))
                
        return data[tuple(slices)]

    def detect_pectoral_side(self, image, threshold=0.1):
        """
        Detects the side of the pectoral muscle based on the mean intensity of the left and right halves of the image.
        Args:
            image (np.ndarray): The input image.
            threshold (float): The threshold to determine the side.
        Returns:
            str: 'left' if the left side is more intense, 'right' if the right side is more intense, 'unknown' otherwise.
        """
        mid_x = image.shape[2] // 2
        left_half = image[:, :, :mid_x]
        right_half = image[:, :, mid_x:]

        left_mean = np.mean(left_half)
        right_mean = np.mean(right_half)

        if left_mean > right_mean + threshold:
            return 'left'
        elif right_mean > left_mean + threshold:
            return 'right'
        else:
            return 'unknown'


    def predict_patch(self, patch):
        # Patch inference
        patch_input = patch[np.newaxis, np.newaxis, ...]  # [1, 1, pz, py, px]
        input_name = self.session_0.get_inputs()[0].name

        pred_patch_0 = self.session_0.run(None, {input_name: patch_input})[0]
        pred_patch_1 = self.session_1.run(None, {input_name: patch_input})[0]
        pred_patch_2 = self.session_2.run(None, {input_name: patch_input})[0]
        pred_patch_3 = self.session_3.run(None, {input_name: patch_input})[0]
        pred_patch_4 = self.session_4.run(None, {input_name: patch_input})[0]

        pred_patch = np.concat([pred_patch_0, pred_patch_1, pred_patch_2, pred_patch_3, pred_patch_4])
        pred_patch = np.mean(pred_patch, axis=0)
        
        return np.squeeze(pred_patch)
    
    def get_sliding_window_positions(self, image_size, patch_size, stride):
        positions = []
        max_pos = image_size - patch_size
        pos = 0
        while pos < max_pos:
            positions.append(pos)
            pos += stride
        positions.append(max_pos)  # Garante que a borda final seja coberta
        return positions

    def sliding_window_inference(self, data):
        # Sliding window to handle larger data than patch_size
        num_classes = 2
        output = np.zeros((num_classes, *data.shape), dtype=np.float32)
        count_map = np.zeros(data.shape, dtype=np.float32)
        z_max, y_max, x_max = data.shape
        pz, py, px = self.patch_size
        sz, sy, sx = self.stride

        z_starts = self.get_sliding_window_positions(z_max, pz, sz)
        y_starts = self.get_sliding_window_positions(y_max, py, sy)
        x_starts = self.get_sliding_window_positions(x_max, px, sx)

        for z in tqdm(z_starts):
            for y in y_starts:
                for x in x_starts:
                    patch = data[z:z+pz, y:y+py, x:x+px]
                    pred_patch = self.predict_patch(patch)
                    output[:, z:z+pz, y:y+py, x:x+px] += pred_patch
                    count_map[z:z+pz, y:y+py, x:x+px] += 1

        count_map[count_map == 0] = 1
        output = output / count_map

        plt.imshow(count_map[count_map.shape[0]//2], cmap="hot")
        plt.title("Central slice count_map")
        plt.savefig("central_slice_count_map.png")
        return output
    
    def keep_largest_component(self, mask):
        '''
        Keep the largest connected component in a binary mask.
        Args:
            mask (np.ndarray): The input binary mask.
        Returns:
            np.ndarray: The binary mask with only the largest connected component retained.
        '''

        labels = measure.label(mask)
        props = measure.regionprops(labels)
        if not props:
            return mask
        largest = max(props, key=lambda x: x.area)
        return (labels == largest.label).astype(np.uint8)
    
    def morphological_closing(self, mask, kernel_size=15):
        '''
        Apply morphological closing to a binary mask.
        Args:
            mask (np.ndarray): The input binary mask.
            kernel_size (int): The size of the structuring element for morphological operations.
        Returns:
            np.ndarray: The binary mask after applying morphological closing.
        '''

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return closed.astype(np.uint8)
    
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
        
    def postprocess_data(self, pred, threshold=0.5):
        # The class 0 is the background
        # The class 1 is the pectoral muscle
        if np.any(pred):
            pectoral_prob = self.sigmoid(pred[1])  # shape: (C, Z, Y, X)
            plt.imshow(pectoral_prob.sum(axis=0), cmap="hot")
            plt.title("Central slice pectoral_prob")
            plt.savefig("central_slice_pectoral_prob.png")
            mask = (pectoral_prob > threshold).astype(np.uint8) # see thresholding after
            mask = self.keep_largest_component(mask)
        mask = self.remove_padding(mask)
        mask = self.morphological_closing(mask)
        if self.pectoral_side == 'left':
            mask = np.flip(mask, axis=2)

        return mask
    

    def predict(self, data):
        # check if data has the correct shape
        self.check_image_shape(data)
        # Preprocess data
        data = self.preprocess_data(data)

        # If shape is the same of patch_size, only predict the patch
        if list(data.shape) == self.patch_size:
            return self.predict_patch(data)
        # If bigger than patch_size, use sliding window
        else:
            return self.sliding_window_inference(data)
    
    

class ImageLoader:
    def __init__(self, file_path):
        self.file_path = file_path 
        self.real_factor = (None, None, None)  # To store the real downsampling factor and use it for upsampling
        self.target_spacing = (1.0, 1.0, 1.0)
        self.img = None  # To store the loaded image with header info
        self.img_array = None  # To store the numpy array of the image

    def load(self, spacing_resample=True):
        # Read the .nii image containing the data with SimpleITK:
        self.img = sitk.ReadImage(self.file_path)
        # Resampling the space
        if spacing_resample:
            self.resample_to_spacing() 
        # and access the numpy array:
        self.img_array = sitk.GetArrayFromImage(self.img).transpose(2, 0, 1) # Transpose to (Z, X, Y) format
        self.img_array = np.flip(self.img_array, axis=2)
        return self.img_array
    

    def resample_to_spacing(self):
        original_spacing = self.img.GetSpacing()
        original_size = self.img.GetSize()
        new_size = [
            int(round(osz * ospc / tspc))
            for osz, ospc, tspc in zip(original_size, original_spacing, self.target_spacing)
        ]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(self.target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(self.img.GetDirection())
        resampler.SetOutputOrigin(self.img.GetOrigin())
        resampler.SetInterpolator(sitk.sitkBSpline)
        self.img = resampler.Execute(self.img)
    
    def downsample_image(self, factor=10):
        zoom_factors = [1, 1 / factor, 1 / factor]
        downsampled_img = zoom(self.img_array, zoom_factors, order=1)
        
        # store the real downsampling factor for later upsampling
        self.real_factor = (1,
                            self.img_array.shape[1] / downsampled_img.shape[1],
                            self.img_array.shape[2] / downsampled_img.shape[2])
        
        return downsampled_img
    
    def upsample_image(self, image):
        zoom_factors = [1, self.real_factor[1], self.real_factor[2]]
        upsampled_img = zoom(image, zoom_factors, order=1)
        return upsampled_img
    
    def save_image(self, array, output_path):
        out_img = sitk.GetImageFromArray(array)

        # Load reference image to get spacing, origin, direction
        out_img.SetSpacing(self.img.GetSpacing())
        out_img.SetOrigin(self.img.GetOrigin())
        out_img.SetDirection(self.img.GetDirection())
        sitk.WriteImage(out_img, output_path)


if __name__ == "__main__":
    model_path = "models/3d_fullres/"
    model = Model(model_path)
    #image_path = "media/21993_39.nii" # Example with pectoral
    image_path = "media/21992_35.nii"
    #image_path = "media/21991_40.nii" # Example without pectoral
    image_loader = ImageLoader(image_path)

    # Load
    original_image = image_loader.load()
    print(f"Original data shape: {original_image.shape}")

    # Downsample 
    downsampled_image = image_loader.downsample_image(factor=10)
    print(f"Data shape after downsampling: {downsampled_image.shape}")

    # Make prediction
    pred = model.predict(downsampled_image)
    print(f"Prediction shape: {pred.shape}")

    # Postprocess prediction
    pred = model.postprocess_data(pred)
    print(f"Postprocessed prediction shape: {pred.shape}")

    # Upsample
    pred = image_loader.upsample_image(pred)
    print(f"upsampled prediction shape: {pred.shape}") 
    
    # Save prediction
    #output_path = "output/21991_40_pred.nii"
    #image_loader.save_image(pred, output_path)
    # 

    central_idx = original_image.shape[0] // 2
    central_slice = original_image[central_idx]
    central_pred = pred[central_idx]

    plt.figure(figsize=(6,6))
    plt.imshow(central_slice, cmap="gray")
    # Sobrepõe a máscara em tons de roxo (alpha controla a transparência)
    plt.imshow(central_pred, cmap="Purples", alpha=0.4)
    plt.title("Overlay: Central slice + Prediction")
    plt.axis('off')
    plt.savefig("central_slice_overlay.png")
    plt.close()    
       
    plt.imshow(central_slice, cmap="gray")
    plt.title("Central slice after normalization")
    plt.savefig("central_slice.png")
    #plt.show()
    plot_pred = pred*255
    plt.imshow(central_pred, cmap="gray")
    plt.title("Central slice prediction")
    plt.savefig("central_slice_prediction.png")
    #plt.show()

