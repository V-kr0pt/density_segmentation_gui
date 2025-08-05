import onnxruntime as ort
from tqdm import tqdm
import SimpleITK as sitk
from scipy.ndimage import zoom
import numpy as np
import json
import os

import matplotlib.pyplot as plt


class Model:
    def __init__(self, model_path):
        self.model_path = model_path
        config_path = os.path.join(os.path.dirname(model_path), 'config.json')
        self.config = json.load(open(config_path, 'r'))
        self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
        self.patch_size = self.config["model_parameters"]["patch_size"]  # exemple: [64, 256, 128] 
        # Defining stride as half of patch_size
        self.stride = [max(1, p // 2) for p in self.patch_size]

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

        # pad to data have at least patch_size
        data = self.pad_to_patch_size(data)
        return data
    
    def pad_to_patch_size(self, data):
        pad_width = []
        for dim, p in zip(data.shape, self.patch_size):
            if dim < p:
                before = (p - dim) // 2
                after = p - dim - before
            else:
                before = 0
                after = 0
            pad_width.append((before, after))
        return np.pad(data, pad_width, mode='constant')

    def predict_patch(self, patch):
        # Patch inference
        patch_input = patch[np.newaxis, np.newaxis, ...]  # [1, 1, pz, py, px]
        input_name = self.session.get_inputs()[0].name
        pred_patch = self.session.run(None, {input_name: patch_input})[0]
        return np.squeeze(pred_patch)
    

    def sliding_window_inference(self, data):
        # Sliding window to handle larger data than patch_size
        num_classes = 2
        output = np.zeros((num_classes, *data.shape), dtype=np.float32)
        count_map = np.zeros(data.shape, dtype=np.float32)
        z_max, y_max, x_max = data.shape
        pz, py, px = self.patch_size
        sz, sy, sx = self.stride

        for z in tqdm(range(0, z_max - pz + 1, sz)):
            for y in range(0, y_max - py + 1, sy):
                for x in range(0, x_max - px + 1, sx):
                    patch = data[z:z+pz, y:y+py, x:x+px]
                    pred_patch = self.predict_patch(patch)
                    output[:, z:z+pz, y:y+py, x:x+px] += pred_patch
                    count_map[z:z+pz, y:y+py, x:x+px] += 1

        count_map[count_map == 0] = 1
        output = output / count_map
        return output

    def predict(self, data):
        data = self.preprocess_data(data)
        # If shape is the same of patch_size, only predict the patch
        if list(data.shape) == self.patch_size:
            return self.predict_patch(data)
        # If bigger than patch_size, use sliding window
        else:
            return self.sliding_window_inference(data)
        
    def postprocess_data(self, pred, threshold=0.5):
        # The class 1 is the pectoral muscle
        pectoral_prob = pred[1]  # shape: (C, Z, Y, X)
        mask = (pectoral_prob > threshold).astype(np.uint8) # see thresholding after
        return mask
    

class ImageLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.real_factor = (1.0, 1.0, 1)  # You can adjust this factor as needed
        self.img = None  # To store the loaded image
        self.img_array = None  # To store the numpy array of the image

    def load(self):
        # Read the .nii image containing the data with SimpleITK:
        self.img = sitk.ReadImage(self.file_path)
        self.img_array = sitk.GetArrayFromImage(self.img)
        # and access the numpy array:
        return self.img_array
    
    def downsample_image(self, factor=10):
        zoom_factors = [1 / factor, 1 / factor, 1]
        downsampled_img = zoom(self.img_array, zoom_factors, order=1)
        self.real_factor = (downsampled_img.shape[0] / self.img_array.shape[0],
                             downsampled_img.shape[1] / self.img_array.shape[1],
                               1)
        
        return downsampled_img
    
    def upsample_image(self, image):
        zoom_factors = [1 / self.real_factor[0], 1 / self.real_factor[1], 1]
        upsampled_img = zoom(image, zoom_factors, order=1)
        return upsampled_img
        
    
    def save_image(self, array, output_path):
        # Load reference image to get spacing, origin, direction
        out_img = sitk.GetImageFromArray(array)
        out_img.SetSpacing(self.img.GetSpacing())
        out_img.SetOrigin(self.img.GetOrigin())
        out_img.SetDirection(self.img.GetDirection())
        sitk.WriteImage(out_img, output_path)


if __name__ == "__main__":
    model_path = "models/3d_fullres/fold_0/checkpoint_final.onnx"
    model = Model(model_path)
    image_path = "media/21993_39.nii" # Example with pectoral
    #image_path = "media/21991_40.nii" # Example without pectoral
    image_loader = ImageLoader(image_path)

    # Preprocess data
    data = image_loader.load()
    print(f"Original data shape: {data.shape}")

    data = image_loader.downsample_image(factor=10)
    print(f"Data shape after downsampling: {data.shape}")
    data = model.preprocess_data(data)
    # Make prediction
    pred = model.predict(data)

    print(f"Prediction shape: {pred.shape}"
          f" | Data shape: {data.shape}")

    # Postprocess prediction
    pred = model.postprocess_data(pred)
    print(f"Postprocessed prediction shape: {pred.shape}")
    pred = image_loader.upsample_image(pred)
    
    # Check if prediction is not all zeros
    if np.any(pred):
        print("Prediction contains non-zero values.")
    else:
        print("Prediction is all zeros, check the model and input data.")
        exit(1)
    
    
    # Save prediction
    #output_path = "output/21991_40_pred.nii"
    #image_loader.save_image(pred, output_path)    
       
    plt.imshow(data[data.shape[0]//2], cmap="gray")
    plt.title("Central slice after normalization")
    plt.show()
    plot_pred = pred*255
    plt.imshow(plot_pred[pred.shape[0]//2], cmap="gray")
    plt.title("Central slice prediction")
    plt.show()

