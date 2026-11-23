cd to the tool folder

conda activate segGUI

streamlit run src/app.py



Click "Start Processing"

Step 1: Select Files
Paste the image folder "xxxxx" to Main Input Folder
Make sure all the files are selected
Click "Start Batch Processing"  ## It may take a whole to load all the images

Step 2: Draw Masks 
1. Left click to put all the points to cover the area where you think has dense tissue, click "Close Polygon" once you are done.
1.1. If you are not satisfied with the polygon you just created --> Click "Delete Last Polygon"
2. Click "Create Mask" --> Scroll down to see Mask Preview --> Click "Save Mask & Continue"

Step 3: Adjust Thresholds
Route 1. Adjust the "Threshold Value", till the mask area seems good to you --> Click "Save & Continue" --> Proceed to next image (Step 2)
Route 2. Adjust the "Threshold Value", if you are not satisfied with the mask area --> Click "Back to Draw Step" --> Click "Redraw Mask" --> Step 2

After all files are finished,  Progress: N/N files completed --> Click "Go to Process Step" --> Click "Start Background Processing (navigate freely while running)" 


To proceed with the images in the next folder, Click "Back to Main Menu" --> Click "Start Processing" --> Proceed to Step 1 and continue as above
