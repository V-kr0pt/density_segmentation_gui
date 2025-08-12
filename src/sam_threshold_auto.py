"""
SAM2 Automatic Threshold Analysis
Automatically detects high threshold regions and generates bounding box
"""
import streamlit as st
import numpy as np
import cv2
import os
import time
from sam_utils import SAM2Manager, show_sam2_setup_info, convert_nii_slice_for_sam2
from utils import ImageOperations
import matplotlib.pyplot as plt

def sam_threshold_auto_step():
    """Automatic threshold analysis to generate bounding box"""
    st.header("🎯 SAM2 Automatic Threshold Analysis")
    
    if "uploaded_file_path" not in st.session_state:
        st.error("No file found. Please go back to file selection.")
        if st.button("↩️ Go to SAM2 Setup"):
            st.session_state["current_step"] = "sam"
            st.rerun()
        return
    
    # Inicializar o gerenciador SAM2
    if "sam2_manager" not in st.session_state:
        st.session_state["sam2_manager"] = SAM2Manager()
    
    sam_manager = st.session_state["sam2_manager"]
    
    # Verificar setup do SAM2
    if not sam_manager.model_loaded:
        st.subheader("📋 SAM2 Setup Status")
        
        deps_ok, deps_msg = sam_manager.check_dependencies()
        if not deps_ok:
            st.error("❌ " + deps_msg)
            show_sam2_setup_info()
            return
        
        checkpoint_ok, checkpoint_msg = sam_manager.check_checkpoint()
        if not checkpoint_ok:
            st.error("❌ " + checkpoint_msg)
            st.info("💡 Execute o script: `./download_sam2_checkpoint.sh`")
            return
        
        # Carregar automaticamente
        with st.spinner("Loading SAM2 model automatically..."):
            success, message = sam_manager.load_model()
            if success:
                st.success("✅ " + message)
            else:
                st.error("❌ " + message)
                return
    
    # Automatic processing
    st.success("✅ Starting automatic processing...")
    
    # Load the uploaded file directly
    uploaded_file_path = st.session_state["uploaded_file_path"]
    
    try:
        # Load central slice of the uploaded file
        img = ImageOperations.load_nii_central_slice(uploaded_file_path)
        # For SAM2, we don't need a pre-existing mask - we'll generate bounding box from the image
        msk = None  # SAM2 will generate the mask based on the bounding box
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return
    
    with st.spinner("Analyzing threshold and detecting bounding box automatically..."):
        
        # Parâmetros automáticos otimizados
        high_threshold = 0.85  # Threshold alto automático
        edge_threshold1 = 80
        edge_threshold2 = 160
        
        st.info(f"🔧 Using automatic parameters:")
        st.write(f"• High Threshold: {high_threshold}")
        st.write(f"• Edge Detection: Canny({edge_threshold1}, {edge_threshold2})")
        
        # Aplicar threshold alto
        binary_mask = (msk > high_threshold).astype(np.uint8) * 255
        
        # Detectar bordas
        edges = cv2.Canny(binary_mask, edge_threshold1, edge_threshold2)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Encontrar o maior contorno
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calcular bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Expandir a bounding box ligeiramente
            margin = 15
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img.shape[1] - x, w + 2*margin)
            h = min(img.shape[0] - y, h + 2*margin)
            
            # Salvar bounding box
            bbox = [x, y, x+w, y+h]
            st.session_state["sam_bounding_box"] = bbox
            st.session_state["sam_threshold_used"] = high_threshold
            st.session_state["sam_central_slice_img"] = img
            
            st.success(f"✅ Bounding box detected automatically: ({x}, {y}, {x+w}, {y+h})")
            
            # Visualização automática
            st.subheader("📊 Automatic Analysis Results")
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Imagem original
            axes[0,0].imshow(img, cmap='gray')
            axes[0,0].set_title("Original Image")
            axes[0,0].axis('off')
            
            # Máscara com threshold
            axes[0,1].imshow(binary_mask, cmap='gray')
            axes[0,1].set_title(f"Binary Mask (threshold={high_threshold})")
            axes[0,1].axis('off')
            
            # Bordas detectadas
            axes[1,0].imshow(edges, cmap='gray')
            axes[1,0].set_title("Detected Edges")
            axes[1,0].axis('off')
            
            # Resultado final com bounding box
            result_img = img.copy()
            if len(result_img.shape) == 2:
                result_img = cv2.cvtColor((result_img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            else:
                result_img = (result_img * 255).astype(np.uint8)
            
            # Desenhar bounding box
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 0), 3)
            axes[1,1].imshow(result_img)
            axes[1,1].set_title("Auto-Generated Bounding Box")
            axes[1,1].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Preparar imagem para SAM2
            sam_image = convert_nii_slice_for_sam2(img)
            success, message = sam_manager.set_image(sam_image)
            
            if success:
                st.success("✅ Image prepared for SAM2 inference")
                
                # Proceed automatically to inference
                st.info("🚀 Proceeding to SAM2 inference automatically...")
                
                # Small delay to show results
                time.sleep(2)
                
                st.session_state["current_step"] = "sam_inference"
                st.rerun()
            else:
                st.error("❌ " + message)
        
        else:
            st.error("❌ No contours found with automatic parameters.")
            st.info("The mask might be too small or the threshold too high.")
            
            # Tentar com threshold mais baixo
            st.warning("Trying with lower threshold...")
            high_threshold = 0.70
            
            binary_mask = (msk > high_threshold).astype(np.uint8) * 255
            edges = cv2.Canny(binary_mask, edge_threshold1, edge_threshold2)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                st.success(f"✅ Found contours with threshold {high_threshold}")
                
                # Repeat process with lower threshold
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                margin = 15
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(img.shape[1] - x, w + 2*margin)
                h = min(img.shape[0] - y, h + 2*margin)
                
                bbox = [x, y, x+w, y+h]
                st.session_state["sam_bounding_box"] = bbox
                st.session_state["sam_threshold_used"] = high_threshold
                st.session_state["sam_central_slice_img"] = img
                
                # Preparar imagem para SAM2
                sam_image = convert_nii_slice_for_sam2(img)
                success, message = sam_manager.set_image(sam_image)
                
                if success:
                    st.success("✅ Proceeding with lower threshold...")
                    st.session_state["current_step"] = "sam_inference"
                    st.rerun()
            else:
                st.error("❌ Could not find suitable regions even with lower threshold.")
                if st.button("↩️ Back to Draw Step"):
                    st.session_state["current_step"] = "draw"
                    st.rerun()
