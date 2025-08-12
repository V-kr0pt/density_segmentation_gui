"""
SAM2 Inference - Box prompt inference on central slice
"""
import streamlit as st
import numpy as np
import nibabel as nib
import time
from sam_utils import convert_nii_slice_for_sam2
import matplotlib.pyplot as plt

def sam_inference_step():
    """Inferência SAM2 com bounding box no slice central"""
    st.header("🎯 SAM2 Inference")
    
    if "sam_bounding_box" not in st.session_state:
        st.error("No bounding box found. Please complete threshold analysis first.")
        return
    
    sam_manager = st.session_state["sam2_manager"]
    
    if not sam_manager.model_loaded:
        st.error("SAM2 model not loaded.")
        return
    
    # Informações da bounding box
    bbox = st.session_state["sam_bounding_box"]
    st.success(f"✅ Using Bounding Box: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})")
    
    # Carregar dados do volume completo para preparar propagação
    original_image_path = st.session_state["original_image_path"]
    nii_img = nib.load(original_image_path)
    nii_data = nii_img.get_fdata()
    
    st.subheader("📊 Volume Information")
    st.write(f"**Shape:** {nii_data.shape}")
    st.write(f"**Total slices:** {nii_data.shape[0]}")
    st.write(f"**Central slice:** {nii_data.shape[0] // 2}")
    
    # Executar inferência automaticamente
    with st.spinner("Running SAM2 inference on central slice..."):
        
        # Preparar imagem central (mesmo slice usado no threshold)
        central_slice_idx = nii_data.shape[0] // 2
        central_slice = nii_data[central_slice_idx, :, :]
        central_slice = np.rot90(central_slice)  # Mesma rotação dos outros steps
        sam_image = convert_nii_slice_for_sam2(central_slice)
        
        # Garantir que a imagem está definida no SAM2
        success, message = sam_manager.set_image(sam_image)
        if not success:
            st.error("❌ " + message)
            return
        
        # Converter bounding box para formato do SAM2
        input_box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
        
        # Executar predição inicial
        masks, scores, message = sam_manager.predict(input_boxes=input_box[None, :])
        
        if masks is not None:
            st.success("✅ SAM2 inference completed successfully!")
            
            # Selecionar melhor máscara
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            best_score = scores[best_mask_idx]
            
            # Salvar resultado inicial
            st.session_state["sam_initial_mask"] = best_mask
            st.session_state["sam_central_slice_idx"] = central_slice_idx
            st.session_state["sam_inference_complete"] = True
            
            st.info(f"🎯 Best mask selected with score: {best_score:.3f}")
            
            # Mostrar resultado
            st.subheader("🎭 SAM2 Inference Result")
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Imagem original
            axes[0].imshow(sam_image)
            axes[0].set_title(f"Central Slice {central_slice_idx}")
            axes[0].axis('off')
            
            # Bounding box prompt
            axes[1].imshow(sam_image)
            rect = plt.Rectangle(
                (bbox[0], bbox[1]), 
                bbox[2]-bbox[0], 
                bbox[3]-bbox[1], 
                linewidth=3, 
                edgecolor='red', 
                facecolor='none'
            )
            axes[1].add_patch(rect)
            axes[1].set_title("Box Prompt")
            axes[1].axis('off')
            
            # Resultado da segmentação
            axes[2].imshow(sam_image)
            axes[2].imshow(best_mask, alpha=0.6, cmap='viridis')
            axes[2].set_title(f"SAM2 Result (Score: {best_score:.3f})")
            axes[2].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Estatísticas da máscara
            mask_area = np.sum(best_mask)
            total_pixels = best_mask.size
            coverage = (mask_area / total_pixels) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mask Area", f"{mask_area:,} pixels")
            with col2:
                st.metric("Coverage", f"{coverage:.1f}%")
            with col3:
                st.metric("Confidence", f"{best_score:.3f}")
            
            # Proceed automatically to propagation
            st.success("🚀 Ready for video-like propagation!")
            st.info("The SAM2 model will now propagate this segmentation through all slices...")
            
            # Small delay and proceed
            time.sleep(2)
            
            st.session_state["current_step"] = "sam_propagation"
            st.rerun()
            
        else:
            st.error("❌ SAM2 inference failed: " + message)
            
            # Opção para voltar
            if st.button("↩️ Back to Threshold Analysis"):
                st.session_state["current_step"] = "sam_threshold_auto"
                st.rerun()
    
    # Botões de navegação (caso algo dê errado)
    if st.session_state.get("sam_inference_complete", False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Re-run Inference"):
                if "sam_inference_complete" in st.session_state:
                    del st.session_state["sam_inference_complete"]
                st.rerun()
        
        with col2:
            if st.button("🌊 Go to Propagation"):
                st.session_state["current_step"] = "sam_propagation"
                st.rerun()
