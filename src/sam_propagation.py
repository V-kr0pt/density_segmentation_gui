"""
SAM2 Propagation - Video-like propagation through all slices
"""
import streamlit as st
import numpy as np
import nibabel as nib
import os
import cv2
from PIL import Image
import tempfile
import shutil
from sam_utils import convert_nii_slice_for_sam2
import matplotlib.pyplot as plt

def sam_propagation_step():
    """Propagação SAM2 através de todos os slices (como frames de vídeo)"""
    st.header("🌊 SAM2 Video-like Propagation")
    
    if not st.session_state.get("sam_inference_complete", False):
        st.error("SAM2 inference not completed. Please complete inference first.")
        return
    
    sam_manager = st.session_state["sam2_manager"]
    
    if not sam_manager.model_loaded:
        st.error("SAM2 model not loaded.")
        return
    
    # Informações do estado atual
    bbox = st.session_state["sam_bounding_box"]
    initial_mask = st.session_state["sam_initial_mask"]
    central_slice_idx = st.session_state["sam_central_slice_idx"]
    
    st.success("✅ Ready for propagation!")
    st.info(f"Starting from slice {central_slice_idx} with bounding box: {bbox}")
    
    # Carregar dados completos
    original_image_path = st.session_state["original_image_path"]
    nii_img = nib.load(original_image_path)
    nii_data = nii_img.get_fdata()
    
    st.subheader("📊 Propagation Setup")
    st.write(f"**Total slices to process:** {nii_data.shape[0]}")
    st.write(f"**Starting slice:** {central_slice_idx}")
    
    # Executar propagação automaticamente
    if st.button("🚀 Start Automatic Propagation", type="primary") or st.session_state.get("auto_start_propagation", False):
        
        st.session_state["auto_start_propagation"] = True
        
        with st.spinner("Preparing slices for video-like processing..."):
            
            # Criar diretório temporário para os slices
            temp_dir = tempfile.mkdtemp()
            slice_paths = []
            
            try:
                # Salvar todos os slices como imagens PNG
                progress_bar = st.progress(0)
                st.write("📁 Converting slices to images...")
                
                for i in range(nii_data.shape[0]):
                    # Processar slice
                    slice_data = nii_data[i, :, :]
                    slice_data = np.rot90(slice_data)  # Mesma rotação
                    
                    # Converter para imagem SAM2
                    sam_image = convert_nii_slice_for_sam2(slice_data)
                    
                    # Salvar como PNG
                    slice_filename = f"slice_{i:03d}.png"
                    slice_path = os.path.join(temp_dir, slice_filename)
                    
                    # Converter para PIL e salvar
                    pil_image = Image.fromarray(sam_image)
                    pil_image.save(slice_path)
                    slice_paths.append(slice_path)
                    
                    # Atualizar progresso
                    progress_bar.progress((i + 1) / nii_data.shape[0])
                
                st.success(f"✅ All {len(slice_paths)} slices prepared!")
                
                # Inicializar estado do SAM2 para propagação
                st.write("🧠 Initializing SAM2 video state...")
                
                # Usar a nova API do SAM2 para propagação em vídeo
                try:
                    # Inicializar estado como se fosse um vídeo
                    inference_state = sam_manager.predictor.init_state(image_paths=slice_paths)
                    sam_manager.predictor.reset_state(inference_state)
                    
                    st.success("✅ SAM2 video state initialized!")
                    
                    # Adicionar bounding box no slice central
                    st.write(f"📍 Adding box prompt on slice {central_slice_idx}...")
                    
                    box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]], dtype=np.float32)
                    obj_id = 1  # ID do objeto
                    
                    _, out_obj_ids, out_mask_logits = sam_manager.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=central_slice_idx,
                        obj_id=obj_id,
                        box=box,
                    )
                    
                    st.success(f"✅ Box prompt added! Object IDs: {out_obj_ids}")
                    
                    # Executar propagação
                    st.write("🌊 Running propagation through all slices...")
                    progress_bar = st.progress(0)
                    
                    slice_segments = {}
                    processed_count = 0
                    
                    for out_frame_idx, out_obj_ids, out_mask_logits in sam_manager.predictor.propagate_in_video(inference_state):
                        # Processar máscaras
                        slice_segments[out_frame_idx] = {}
                        
                        for i, out_obj_id in enumerate(out_obj_ids):
                            mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                            slice_segments[out_frame_idx][out_obj_id] = mask
                        
                        processed_count += 1
                        progress_bar.progress(processed_count / len(slice_paths))
                    
                    st.success(f"✅ Propagation completed! Processed {len(slice_segments)} slices.")
                    
                    # Salvar resultados
                    st.session_state["sam_propagation_results"] = slice_segments
                    st.session_state["sam_slice_paths"] = slice_paths
                    st.session_state["sam_temp_dir"] = temp_dir
                    st.session_state["propagation_complete"] = True
                    
                    # Mostrar estatísticas
                    st.subheader("📊 Propagation Statistics")
                    
                    total_slices = len(slice_segments)
                    avg_coverage = []
                    
                    for frame_idx in sorted(slice_segments.keys()):
                        for obj_id, mask in slice_segments[frame_idx].items():
                            coverage = (np.sum(mask) / mask.size) * 100
                            avg_coverage.append(coverage)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Slices Processed", total_slices)
                    with col2:
                        st.metric("Avg Coverage", f"{np.mean(avg_coverage):.1f}%")
                    with col3:
                        st.metric("Objects Tracked", len(out_obj_ids))
                    
                    # Visualizar alguns resultados
                    st.subheader("🎭 Sample Results")
                    
                    # Mostrar resultados de algumas fatias
                    sample_indices = [0, len(slice_segments)//4, len(slice_segments)//2, 3*len(slice_segments)//4, len(slice_segments)-1]
                    sample_indices = [idx for idx in sample_indices if idx in slice_segments]
                    
                    if len(sample_indices) >= 3:
                        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                        axes = axes.flatten()
                        
                        for i, frame_idx in enumerate(sample_indices[:6]):
                            if i < len(axes):
                                # Carregar imagem
                                img = Image.open(slice_paths[frame_idx])
                                axes[i].imshow(img)
                                
                                # Sobrepor máscara
                                if frame_idx in slice_segments:
                                    for obj_id, mask in slice_segments[frame_idx].items():
                                        axes[i].imshow(mask, alpha=0.5, cmap='viridis')
                                
                                axes[i].set_title(f"Slice {frame_idx}")
                                axes[i].axis('off')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    
                    # Opções de salvamento
                    st.subheader("💾 Save Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💾 Save All Masks"):
                            save_masks_to_nii(slice_segments, nii_img, nii_data.shape)
                            st.success("✅ Masks saved as NII files!")
                    
                    with col2:
                        if st.button("📊 Generate Report"):
                            generate_propagation_report(slice_segments, slice_paths)
                            st.success("✅ Report generated!")
                    
                    st.balloons()
                    st.success("🎉 SAM2 propagation completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error during SAM2 propagation: {str(e)}")
                    st.write("This might be due to SAM2 version compatibility.")
                    st.write("Expected SAM2 API: predictor.init_state(image_paths=...)")
                
            except Exception as e:
                st.error(f"❌ Error preparing slices: {str(e)}")
            
            finally:
                # Não limpar ainda - vamos manter para visualização
                pass
    
    # Mostrar resultados se já processados
    if st.session_state.get("propagation_complete", False):
        st.success("✅ Propagation results available!")
        
        if st.button("🔍 View Detailed Results"):
            show_detailed_results()
        
        if st.button("🏠 Complete Processing"):
            # Limpar arquivos temporários
            cleanup_temp_files()
            st.session_state.clear()
            st.session_state["current_step"] = "mode_selection"
            st.success("✅ Processing completed! Returning to main menu.")
            st.rerun()

def save_masks_to_nii(slice_segments, original_nii, original_shape):
    """Salva as máscaras como arquivos NII"""
    try:
        # Criar volume 3D das máscaras
        mask_volume = np.zeros(original_shape, dtype=np.uint8)
        
        for frame_idx in sorted(slice_segments.keys()):
            for obj_id, mask in slice_segments[frame_idx].items():
                # Converter máscara de volta para orientação original
                mask_rotated = np.rot90(mask, -1)  # Rotação inversa
                mask_volume[frame_idx, :, :] = mask_rotated.astype(np.uint8) * 255
        
        # Criar imagem NII
        mask_nii = nib.Nifti1Image(mask_volume, original_nii.affine, original_nii.header)
        
        # Create output directory based on uploaded file name
        file_name = st.session_state.get("uploaded_file_name", "output")
        base_name = os.path.splitext(file_name.replace('.nii.gz', ''))[0]  # Remove .nii or .nii.gz
        output_dir = os.path.join("output", f"sam2_{base_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        mask_path = os.path.join(output_dir, "sam2_propagated_mask.nii")
        nib.save(mask_nii, mask_path)
        
        st.success(f"✅ Mask saved to: {mask_path}")
        
    except Exception as e:
        st.error(f"❌ Error saving masks: {str(e)}")

def generate_propagation_report(slice_segments, slice_paths):
    """Gera relatório da propagação"""
    try:
        report = []
        report.append("# SAM2 Propagation Report\n")
        report.append(f"Total slices processed: {len(slice_segments)}\n")
        report.append(f"Total images: {len(slice_paths)}\n\n")
        
        report.append("## Coverage by Slice\n")
        for frame_idx in sorted(slice_segments.keys()):
            for obj_id, mask in slice_segments[frame_idx].items():
                coverage = (np.sum(mask) / mask.size) * 100
                report.append(f"Slice {frame_idx}: {coverage:.1f}% coverage\n")
        
        # Save report
        file_name = st.session_state.get("uploaded_file_name", "output")
        base_name = os.path.splitext(file_name.replace('.nii.gz', ''))[0]  # Remove .nii or .nii.gz
        output_dir = os.path.join("output", f"sam2_{base_name}")
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "sam2_propagation_report.txt")
        
        with open(report_path, 'w') as f:
            f.writelines(report)
        
        st.success(f"✅ Report saved to: {report_path}")
        
    except Exception as e:
        st.error(f"❌ Error generating report: {str(e)}")

def show_detailed_results():
    """Mostra resultados detalhados"""
    if "sam_propagation_results" not in st.session_state:
        st.error("No propagation results found.")
        return
    
    slice_segments = st.session_state["sam_propagation_results"]
    slice_paths = st.session_state["sam_slice_paths"]
    
    st.subheader("🔍 Detailed Propagation Results")
    
    # Seletor de slice
    available_slices = sorted(slice_segments.keys())
    selected_slice = st.selectbox("Select slice to view:", available_slices)
    
    if selected_slice in slice_segments:
        # Carregar e mostrar imagem
        img = Image.open(slice_paths[selected_slice])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Imagem original
        axes[0].imshow(img)
        axes[0].set_title(f"Original Slice {selected_slice}")
        axes[0].axis('off')
        
        # Imagem com máscara
        axes[1].imshow(img)
        for obj_id, mask in slice_segments[selected_slice].items():
            axes[1].imshow(mask, alpha=0.6, cmap='viridis')
        axes[1].set_title(f"Slice {selected_slice} with SAM2 Mask")
        axes[1].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Estatísticas
        for obj_id, mask in slice_segments[selected_slice].items():
            coverage = (np.sum(mask) / mask.size) * 100
            st.write(f"**Object {obj_id}:** {coverage:.1f}% coverage, {np.sum(mask):,} pixels")

def cleanup_temp_files():
    """Limpa arquivos temporários"""
    if "sam_temp_dir" in st.session_state:
        try:
            temp_dir = st.session_state["sam_temp_dir"]
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            del st.session_state["sam_temp_dir"]
        except Exception as e:
            st.warning(f"Could not clean temporary files: {str(e)}")
