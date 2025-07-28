import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import ImageOperations, ThresholdOperations
import cv2
from PIL import Image

def threshold_step():
    st.header("Step 2: Adjust Threshold")
    
    if "output_path" not in st.session_state:
        st.error("No mask found. Please go back to Step 1.")
        return
    
    mask_path = os.path.join(st.session_state["output_path"], 'dense.nii')
    original_image_path = st.session_state["original_image_path"]
    
    try:
        # Carrega a imagem da mesma forma que no draw_step
        img, _, _ = ImageOperations.load_image(original_image_path)  # retorna RGB e normalizado
        
        # Para a máscara, carregue o slice central, sem rotação nem normalização
        msk = ImageOperations.load_nii_central_slice(mask_path, dtype=np.uint8)  # sem alterar shape nem rotacionar
        
        st.write(f"Image shape: {img.shape}")  # deve ser (H, W, 3)
        st.write(f"Mask shape: {msk.shape}")   # deve ser (H, W)
        
        # Mostrar slice central da imagem (convertendo para grayscale)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        st.subheader("Slice central da imagem (grayscale)")
        st.image(img_gray, use_column_width=True, clamp=True)
        
        # Mostrar slice central da máscara
        st.subheader("Slice central da máscara")
        st.image(msk * 255, use_column_width=True, clamp=True, channels="L")  # escala para 0-255
        
        # Verificações adicionais
        if np.count_nonzero(msk) == 0:
            st.warning("⚠️ Atenção: O slice central da máscara está completamente vazio!")
        else:
            st.success(f"Slice central da máscara tem {np.count_nonzero(msk)} pixels ativados.")
        
        if np.count_nonzero(img_gray) == 0:
            st.warning("⚠️ Atenção: O slice central da imagem está completamente vazio!")
        else:
            st.success(f"Slice central da imagem tem valores não nulos.")
        
        # Estatísticas da imagem para debug
        st.write(f"Image grayscale min/max: {img_gray.min()}/{img_gray.max()}")
        st.write(f"Mask pixels > 0: {np.count_nonzero(msk)}")
        
    except Exception as e:
        st.error(f"Error loading images: {str(e)}")
        return
    
    # Slider de threshold
    threshold = st.slider(
        "Select threshold value",
        min_value=0.0,
        max_value=1.0,
        value=0.38,
        step=0.01,
        key="threshold_slider"
    )
    
    # Binariza a máscara com o threshold
    bin_mask = ThresholdOperations.threshold_image(img_gray, msk, threshold)

    # Quantos pixels foram ativados?
    activated_pixels = np.count_nonzero(bin_mask)
    st.write(f"Pixels ativados após threshold: {activated_pixels}")
    if activated_pixels == 0:
        st.warning("Nenhum pixel foi ativado com o threshold selecionado. Tente reduzir o valor.")
    
    # Exibe sobreposição da máscara na imagem
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img_gray, cmap='gray')
    ax.imshow(bin_mask, cmap='jet', alpha=0.3)
    ax.set_title(f"Threshold: {threshold:.2f}")
    ax.axis('off')
    st.pyplot(fig)
    
    # Botão para salvar
    if st.button("💾 Save Thresholded Mask"):
        if activated_pixels == 0:
            st.error("Não é possível salvar máscara vazia. Ajuste o threshold antes.")
        else:
            save_dir = os.path.join(st.session_state["output_path"], 'dense_mask')
            os.makedirs(save_dir, exist_ok=True)

            # Salva máscara como imagem PNG
            mask_uint8 = np.where(bin_mask > 0, 255, 0).astype(np.uint8)
            mask_image = Image.fromarray(mask_uint8, mode='L')
            mask_image.save(os.path.join(save_dir, "central_slice_thresholded.png"))

            # (Opcional) salvar imagem original grayscale para referência
            Image.fromarray(img_gray).save(os.path.join(save_dir, "central_slice_gray.png"))

            # Salva o threshold e avança
            st.session_state["final_threshold"] = threshold
            st.session_state["current_step"] = "process"
            st.success(f"Threshold {threshold:.2f} e máscara do slice central salvos.")
            st.rerun()

    if st.button("↩️ Back to Mask Drawing"):
        st.session_state["current_step"] = "draw"
        st.rerun()
