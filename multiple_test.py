import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from sklearn.preprocessing import StandardScaler
import nibabel as nib
from torchvision import transforms
from tqdm import tqdm  # Barra di progresso (opzionale, ma consigliata)

# --- IMPORT MODELLI ---
from models.multi_resnet import MultiViewResNet
from models.multi_modal_resnet import MultiModalMultiViewResNet
from models.mlp import MLP_tabular

# --- IMPORT XAI ---
from utils_xai import MultiViewGradCAM, overlay_heatmap_original_size

# --- IMPORT VESSEL METRICS ---
try:
    import VESSEL_METRICS
    from VESSEL_METRICS import process, get_component_rows_from_results, _aggregate, ALL_METRIC_KEYS
except ImportError:
    print("ERRORE CRITICO: 'VESSEL_METRICS.py' non trovato.")
    exit(1)

# Costanti ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --------------------------------------------------------------------------
# Funzioni di Supporto
# --------------------------------------------------------------------------

def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def ensure_portrait(img_array):
    """Ruota l'immagine se è in landscape."""
    h, w = img_array.shape[:2]
    if w > h:
        return np.rot90(img_array)
    return img_array

def generate_mips_from_nifti(nifti_path):
    """Genera le 3 proiezioni MIP allineate verticalmente."""
    try:
        img_obj = nib.load(nifti_path)
        data = img_obj.get_fdata()
    except Exception as e:
        print(f"Errore lettura NIfTI {os.path.basename(nifti_path)}: {e}")
        return None
    
    mip_ax = np.max(data, axis=2)
    mip_sag = np.max(data, axis=0)
    mip_cor = np.max(data, axis=1)

    # Rotazioni standard
    mip_ax = np.rot90(mip_ax)
    mip_sag = ensure_portrait(mip_sag)
    mip_cor = ensure_portrait(mip_cor)

    def to_uint8_rgb(mip):
        mip = mip - mip.min()
        if mip.max() > 0:
            mip = (mip / mip.max()) * 255
        mip = mip.astype(np.uint8)
        return np.stack([mip, mip, mip], axis=-1)

    return {
        "axial": to_uint8_rgb(mip_ax),
        "sagittal": to_uint8_rgb(mip_sag),
        "coronal": to_uint8_rgb(mip_cor)
    }

def extract_features_from_nifti(nifti_path, feature_names_ordered):
    """Estrae feature tabulari usando VESSEL_METRICS."""
    try:
        # Check rapido anti-raw
        img_tmp = nib.load(nifti_path)
        if len(np.unique(img_tmp.get_fdata())) > 50:
            print(f"SKIP FEATURE: {os.path.basename(nifti_path)} sembra Raw MRA, non maschera.")
            return None

        results = process(
            nifti_path, 
            selected_metrics=set(ALL_METRIC_KEYS), 
            save_conn_comp_masks=False, 
            save_seg_masks=False
        )
        
        if not results:
            return np.zeros(len(feature_names_ordered), dtype=np.float32)

        rows = get_component_rows_from_results(results)
        df_components = pd.DataFrame(rows)
        agg_metrics = _aggregate(df_components)

        feature_vector = []
        for name in feature_names_ordered:
            val = agg_metrics.get(name, 0.0)
            if pd.isna(val): val = 0.0
            feature_vector.append(val)
        
        return np.array(feature_vector, dtype=np.float32)
    except Exception as e:
        print(f"Errore estrazione feature {os.path.basename(nifti_path)}: {e}")
        return None

# --------------------------------------------------------------------------
# Funzione Core per singolo caso
# --------------------------------------------------------------------------

def process_single_case(nifti_path, model, device, scaler, tab_cols, transform, args):
    """Processa un singolo file e restituisce pred, prob e l'immagine visualizzata."""
    
    # 1. Input Immagini
    inputs = {}
    mips = None
    
    if args.selected_model in ['multi_CNN', 'multimodal']:
        mips = generate_mips_from_nifti(nifti_path)
        if mips is None: return None, None, None # Errore caricamento
        
        for view in ['axial', 'sagittal', 'coronal']:
            img_pil = Image.fromarray(mips[view])
            inputs[view] = transform(img_pil).unsqueeze(0).to(device)

    # 2. Input Tabulare
    if args.selected_model in ['MLP_tabular', 'multimodal']:
        raw_feats = extract_features_from_nifti(nifti_path, tab_cols)
        if raw_feats is None: return None, None, None # Errore feature o Raw file
        
        feats_scaled = scaler.transform(raw_feats.reshape(1, -1))
        inputs['tabular'] = torch.tensor(feats_scaled, dtype=torch.float32).to(device)

    # 3. Inferenza & GradCAM
    grad_cam = MultiViewGradCAM(model, device)
    
    cam_ax, cam_sag, cam_cor = None, None, None
    
    with torch.enable_grad():
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float:
                v.requires_grad = True
        
        logits = model(inputs)
        prob = torch.sigmoid(logits).item()
        pred = 1 if prob >= 0.5 else 0

        if args.selected_model in ['multi_CNN', 'multimodal']:
            cam_ax, cam_sag, cam_cor = grad_cam.generate_maps(inputs)

    # 4. Creazione Immagine Finale
    montage_bgr = None
    if cam_ax is not None and mips is not None:
        viz_ax = overlay_heatmap_original_size(mips['axial'], cam_ax)
        viz_sag = overlay_heatmap_original_size(mips['sagittal'], cam_sag)
        viz_cor = overlay_heatmap_original_size(mips['coronal'], cam_cor)

        # Allineamento altezza
        h_max = max(viz_ax.shape[0], viz_sag.shape[0], viz_cor.shape[0])
        
        def resize_h(img, target_h):
            h, w = img.shape[:2]
            if h == target_h: return img
            scale = target_h / float(h)
            new_w = int(w * scale)
            return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)

        final_ax = resize_h(viz_ax, h_max)
        final_sag = resize_h(viz_sag, h_max)
        final_cor = resize_h(viz_cor, h_max)

        # Spaziatore
        spacer = np.zeros((h_max, 30, 3), dtype=np.uint8)
        montage = np.hstack([final_ax, spacer, final_sag, spacer, final_cor])
        
        # Testo
        montage_bgr = cv2.cvtColor(montage, cv2.COLOR_RGB2BGR)
        color = (0, 0, 255) if pred == 1 else (0, 255, 0)
        txt = f"Pred: {pred} | Prob: {prob:.3f}"
        cv2.putText(montage_bgr, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
    return pred, prob, montage_bgr

# --------------------------------------------------------------------------
# Main Batch Loop
# --------------------------------------------------------------------------

def run_batch_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo device: {device}")
    
    # Creazione cartella output
    os.makedirs(args.output_folder, exist_ok=True)
    print(f"Output folder: {args.output_folder}")

    # 1. Setup Scaler (Una volta sola)
    if not os.path.exists(args.excel_path):
        raise FileNotFoundError(f"File Excel training non trovato: {args.excel_path}")
    
    print("Fitting scaler sul dataset di training...")
    df_train = pd.read_excel(args.excel_path)
    drop_cols = {"file_sorgente", "label1", "label2"}
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    tab_cols = [c for c in numeric_cols if c not in drop_cols]
    
    scaler = StandardScaler()
    scaler.fit(df_train[tab_cols].astype(np.float32).values)
    tabular_dim = len(tab_cols)
    print(f"Scaler pronto. Features: {tabular_dim}")

    # 2. Caricamento Modello (Una volta sola)
    print(f"Caricamento modello: {args.selected_model}")
    if args.selected_model == 'multi_CNN':
        model = MultiViewResNet(backbone_name=args.backbone, pretrained=False, hidden_dim=args.hidden_dim).to(device)
    elif args.selected_model == 'MLP_tabular':
        model = MLP_tabular(tabular_dim=tabular_dim, tab_emb_dim=64, tab_hidden=128, hidden_layer=args.hidden_dim).to(device)
    elif args.selected_model == 'multimodal':
        model = MultiModalMultiViewResNet(
            tabular_dim=tabular_dim, backbone_name=args.backbone, pretrained=False, 
            tab_emb_dim=64, tab_hidden=128, fusion_hidden=args.hidden_dim
        ).to(device)
    
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Checkpoint non trovato: {args.checkpoint_path}")
    
    model.eval()
    transform = get_val_transform()

    # 3. Ricerca File
    search_path = os.path.join(args.input_folder, "*.nii.gz")
    files = glob.glob(search_path)
    if not files:
        print(f"Nessun file .nii.gz trovato in {args.input_folder}")
        return

    print(f"Trovati {len(files)} casi da analizzare.")

    # 4. Loop di elaborazione
    results_list = []
    
    # Usa tqdm per progress bar se installato, altrimenti loop normale
    iterator = tqdm(files, desc="Processing") if 'tqdm' in globals() else files

    for nifti_path in iterator:
        filename = os.path.basename(nifti_path)
        
        # Processamento
        pred, prob, img_result = process_single_case(
            nifti_path, model, device, scaler, tab_cols, transform, args
        )
        
        if pred is not None:
            # Salva Risultato nel dizionario per Excel
            results_list.append({
                "Filename": filename,
                "Prediction": int(pred),
                "Probability": float(prob),
                "Label": "Malato" if pred == 1 else "Sano"
            })
            
            # Salva Immagine
            if img_result is not None:
                out_img_name = filename.replace(".nii.gz", "_result.jpg")
                out_img_path = os.path.join(args.output_folder, out_img_name)
                cv2.imwrite(out_img_path, img_result)
        else:
            print(f"Skipped {filename} (Errore)")

    # 5. Salvataggio Excel Finale
    if results_list:
        df_results = pd.DataFrame(results_list)
        # Ordina per nome file
        df_results.sort_values(by="Filename", inplace=True)
        
        out_excel_path = os.path.join(args.output_folder, "Results_Summary.xlsx")
        df_results.to_excel(out_excel_path, index=False)
        print("\n" + "="*50)
        print(f"COMPLETATO.")
        print(f"Excel salvato in: {out_excel_path}")
        print(f"Immagini salvate in: {args.output_folder}")
        print("="*50)
    else:
        print("Nessun risultato valido generato.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Test Script")
    # Argomenti Modificati per Batch
    parser.add_argument("--input_folder", type=str, required=True, help="Cartella contenente i file .nii.gz")
    parser.add_argument("--output_folder", type=str, required=True, help="Cartella dove salvare immagini e excel")
    
    # Argomenti Standard
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path del modello .pth")
    parser.add_argument("--excel_path", type=str, required=True, help="Excel originale per lo scaler")
    parser.add_argument("--selected_model", type=str, default="multimodal")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--hidden_dim", type=int, default=256)

    args = parser.parse_args()
    
    run_batch_test(args)