import os
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from sklearn.preprocessing import StandardScaler
import nibabel as nib
from torchvision import transforms

# --- IMPORT MODELLI ---
from models.multi_resnet import MultiViewResNet
from models.multi_modal_resnet import MultiModalMultiViewResNet
from models.mlp import MLP_tabular

# --- IMPORT XAI ---
# Importiamo la funzione specifica per l'alta risoluzione
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

def generate_mips_from_nifti(nifti_path):
    """
    Genera le 3 proiezioni MIP (Axial, Sagittal, Coronal).
    Esegue una rotazione di 90 gradi per allinearle visivamente allo standard medico.
    """
    img_obj = nib.load(nifti_path)
    data = img_obj.get_fdata()
    
    # Calcolo MIP (Maximum Intensity Projection)
    # Assumiamo orientamento standard (X, Y, Z)
    mip_ax = np.max(data, axis=2)  # Proiezione Assiale
    mip_sag = np.max(data, axis=0) # Proiezione Sagittale
    mip_cor = np.max(data, axis=1) # Proiezione Coronale

    # Rotazione standard per visualizzazione "a testa in su"
    mip_ax = np.rot90(mip_ax)
    mip_sag = np.rot90(mip_sag)
    mip_cor = np.rot90(mip_cor)

    def to_uint8_rgb(mip):
        # Normalizza tra 0 e 255
        mip = mip - mip.min()
        if mip.max() > 0:
            mip = (mip / mip.max()) * 255
        mip = mip.astype(np.uint8)
        # Converte in RGB (3 canali)
        return np.stack([mip, mip, mip], axis=-1)

    return {
        "axial": to_uint8_rgb(mip_ax),
        "sagittal": to_uint8_rgb(mip_sag),
        "coronal": to_uint8_rgb(mip_cor)
    }

def extract_features_from_nifti(nifti_path, feature_names_ordered):
    print(f"--- Estrazione Feature Morfologiche da: {os.path.basename(nifti_path)} ---")
    
    # Check rapido per evitare il blocco se si passa un'immagine grezza
    img_tmp = nib.load(nifti_path)
    if len(np.unique(img_tmp.get_fdata())) > 50:
        print("!!! ATTENZIONE: Sembra un file Raw MRA (grigio), non una maschera binaria.")
        print("Il calcolo morfologico potrebbe essere lento o errato.")

    results = process(
        nifti_path, 
        selected_metrics=set(ALL_METRIC_KEYS), 
        save_conn_comp_masks=False, 
        save_seg_masks=False
    )
    
    if not results:
        print("ATTENZIONE: Nessun vaso rilevato.")
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

# --------------------------------------------------------------------------
# Logica Principale
# --------------------------------------------------------------------------

def run_single_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo device: {device}")

    # 1. Preparazione Scaler
    if not os.path.exists(args.excel_path):
        raise FileNotFoundError(f"File Excel non trovato: {args.excel_path}")
        
    print("Lettura Excel e fitting scaler...")
    df = pd.read_excel(args.excel_path)
    drop_cols = {"file_sorgente", "label1", "label2"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    tab_cols = [c for c in numeric_cols if c not in drop_cols]
    
    scaler = StandardScaler()
    scaler.fit(df[tab_cols].astype(np.float32).values)
    tabular_dim = len(tab_cols)
    print(f"Modello addestrato su {tabular_dim} features tabulari.")

    # 2. Inizializzazione Modello
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
    else:
        raise ValueError(f"Modello {args.selected_model} non supportato.")

    # 3. Caricamento Pesi
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print("Pesi caricati con successo.")
    else:
        raise FileNotFoundError(f"Checkpoint non trovato: {args.checkpoint_path}")

    model.eval()

    # 4. Elaborazione Input
    print(f"\nAnalisi paziente: {args.nifti_path}")
    inputs = {}

    mips = None
    if args.selected_model in ['multi_CNN', 'multimodal']:
        # Generiamo i MIPs originali (numpy uint8)
        mips = generate_mips_from_nifti(args.nifti_path)
        
        # Creiamo i tensor normalizzati per il modello (resize a 224 per l'inferenza)
        transform = get_val_transform()
        for view in ['axial', 'sagittal', 'coronal']:
            img_pil = Image.fromarray(mips[view])
            inputs[view] = transform(img_pil).unsqueeze(0).to(device)

    if args.selected_model in ['MLP_tabular', 'multimodal']:
        raw_feats = extract_features_from_nifti(args.nifti_path, tab_cols)
        feats_scaled = scaler.transform(raw_feats.reshape(1, -1))
        inputs['tabular'] = torch.tensor(feats_scaled, dtype=torch.float32).to(device)

    # 5. Predizione & GradCAM
    grad_cam = MultiViewGradCAM(model, device)
    
    with torch.enable_grad():
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float:
                v.requires_grad = True
        
        logits = model(inputs)
        prob = torch.sigmoid(logits).item()
        pred = 1 if prob >= 0.5 else 0

        cam_ax, cam_sag, cam_cor = None, None, None
        if args.selected_model in ['multi_CNN', 'multimodal']:
            cam_ax, cam_sag, cam_cor = grad_cam.generate_maps(inputs)

    # 6. Output e Visualizzazione
    print("\n" + "="*40)
    print(f" PREDITIONE: {'CLASSE 1 (Malato)' if pred==1 else 'CLASSE 0 (Sano)'}")
    print(f" PROBABILITÀ: {prob:.4f}")
    print("="*40 + "\n")

    # Salvataggio immagine risultato
    if cam_ax is not None and mips is not None:
        # Overlay heatmap
        viz_ax = overlay_heatmap_original_size(mips['axial'], cam_ax)
        viz_sag = overlay_heatmap_original_size(mips['sagittal'], cam_sag)
        viz_cor = overlay_heatmap_original_size(mips['coronal'], cam_cor)

        # --- LOGICA DI ALLINEAMENTO (MODIFICATA) ---
        # Trova l'altezza massima tra le tre viste
        h_max = max(viz_ax.shape[0], viz_sag.shape[0], viz_cor.shape[0])
        
        def pad_centered(img, target_h):
            """Centra l'immagine verticalmente aggiungendo bordo nero sopra e sotto"""
            h, w, c = img.shape
            diff = target_h - h
            if diff <= 0:
                return img
            
            top = diff // 2
            bottom = diff - top
            
            # Padding: ((top, bottom), (left, right), (channels))
            return np.pad(img, ((top, bottom), (0, 0), (0, 0)), mode='constant', constant_values=0)

        # Applica padding centrato
        final_ax = pad_centered(viz_ax, h_max)
        final_sag = pad_centered(viz_sag, h_max)
        final_cor = pad_centered(viz_cor, h_max)

        # Unione orizzontale
        montage = np.hstack([final_ax, final_sag, final_cor])
        
        # Aggiunta testo
        montage_bgr = cv2.cvtColor(montage, cv2.COLOR_RGB2BGR)
        color = (0, 0, 255) if pred == 1 else (0, 255, 0)
        txt = f"Pred: {pred} | Prob: {prob:.3f}"
        cv2.putText(montage_bgr, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        out_name = f"TestResult_{os.path.basename(args.nifti_path)}.jpg"
        cv2.imwrite(out_name, montage_bgr)
        print(f"Immagine salvata correttamente: {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script di Test per singolo caso")
    parser.add_argument("--nifti_path", type=str, required=True, help="Path della MASCHERA .nii.gz")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path del modello .pth")
    parser.add_argument("--excel_path", type=str, required=True, help="Excel originale")
    
    parser.add_argument("--selected_model", type=str, default="multimodal")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--hidden_dim", type=int, default=256)

    args = parser.parse_args()
    
    run_single_test(args)