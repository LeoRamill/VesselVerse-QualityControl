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

# Import dei modelli e delle utility (assicurati che le cartelle 'models' e 'utils_xai' siano presenti)
from models.multi_resnet import MultiViewResNet
from models.multi_modal_resnet import MultiModalMultiViewResNet
from models.mlp import MLP_tabular
from utils_xai import MultiViewGradCAM, overlay_heatmap

# Prova a importare VESSEL_METRICS
try:
    import VESSEL_METRICS
except ImportError:
    print("ATTENZIONE: 'VESSEL_METRICS.py' non trovato. La parte tabulare (lunghezza vasi, ecc.) fallirà.")
    VESSEL_METRICS = None

# --------------------------------------------------------------------------
# Funzioni di Supporto (Duplicate dal main per renderlo standalone)
# --------------------------------------------------------------------------

def get_val_transform():
    """Restituisce solo le trasformazioni di validazione (Resize + Norm)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def generate_mips_from_nifti(nifti_path):
    """Carica NIfTI e crea le 3 proiezioni (Assiale, Sagittale, Coronale)."""
    img_obj = nib.load(nifti_path)
    data = img_obj.get_fdata()
    
    # Proiezioni MIP (Maximum Intensity Projection)
    # Assumiamo orientamento standard (X, Y, Z) -> (Sag, Cor, Ax)
    mip_ax = np.max(data, axis=2)
    mip_sag = np.max(data, axis=0)
    mip_cor = np.max(data, axis=1)

    # Rotazioni standard (possono variare in base allo scanner)
    mip_ax = np.rot90(mip_ax)
    mip_sag = np.rot90(mip_sag)
    mip_cor = np.rot90(mip_cor)

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
    """Calcola le metriche geometriche usando VESSEL_METRICS."""
    if VESSEL_METRICS is None:
        raise ImportError("Modulo VESSEL_METRICS mancante.")

    metrics_to_compute = [
        'total_length', 'num_bifurcations', 'bifurcation_density', 'volume',
        'fractal_dimension', 'lacunarity', 'num_loops', 
        'num_abnormal_degree_nodes', 'mean_loop_length', 'max_loop_length',
        'avg_diameter'
    ]
    
    print(f"Calcolo metriche morfologiche per {os.path.basename(nifti_path)}...")
    results = VESSEL_METRICS.process(
        nifti_path, 
        selected_metrics=set(metrics_to_compute), 
        save_conn_comp_masks=False, 
        save_seg_masks=False
    )
    
    # Aggregazione semplice (somma o media pesata)
    agg_data = {k: 0.0 for k in feature_names_ordered}
    for cid, data in results.items():
        for key in ['total_length', 'num_bifurcations', 'volume', 'num_loops', 'num_abnormal_degree_nodes']:
             if key in agg_data and key in data:
                 agg_data[key] += data[key]
                 
    # Creazione vettore ordinato
    feature_vector = []
    for name in feature_names_ordered:
        val = agg_data.get(name, 0.0)
        feature_vector.append(val)
        
    return np.array(feature_vector, dtype=np.float32)

# --------------------------------------------------------------------------
# Logica Principale di Test
# --------------------------------------------------------------------------

def run_single_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo device: {device}")

    # 1. Preparazione Scaler Tabulare
    # È necessario caricare il dataset originale (o un file salvato) per sapere 
    # come scalare i dati del nuovo paziente (media e varianza del training set).
    if args.excel_path:
        print("Fit dello scaler sul dataset di training...")
        df = pd.read_excel(args.excel_path)
        drop_cols = {"file_sorgente", "label1", "label2"}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        tab_cols = [c for c in numeric_cols if c not in drop_cols]
        
        scaler = StandardScaler()
        scaler.fit(df[tab_cols].astype(np.float32).values)
        tabular_dim = len(tab_cols)
    else:
        # Fallback se non si passa l'excel (funziona solo per modelli solo immagini)
        scaler = None
        tab_cols = []
        tabular_dim = 0
        if args.selected_model in ['multimodal', 'MLP_tabular']:
            raise ValueError("Per i modelli multimodali è obbligatorio fornire --excel_path per calcolare lo scaler.")

    # 2. Inizializzazione del Modello
    print(f"Caricamento architettura modello: {args.selected_model}")
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
        raise ValueError(f"Modello {args.selected_model} sconosciuto.")

    # 3. Caricamento Pesi (Checkpoint)
    if os.path.exists(args.checkpoint_path):
        print(f"Caricamento pesi da: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        # Gestione caso in cui il checkpoint contenga l'intero stato o solo model_state_dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Checkpoint non trovato: {args.checkpoint_path}")

    model.eval()

    # 4. Elaborazione del File NIfTI
    print(f"Elaborazione paziente: {args.nifti_path}")
    mips = generate_mips_from_nifti(args.nifti_path)
    transform = get_val_transform()

    # Preparazione input tensori
    inputs = {}
    
    # Immagini
    if args.selected_model in ['multi_CNN', 'multimodal']:
        for view in ['axial', 'sagittal', 'coronal']:
            img_pil = Image.fromarray(mips[view])
            inputs[view] = transform(img_pil).unsqueeze(0).to(device)

    # Dati Tabulari
    if args.selected_model in ['MLP_tabular', 'multimodal']:
        raw_feats = extract_features_from_nifti(args.nifti_path, tab_cols)
        feats_scaled = scaler.transform(raw_feats.reshape(1, -1))
        inputs['tabular'] = torch.tensor(feats_scaled, dtype=torch.float32).to(device)

    # 5. Predizione e GradCAM
    grad_cam = MultiViewGradCAM(model, device)
    
    with torch.enable_grad(): # Necessario per GradCAM
        # Attiva gradienti sugli input immagini
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float:
                v.requires_grad = True
        
        logits = model(inputs)
        prob = torch.sigmoid(logits).item()
        pred = 1 if prob >= 0.5 else 0

        cam_ax, cam_sag, cam_cor = None, None, None
        if args.selected_model in ['multi_CNN', 'multimodal']:
            cam_ax, cam_sag, cam_cor = grad_cam.generate_maps(inputs)

    # 6. Visualizzazione Risultati
    print("-" * 30)
    print(f"RISULTATO PREDITTO: {pred}")
    print(f"PROBABILITÀ (Classe 1): {prob:.4f}")
    print("-" * 30)

    if cam_ax is not None:
        # Funzione helper per overlay
        def apply_overlay(orig, cam):
            cam_resized = cv2.resize(cam, (orig.shape[1], orig.shape[0]))
            return overlay_heatmap(orig, cam_resized, alpha=0.5)

        viz_ax = apply_overlay(mips['axial'], cam_ax)
        viz_sag = apply_overlay(mips['sagittal'], cam_sag)
        viz_cor = apply_overlay(mips['coronal'], cam_cor)

        # Padding per altezze diverse
        h_max = max(viz_ax.shape[0], viz_sag.shape[0], viz_cor.shape[0])
        def pad_h(img, target_h):
            if img.shape[0] < target_h:
                pad = np.zeros((target_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
                return np.vstack([img, pad])
            return img

        viz_ax = pad_h(viz_ax, h_max)
        viz_sag = pad_h(viz_sag, h_max)
        viz_cor = pad_h(viz_cor, h_max)

        # Montage orizzontale
        border = np.ones((h_max, 10, 3), dtype=np.uint8) * 255
        montage = np.hstack([viz_ax, border, viz_sag, border, viz_cor])

        # Testo
        montage_bgr = cv2.cvtColor(montage, cv2.COLOR_RGB2BGR)
        label_txt = f"Pred: {pred} | Prob: {prob:.4f}"
        cv2.putText(montage_bgr, label_txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        output_name = f"Result_{os.path.basename(args.nifti_path)}.jpg"
        cv2.imwrite(output_name, montage_bgr)
        print(f"Immagine salvata come: {output_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Script for Single NIfTI Prediction")
    
    # Argomenti Obbligatori
    parser.add_argument("--nifti_path", type=str, required=True, help="Percorso del file .nii.gz da testare")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Percorso del file .pth (pesi del modello)")
    parser.add_argument("--excel_path", type=str, required=True, help="Percorso Excel originale (necessario per lo scaler)")
    
    # Argomenti Modello (Devono coincidere con quelli usati nel training)
    parser.add_argument("--selected_model", type=str, default="multimodal", choices=['multi_CNN', 'MLP_tabular', 'multimodal'])
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--hidden_dim", type=int, default=256)

    args = parser.parse_args()
    
    run_single_test(args)