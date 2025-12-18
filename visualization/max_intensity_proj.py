import os
from tqdm import tqdm # Per una barra di progresso
import nibabel as nib
import numpy as np
from skimage import io

def maximum_intensity_projection(nifti_filepath, projection_dimension, mask_filepath=None):
    """
    Estimates the Maximum Intensity Projection (MIP) of a 3D NIfTI image.

    Args:
        nifti_filepath (str): Path to the input 3D NIfTI image file.
        projection_dimension (str): The dimension along which to perform the projection.
                                    Can be 'axial', 'sagittal', or 'coronal'.
        mask_filepath (str, optional): Path to an optional NIfTI mask file.
                                       If provided, MIP is computed only within the masked area.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The 2D MIP image.
            - nibabel.Nifti1Image: A NIfTI image object of the MIP, with appropriate affine.
                                   Returns None if an error occurs.
    """
    try:
        # Load the NIfTI image
        '''
        Caricamento Dati: Utilizza nibabel per caricare l'immagine NIfTI (img = nib.load(nifti_filepath)). 
        Estrae i dati dell'immagine come array NumPy (data = img.get_fdata()) e la matrice affine, che mappa 
        le coordinate dei voxel allo spazio fisico (affine = img.affine).
        '''
        img = nib.load(nifti_filepath)
        data = img.get_fdata()
        affine = img.affine

        # Load the mask if provided
        '''
        Mascheramento: 
         1. Carica l'immagine della maschera.
         2. Controlla che le dimensioni della maschera corrispondano a quelle dell'immagine.
         3. Applica la maschera: Moltiplica i dati dell'immagine per i dati della maschera 
         (data *= mask_data). Questo "azzera" efficacemente tutti i voxel al di fuori della 
         regione di interesse definita dalla maschera (supponendo che la maschera sia 0 all'esterno e 1 all'interno).
        '''
        mask_data = None
        if mask_filepath:
            mask_img = nib.load(mask_filepath)
            mask_data = mask_img.get_fdata()

            # Ensure mask has the same dimensions as the image
            if mask_data.shape != data.shape:
                raise ValueError("Mask dimensions do not match image dimensions.")

            # Apply the mask: set values outside the mask to 0
            data *= mask_data

        # Determine the axis for projection
        '''
        Selezione dell'Asse: 
            Converte il nome della dimensione della proiezione (es. 'axial') nell'asse numerico corrispondente dell'array NumPy:
            - 'axial' (vista dall'alto) -> axis = 2 (proiezione lungo l'asse Z)
            - 'sagittal' (vista laterale) -> axis = 0 (proiezione lungo l'asse X)
            - 'coronal' (vista frontale) -> axis = 1 (proiezione lungo l'asse Y)
        '''
        if projection_dimension.lower() == 'axial':
            axis = 2  # Project along the Z-axis (last dimension)
        elif projection_dimension.lower() == 'sagittal':
            axis = 0  # Project along the X-axis (first dimension)
        elif projection_dimension.lower() == 'coronal':
            axis = 1  # Project along the Y-axis (second dimension)
        else:
            raise ValueError("Invalid projection_dimension. Choose 'axial', 'sagittal', or 'coronal'.")

        # Compute the Maximum Intensity Projection (MIP)
        '''
        Calcolo del MIP: 
        Utilizza np.max(data, axis=axis) per trovare il valore massimo
        lungo l'asse specificato. Il risultato (mip_data) è un array 2D.
        '''
        mip_data = np.max(data, axis=axis)

        # Create a NIfTI image object for the MIP
        # The affine needs to be adjusted for the 2D projection
        '''
        Aggiustamento dell'Affine:
        Poiché l'output è un'immagine 2D, la matrice affine 4x4 originale deve essere adattata. Il codice crea una nuova matrice affine 2D (rappresentata come 4x4) estraendo
        le componenti di scala e traslazione rilevanti dalla matrice 3D originale, in base all'asse che è stato 
        "collassato".
        '''
        new_affine = np.eye(4)
        if axis == 0: # sagittal
            new_affine[0:2, 0:2] = affine[1:3, 1:3]
            new_affine[0:2, 3] = affine[1:3, 3]
            new_affine[3,3] = affine[3,3]
        elif axis == 1: # coronal
            new_affine[0,0] = affine[0,0]
            new_affine[1,1] = affine[2,2]
            new_affine[0,3] = affine[0,3]
            new_affine[1,3] = affine[2,3]
            new_affine[3,3] = affine[3,3]
        elif axis == 2: # axial
            new_affine[0:2, 0:2] = affine[0:2, 0:2]
            new_affine[0:2, 3] = affine[0:2, 3]
            new_affine[3,3] = affine[3,3]

        # if axis == 0:  # Sagittal projection (removing first dimension)
        #     mip_affine = affine[[1, 2, 3]][:, [1, 2, 3]]
        #     mip_affine[:2, :2] = affine[1:3, 1:3]
        #     mip_affine[:2, 2] = affine[1:3, 3] # Adjusted translation
        #     mip_affine[2, :2] = 0
        #     mip_affine[2, 2] = 1 # Keep Z scale and last row/col same for 2D representation
        #     mip_affine[2, 3] = affine[3,3]

        #     # Reconstruct affine for 2D. We need to handle the removed dimension carefully
        #     # A common approach is to keep the original spacing for the remaining dimensions
        #     # and set the missing dimension's component to identity or 0.
        #     # For simplicity, we'll create a new 3x3 affine for the 2D image.
        #     # This part can be tricky and might need fine-tuning depending on how
        #     # you want the 2D affine to represent the 3D space.
        #     # For now, let's just make it a simpler 2D affine from the 3D one.
        #     #
        #     # The exact affine for a 2D projection is not straightforward as it implies a loss of information.
        #     # For visualization, often a simple 2D image is sufficient, and the affine
        #     # is less critical unless you plan further spatial transformations on the MIP itself.
        #     # Let's create a simplified affine for the 2D output.
            
        #     # Simple affine for 2D image
        #     new_affine = np.eye(4)
        #     if axis == 0: # sagittal
        #         new_affine[0:2, 0:2] = affine[1:3, 1:3]
        #         new_affine[0:2, 3] = affine[1:3, 3]
        #         new_affine[3,3] = affine[3,3]
        #     elif axis == 1: # coronal
        #         new_affine[0,0] = affine[0,0]
        #         new_affine[1,1] = affine[2,2]
        #         new_affine[0,3] = affine[0,3]
        #         new_affine[1,3] = affine[2,3]
        #         new_affine[3,3] = affine[3,3]
        #     elif axis == 2: # axial
        #         new_affine[0:2, 0:2] = affine[0:2, 0:2]
        #         new_affine[0:2, 3] = affine[0:2, 3]
        #         new_affine[3,3] = affine[3,3]
            
        #     # This simpler affine for 2D may not fully capture the 3D to 2D transformation
        #     # for all cases, but it provides a reasonable starting point.
        #     mip_nifti = nib.Nifti1Image(mip_data, new_affine, img.header)

        # else:
        #     # For axial and coronal, a more direct mapping of the relevant 2D part of affine
        #     # can be done. However, for a general solution that works across different
        #     # affine matrices, it's safer to reconstruct.
        #     # Let's use a simplified approach as a starting point.
            
        #     new_affine = np.eye(4)
        #     if axis == 0: # sagittal
        #         new_affine[0:2, 0:2] = affine[1:3, 1:3]
        #         new_affine[0:2, 3] = affine[1:3, 3]
        #         new_affine[3,3] = affine[3,3]
        #     elif axis == 1: # coronal
        #         new_affine[0,0] = affine[0,0]
        #         new_affine[1,1] = affine[2,2]
        #         new_affine[0,3] = affine[0,3]
        #         new_affine[1,3] = affine[2,3]
        #         new_affine[3,3] = affine[3,3]
        #     elif axis == 2: # axial
        #         new_affine[0:2, 0:2] = affine[0:2, 0:2]
        #         new_affine[0:2, 3] = affine[0:2, 3]
        #         new_affine[3,3] = affine[3,3]
        
        '''
        Creazione NIfTI: 
        Crea un nuovo oggetto nib.Nifti1Image utilizzando l'array 2D mip_data, la nuova matrice
        new_affine e l'header dell'immagine originale.
        '''
        
        mip_nifti = nib.Nifti1Image(mip_data, new_affine, img.header)

        return mip_data, mip_nifti

    except FileNotFoundError:
        print(f"Error: NIfTI file not found at {nifti_filepath}")
        return None, None
    except ValueError as ve:
        print(f"Error: {ve}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

def normalize_to_8bit(mip_array):
    """Normalizes a 2D array to an 8-bit image (0-255)."""
    min_val = np.min(mip_array)
    max_val = np.max(mip_array)
    if max_val > min_val:
        normalized = 255 * (mip_array - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(mip_array)
    return normalized.astype(np.uint8)

def pad_to_square(image, size, bg_color=0):
    """Add padding to an image to make it square."""
    h, w = image.shape
    # Canvas Square
    canvas = np.full((size, size), bg_color, dtype=image.dtype)
    # Offset for centering
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    # Place image in center
    canvas[y_off:y_off+h, x_off:x_off+w] = image
    return canvas

def create_mip(nifti_filepath, mask_filepath, projection_dimension,border_size=10, bg_color=0):
    """
    Creation of a single TIF image (mosaic) with Axial, Sagittal, and Coronal projections.
    Crea un'unica immagine TIF (mosaico) con le proiezioni Assiale, Sagittale e Coronale.
    """
    
    mip, _ = maximum_intensity_projection(nifti_filepath, projection_dimension, mask_filepath)
    print(mip.shape)

    if mip is None:
        print(f"Creation MIP fail for {nifti_filepath}")
        return None

    # Normalization to 8-bit
    mip_8bit = normalize_to_8bit(mip)
    
    return mip_8bit

def create_inverted_mip(nifti_filepath, projection_dimension, mask_filepath):
    """
    Create an inverted angiography-style image (dark vessels on white background).
    """
    print(f"Generating inverted angiography from: {nifti_filepath}")
    
    # MIP standard
    mip, _ = maximum_intensity_projection(nifti_filepath, projection_dimension, mask_filepath)
    
    if mip is None:
        return None

    # Normalization to 8-bit
    mip_8bit = normalize_to_8bit(mip)

    # Inverted MIP
    inverted_mip = 255 - mip_8bit
    
    return inverted_mip

def create_mip_montage(nifti_filepath, mask_filepath, border_size=10, bg_color=0):
    """
    Create a horizontal mosaic containing axial, sagittal, and coronal projections.
    """
    
    # Projection MIP
    mip_ax, _ = maximum_intensity_projection(nifti_filepath, 'axial', mask_filepath)
    mip_sag,_ = maximum_intensity_projection(nifti_filepath, 'sagittal', mask_filepath)
    mip_cor,_ = maximum_intensity_projection(nifti_filepath, 'coronal', mask_filepath)

    if mip_ax is None or mip_sag is None or mip_cor is None:
        print(f"Creation MIP fail for {nifti_filepath}")
        return None

    # Normalization
    ax_8bit = normalize_to_8bit(mip_ax)
    sag_8bit = normalize_to_8bit(mip_sag)
    cor_8bit = normalize_to_8bit(mip_cor)

    # Padding and Montage Creation
    max_dim = max(ax_8bit.shape[0], ax_8bit.shape[1],
                  sag_8bit.shape[0], sag_8bit.shape[1],
                  cor_8bit.shape[0], cor_8bit.shape[1])

    ax_pad = pad_to_square(ax_8bit, max_dim, bg_color)
    sag_pad = pad_to_square(sag_8bit, max_dim, bg_color)
    cor_pad = pad_to_square(cor_8bit, max_dim, bg_color)

    # Create border
    border = np.full((max_dim, border_size), bg_color, dtype=np.uint8)

    # Create montage
    montage = np.hstack([ax_pad, border, sag_pad, border, cor_pad])
    
    return montage


def create_inverted_mip_montage(nifti_filepath, mask_filepath, border_size=10, bg_color=255):
    """
    Create a white-background montage of axial, sagittal, and coronal inverted projections.
    """
    
    # Projection MIP
    mip_ax, _ = maximum_intensity_projection(nifti_filepath, 'axial', mask_filepath)
    mip_sag,_ = maximum_intensity_projection(nifti_filepath, 'sagittal', mask_filepath)
    mip_cor,_ = maximum_intensity_projection(nifti_filepath, 'coronal', mask_filepath)

    if mip_ax is None or mip_sag is None or mip_cor is None:
        print(f"Creation MIP fail for {nifti_filepath}")
        return None

    # Normalization
    ax_8bit = normalize_to_8bit(mip_ax)
    sag_8bit = normalize_to_8bit(mip_sag)
    cor_8bit = normalize_to_8bit(mip_cor)
    # Inverted MIP
    ax_8bit = 255 - ax_8bit
    sag_8bit = 255 - sag_8bit
    cor_8bit = 255 - cor_8bit
    # Padding and Montage Creation
    max_dim = max(ax_8bit.shape[0], ax_8bit.shape[1],
                  sag_8bit.shape[0], sag_8bit.shape[1],
                  cor_8bit.shape[0], cor_8bit.shape[1])

    ax_pad = pad_to_square(ax_8bit, max_dim, bg_color)
    sag_pad = pad_to_square(sag_8bit, max_dim, bg_color)
    cor_pad = pad_to_square(cor_8bit, max_dim, bg_color)

    # Create border
    border = np.full((max_dim, border_size), bg_color, dtype=np.uint8)

    # Create montage
    montage = np.hstack([ax_pad, border, sag_pad, border, cor_pad])
    
    return montage

