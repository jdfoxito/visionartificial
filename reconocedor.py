import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import csv

DATASET_PATH = Path(r"D:/unir/low")
OUTPUT_PATH = Path("resultados_actividad")
OUTPUT_PATH.mkdir(exist_ok=True)

GAMMA_VALUES = [0.4, 0.6, 1.0, 1.5, 2.0] #granularidad
CLIP_LIMIT = 2.0                         # ver efecto en artefactos
TILE_GRID = (8, 8)                       # adaptación vs ruido


def calcular_metricas_extendidas(imagen):
    """SNR (dB), entropía deshanon, medi y desviación estandar"""
    if len(imagen.shape) == 3 and imagen.shape[2] == 3:
        imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    else:
        imagen_gray = imagen
    mean_val = np.mean(imagen_gray)
    std_val = np.std(imagen_gray)
    
    if std_val < 1e-10:
        snr_db = -float('inf')
    else:
        snr = mean_val / std_val
        snr_db = 20 * np.log10(snr) if snr > 0 else -float('inf')
    hist = np.histogram(imagen_gray, bins=256, range=(0, 256))[0]
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    
    return {
        'snr_db': snr_db,
        'entropia': entropy,
        'media': mean_val,
        'std': std_val
    }

def aplicar_transformacion_log(imagen):
    """logarítmica:"""
    img_float = imagen.astype(np.float32) / 255.0
    c = 1 / np.log(1 + img_float.max())
    resultado = c * np.log(1 + img_float)
    return (resultado * 255).astype(np.uint8)

def aplicar_correccion_gamma(imagen, gamma):
    img_float = imagen.astype(np.float32) / 255.0
    correccion = np.power(img_float, gamma)
    return (correccion * 255).astype(np.uint8)

def aplicar_clahe(imagen):
    """solo canal L"""
    lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_obj = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID)
    l_clahe = clahe_obj.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def aplicar_clahe_hsv(imagen):
    """solo canal V"""
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe_obj = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID)
    v_clahe = clahe_obj.apply(v)
    hsv_clahe = cv2.merge([h, s, v_clahe])
    return cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2RGB)

def aplicar_clahe_rgb(imagen):
    """RGB"""
    resultado = np.zeros_like(imagen)
    for canal in range(3):
        clahe_obj = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID)
        resultado[:, :, canal] = clahe_obj.apply(imagen[:,:,canal])
    return resultado

def aplicar_negativo(imagen):
    return 255 - imagen

def aplicar_brillo_multiplicativo(imagen, factor):
    """Ajuste de brillo multiplicativo"""
    resultado = imagen.astype(np.float32) * factor
    return np.clip(resultado, 0, 255).astype(np.uint8)

def procesar_imagen(ruta_imagen):
    nombre = ruta_imagen.stem
    img_original = cv2.imread(str(ruta_imagen))
    
    if img_original is None:
        print(f"err No se pudo cargar: {ruta_imagen}")
        return None
    
    print(f"[INFO] Procesando: {nombre} - {img_original.shape[:2]}")
    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    
    tecnicas = {
        'original': img_rgb,
        'negativo': aplicar_negativo(img_rgb),
        'log': aplicar_transformacion_log(img_rgb),
        'clahe_lab': cv2.cvtColor(aplicar_clahe(img_original), cv2.COLOR_BGR2RGB),
        'aplicar_clahe_hsv': aplicar_clahe_hsv(img_original),
        'clahe_rgb': aplicar_clahe_rgb(img_rgb),
        'brillo_15': aplicar_brillo_multiplicativo(img_rgb, 1.5)
    }
    
    for gamma in GAMMA_VALUES:
        tecnicas[f'gamma_{gamma}'] = aplicar_correccion_gamma(img_rgb, gamma)
    
    metricas = {}
    for nombre_tecnica, img_proc in tecnicas.items():
        metricas[nombre_tecnica] = calcular_metricas_extendidas(img_proc)
    
    return metricas

def calcular_metricas_extendidas(imagen):
    """Calcula SNR (dB), Entropía, Media y Desviación Estándar"""
    if len(imagen.shape) == 3 and imagen.shape[2] == 3:
        imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    else:
        imagen_gray = imagen
    
    mean_val = np.mean(imagen_gray)
    std_val = np.std(imagen_gray)
    
    if std_val < 1e-10:
        snr_db = -float('inf')
    else:
        snr = mean_val / std_val
        snr_db = 20 * np.log10(snr) if snr > 0 else -float('inf')
    
    # Entropía de Shannon
    hist = np.histogram(imagen_gray, bins=256, range=(0, 256))[0]
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    
    return {
        'snr_db': snr_db,
        'entropia': entropy,
        'media': mean_val,
        'std': std_val
    }
    
    
def main():
    print("Mejora de imagnes segun clases")
    if not DATASET_PATH.exists():
        print(f"err {DATASET_PATH} no existe")
        return
    
    imagenes = list(DATASET_PATH.rglob("*.jpg")) + list(DATASET_PATH.rglob("*.png"))
    print(f"{len(imagenes)} imágenes encontradas")
    
    resultados = {}
    for ruta_img in imagenes[:4]:
        resultado = procesar_imagen(ruta_img)
        if resultado:
            resultados[ruta_img.stem] = resultado
    
    guardar_metricas_csv(resultados)

if __name__ == "__main__":
    main()