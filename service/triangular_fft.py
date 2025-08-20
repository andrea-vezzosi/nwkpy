"""
FFT su Griglia Triangolare - Soluzione Semplice
===============================================

Implementazione semplice per trasformata di Fourier 2D su griglie triangolari
tramite interpolazione su griglia regolare + FFT standard.

Autore: Claude
Data: Agosto 2025
"""

import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from scipy.fft import fft2, fftfreq
import matplotlib.pyplot as plt
import matplotlib.tri as tri


def fft_triangular_mesh(tri_points, tri_values, grid_size=128):
    """
    FFT su griglia triangolare via interpolazione
    
    Args:
        tri_points: Array (N, 2) con coordinate (x, y) dei vertici triangolari
        tri_values: Array (N,) con valori della funzione sui vertici
        grid_size: Dimensione della griglia regolare per interpolazione
    
    Returns:
        fft_result: Trasformata di Fourier 2D complessa
        freqs_x: Array delle frequenze lungo x
        freqs_y: Array delle frequenze lungo y
        X: Griglia delle coordinate x
        Y: Griglia delle coordinate y  
        Z: Valori interpolati sulla griglia regolare
    """
    # Determina i limiti della griglia
    x_min, x_max = tri_points[:, 0].min(), tri_points[:, 0].max()
    y_min, y_max = tri_points[:, 1].min(), tri_points[:, 1].max()
    
    # Crea griglia regolare
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Interpola i valori dalla griglia triangolare alla griglia regolare
    Z = griddata(tri_points, tri_values, (X, Y), method='cubic')
    
    # Gestisci eventuali NaN ai bordi (riempi con zero)
    Z = np.nan_to_num(Z, nan=0.0)
    
    # Calcola FFT 2D
    fft_result = fft2(Z)
    
    # Calcola le frequenze corrispondenti
    dx = (x_max - x_min) / grid_size
    dy = (y_max - y_min) / grid_size
    freqs_x = fftfreq(grid_size, dx)
    freqs_y = fftfreq(grid_size, dy)
    
    return fft_result, freqs_x, freqs_y, X, Y, Z


def plot_results(tri_points, tri_values, fft_result, freqs_x, freqs_y, X, Y, Z):
    """
    Visualizza i risultati dell'analisi FFT
    
    Args:
        tri_points: Punti della mesh triangolare originale
        tri_values: Valori originali
        fft_result: Risultato della FFT
        freqs_x, freqs_y: Frequenze
        X, Y, Z: Griglia regolare e valori interpolati
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Mesh triangolare originale
    triang = tri.Triangulation(tri_points[:, 0], tri_points[:, 1])
    axes[0,0].tricontourf(triang, tri_values, levels=20, cmap='viridis')
    axes[0,0].triplot(triang, 'k-', alpha=0.3, linewidth=0.5)
    axes[0,0].set_title('Mesh Triangolare Originale')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    axes[0,0].set_aspect('equal')
    
    # 2. Interpolazione su griglia regolare
    im2 = axes[0,1].contourf(X, Y, Z, levels=20, cmap='viridis')
    axes[0,1].set_title('Interpolazione su Griglia Regolare')
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('y')
    axes[0,1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0,1])
    
    # 3. Spettro di potenza (ampiezza della FFT)
    power_spectrum = np.abs(fft_result)**2
    # Usa fftshift per centrare le frequenze zero
    from scipy.fft import fftshift
    power_shifted = fftshift(power_spectrum)
    freqs_x_shifted = fftshift(freqs_x)
    freqs_y_shifted = fftshift(freqs_y)
    
    im3 = axes[1,0].imshow(np.log10(power_shifted + 1e-12), 
                          extent=[freqs_x_shifted.min(), freqs_x_shifted.max(),
                                 freqs_y_shifted.min(), freqs_y_shifted.max()],
                          cmap='hot', aspect='auto', origin='lower')
    axes[1,0].set_title('Spettro di Potenza (log scale)')
    axes[1,0].set_xlabel('Frequenza x')
    axes[1,0].set_ylabel('Frequenza y')
    plt.colorbar(im3, ax=axes[1,0])
    
    # 4. Fase della trasformata
    phase = np.angle(fft_result)
    phase_shifted = fftshift(phase)
    im4 = axes[1,1].imshow(phase_shifted,
                          extent=[freqs_x_shifted.min(), freqs_x_shifted.max(),
                                 freqs_y_shifted.min(), freqs_y_shifted.max()],
                          cmap='hsv', aspect='auto', origin='lower')
    axes[1,1].set_title('Fase della Trasformata')
    axes[1,1].set_xlabel('Frequenza x')
    axes[1,1].set_ylabel('Frequenza y')
    plt.colorbar(im4, ax=axes[1,1])
    
    plt.tight_layout()
    return fig


def create_example_mesh(n_points=100):
    """
    Crea una mesh triangolare di esempio con una funzione test
    
    Args:
        n_points: Numero approssimativo di punti
    
    Returns:
        points: Coordinate dei vertici (N, 2)
        values: Valori della funzione test sui vertici (N,)
    """
    # Crea punti su una griglia con piccole perturbazioni random
    n_side = int(np.sqrt(n_points))
    x = np.linspace(0, 2*np.pi, n_side)
    y = np.linspace(0, 2*np.pi, n_side)
    X, Y = np.meshgrid(x, y)
    
    # Aggiungi piccole perturbazioni per rendere la mesh meno regolare
    noise_x = np.random.normal(0, 0.05, X.shape)
    noise_y = np.random.normal(0, 0.05, Y.shape)
    
    points = np.column_stack([(X + noise_x).ravel(), (Y + noise_y).ravel()])
    
    # Funzione test: combinazione di onde sinusoidali
    values = (np.sin(points[:, 0]) * np.cos(points[:, 1]) + 
              0.5 * np.sin(2*points[:, 0]) * np.sin(2*points[:, 1]))
    
    return points, values


# Esempio di utilizzo
if __name__ == "__main__":
    print("Creazione mesh triangolare di esempio...")
    
    # Crea mesh di test
    tri_points, tri_values = create_example_mesh(n_points=150)
    
    print(f"Mesh creata con {len(tri_points)} punti")
    print(f"Range valori: [{tri_values.min():.3f}, {tri_values.max():.3f}]")
    
    # Esegui FFT
    print("Esecuzione FFT su griglia triangolare...")
    fft_result, freqs_x, freqs_y, X, Y, Z = fft_triangular_mesh(
        tri_points, tri_values, grid_size=128
    )
    
    # Calcola alcune statistiche
    power_spectrum = np.abs(fft_result)**2
    total_power = np.sum(power_spectrum)
    max_power_idx = np.unravel_index(np.argmax(power_spectrum), power_spectrum.shape)
    dominant_freq_x = freqs_x[max_power_idx[1]]
    dominant_freq_y = freqs_y[max_power_idx[0]]
    
    print(f"Potenza totale nello spettro: {total_power:.2e}")
    print(f"Frequenza dominante: fx={dominant_freq_x:.3f}, fy={dominant_freq_y:.3f}")
    
    # Visualizza risultati
    print("Visualizzazione risultati...")
    fig = plot_results(tri_points, tri_values, fft_result, freqs_x, freqs_y, X, Y, Z)
    plt.show()
    
    print("Completato!")
