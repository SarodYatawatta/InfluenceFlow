import numpy as np
import numpy.fft as fft

def generate_2d_grf(size=512,  seed=111, Hurst=0.5):
    """
    Generates a 2D Gaussian Random Field using the FFT method.
    
    Parameters:
        size (int): The pixel width and height of the grid (must be even).
        seed (int): random seed
        Hurst (float): Controls surface roughness. Closer to 1 is smoother.
                       The power spectrum scales as P(k) ~ 1 / k^(2 * Hurst + 2)
    
    Returns:
        numpy.ndarray: A 2D array representing the Gaussian Random Field.
    """
    # 1. Generate independent white noise in the spatial domain
    rng = np.random.default_rng(seed)
    white_noise = rng.normal(loc=0.0, scale=1.0, size=(size, size))
    
    # 2. Transform the noise into the frequency domain
    noise_fft = fft.fft2(white_noise)
    
    # 3. Create a coordinate grid of frequencies
    k_vec = fft.fftfreq(size)
    kx, ky = np.meshgrid(k_vec, k_vec)
    
    # 4. Calculate the frequency magnitude (k) for each grid point
    k_amplitude = np.sqrt(kx**2 + ky**2)
    
    # Avoid division by zero at the DC component (zero frequency)
    k_amplitude[0, 0] = 1.0 
    
    # 5. Define the power spectrum filter based on the Hurst exponent
    # Power spectrum P(k) is proportional to 1 / k^(2H + 2)
    # Amplitude filter is the square root of the Power spectrum
    exponent = Hurst + 1.0
    amplitude_filter = 1.0 / (k_amplitude ** exponent)
    
    # Set the DC component (mean of the field) explicitly to zero
    amplitude_filter[0, 0] = 0.0
    
    # 6. Apply the filter to the noise in the frequency domain
    filtered_fft = noise_fft * amplitude_filter
    
    # 7. Transform back to the spatial domain and take the real part
    grf = fft.ifft2(filtered_fft).real
    
    # 8. Normalize the output field to have zero mean and unit variance
    grf = (grf - np.mean(grf)) / np.std(grf)
    
    return grf

# --- Execution and Visualization ---
if 0:
    import matplotlib.pyplot as plt
    # Configure parameters
    grid_size = 40
    roughness = 0.5  # Try 0.1 for rough/noisy or 0.9 for smooth/pillowy
    
    # Generate field
    field = generate_2d_grf(size=grid_size, Hurst=roughness)
    
    # Plot the result
    plt.figure(figsize=(8, 6))
    plt.imshow(field, cmap='viridis', origin='lower')
    plt.colorbar(label='Field Amplitude')
    plt.title(f'2D Gaussian Random Field (Hurst = {roughness})')
    plt.axis('off')
    plt.savefig('img.png')
