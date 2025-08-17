import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --- Функції обробки зображень SVD ---

def read_image(path, mode='RGB'):
    """
    Reads an image from the specified path.
    Uses matplotlib for reading.
    Returns a NumPy array, normalized to [0.0, 1.0].
    """
    try:
        img = mpimg.imread(path)
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        return img.astype(np.float32)
    except FileNotFoundError:
        print(f"Помилка: Файл '{path}' не знайдено.")
        print("Будь ласка, переконайтеся, що файли зображень (grayscale_cat.jpg, rgb_dog.jpg) існують у тій же директорії.")
        if mode == 'RGB':
            return np.zeros((100, 100, 3), dtype=np.float32)
        else:
            return np.zeros((100, 100), dtype=np.float32)

def show_image(image_data, title="Відновлене зображення", mode='Grayscale'):
    """
    Displays an image using Matplotlib.
    Handles float [0.0, 1.0] ranges.
    """
    plt.figure(figsize=(6, 6))
    # Забезпечуємо, що значення знаходяться в діапазоні [0.0, 1.0] для float зображень
    display_image = np.clip(image_data, 0.0, 1.0)
    
    # Визначаємо колірну карту
    cmap = 'gray' if mode == 'Grayscale' else None
    
    plt.imshow(display_image, cmap=cmap)
    plt.title(title)
    plt.axis('off') 
    plt.show()

def get_image_shape(image_data):
    """Returns the shape of the image data."""
    return image_data.shape

def perform_svd(channel_data):
    """
    Performs Singular Value Decomposition on a 2D channel.
    Returns U, S, Vt (V transposed).
    """
    if channel_data.ndim != 2:
        raise ValueError("perform_svd очікує 2D масив для каналу.")
    
    # дані є float для SVD
    if channel_data.dtype not in [np.float32, np.float64]:
         channel_data = channel_data.astype(np.float32)

    U, S, Vt = np.linalg.svd(channel_data, full_matrices=False)
    return U, S, Vt

def reconstruct_image(U, S, Vt, num_singular_values):
    """
    Reconstructs the image from the SVD components.

    Parameters:
    U (numpy.ndarray): The U matrix from SVD.
    S (numpy.ndarray): The singular values from SVD.
    Vt (numpy.ndarray): The Vt matrix from SVD.
    num_singular_values (int): The number of singular values to use for reconstruction.

    Returns:
    numpy.ndarray: The reconstructed image data.
    """
    k = num_singular_values

    S_reduced = np.zeros_like(S, dtype=S.dtype) 
    S_reduced[:k] = S[:k]

    reconstructed_channel = np.dot(U, np.dot(np.diag(S_reduced), Vt))
    reconstructed_channel = np.clip(reconstructed_channel, 0.0, 1.0)
    
    return reconstructed_channel.astype(np.float32)

def reconstruct_rgb_image(Ur, Sr, Vr, Ug, Sg, Vg, Ub, Sb, Vb, num_singular_values):
    """
    Reconstructs an RGB image from the SVD components of its channels.
    Each channel is reconstructed and then stacked to form the RGB image.
    """
    r_channel = reconstruct_image(Ur, Sr, Vr, num_singular_values)
    g_channel = reconstruct_image(Ug, Sg, Vg, num_singular_values)
    b_channel = reconstruct_image(Ub, Sb, Vb, num_singular_values)

    reconstructed_rgb_image = np.stack([r_channel, g_channel, b_channel], axis=-1)

    reconstructed_rgb_image = np.clip(reconstructed_rgb_image, 0.0, 1.0)
    return reconstructed_rgb_image

def reconstruction_error(reconstructed_image, original_image):
    """
    Computes the reconstruction error between the original and reconstructed images.
    """
    flat_reconstructed = reconstructed_image.flatten()
    flat_original = original_image.flatten()
    return np.mean(np.square(flat_reconstructed - flat_original))

if __name__ == "__main__":
    print("--- Розпочинаємо SVD-реконструкцію зображень ---")
    print("Будь ласка, переконайтеся, що файли 'grayscale_cat.jpg' та 'rgb_dog.jpg' існують у поточній директорії.")

    img1 = read_image("grayscale_cat.jpg", mode='Grayscale')
    img2 = read_image("rgb_dog.jpg", mode='RGB')

    if img1.size == 0 or img2.size == 0:
        print("\nНеможливо продовжити демонстрацію, оскільки файли зображень не завантажено.")
    else:
        print("\n--- Вихідні зображення ---")
        show_image(img1, title='сірий кіт', mode='Grayscale')
        show_image(img2, title='кольоровий собака', mode='RGB')

        img1_shape = get_image_shape(img1)
        img2_shape = get_image_shape(img2)
        print("Форма зображення 1 (сірий кіт):", img1_shape)
        print("Форма зображення 2 (кольоровий собака):", img2_shape)

        print("\n--- SVD для сірого зображення ---")
        U, S, Vt = perform_svd(img1)
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(len(S)), S)
        plt.title('Сингулярні значення сірого зображення')
        plt.xlabel('Індекс сингулярного значення')
        plt.ylabel('Величина сингулярного значення')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        # Реконструкція сірого зображення з різною кількістю сингулярних значень
        print("\n--- Реконструкція сірого зображення ---")
        reconstructed_image_10 = reconstruct_image(U, S, Vt, num_singular_values=10)
        show_image(reconstructed_image_10, title='Реконструкція сірого (k=10)', mode='Grayscale')
        print("Помилка реконструкції (k=10):", reconstruction_error(reconstructed_image_10, img1))
        reconstructed_image_75 = reconstruct_image(U, S, Vt, num_singular_values=75)
        show_image(reconstructed_image_75, title='Реконструкція сірого (k=75)', mode='Grayscale')
        print("Помилка реконструкції (k=75):", reconstruction_error(reconstructed_image_75, img1))

        print("\n--- SVD та реконструкція для кольорового зображення ---")
        # Розділення RGB зображення на канали та SVD для кожного
        # Переконайтеся, що img2 має 3 канали
        if img2.ndim == 3 and img2.shape[2] == 3:
            Ur, Sr, Vr = perform_svd(img2[:, :, 0])  # Червоний канал
            Ug, Sg, Vg = perform_svd(img2[:, :, 1])  # Зелений канал
            Ub, Sb, Vb = perform_svd(img2[:, :, 2])  # Синій канал

            # Реконструкція RGB зображення з різною кількістю сингулярних значень
            print("Реконструкція кольорового зображення (k=10)...")
            reconstructed_rgb_image_10 = reconstruct_rgb_image(Ur, Sr, Vr, Ug, Sg, Vg, Ub, Sb, Vb, num_singular_values=10)
            show_image(reconstructed_rgb_image_10, title='Реконструкція RGB (k=10)', mode='RGB')
            print("Помилка реконструкції (k=10):", reconstruction_error(reconstructed_rgb_image_10, img2))
            print("Реконструкція кольорового зображення (k=75)...")
            reconstructed_rgb_image_75 = reconstruct_rgb_image(Ur, Sr, Vr, Ug, Sg, Vg, Ub, Sb, Vb, num_singular_values=75)
            show_image(reconstructed_rgb_image_75, title='Реконструкція RGB (k=75)', mode='RGB')
            print("Помилка реконструкції (k=75):", reconstruction_error(reconstructed_rgb_image_75, img2))
        else:
            print("Зображення 'rgb_dog.jpg' не є коректним 3-канальним RGB зображенням.")

