import cv2, math
import matplotlib.pyplot as plt
import numpy as np

# Función que puede ser util para empezar a pensar como evitar loops
def meshgrid(x, y):
    A = np.empty(shape=(2,x.size*y.size))
    for j in range(y.size):
        for k in range(x.size):
            A[0,j*y.size + k] = x[k]
            A[1,j*y.size + k] = y[j]
    return A

def meshgrid_vectorized(x, y):
    # Repito el vector x len(y) veces
    x_coords = np.tile(x, y.size)
    # Repito el vector y len(x) veces
    y_coords = np.repeat(y, x.size)
    # los stackeo en una matriz de 2xn
    return np.vstack((x_coords, y_coords))

def rotate(image, angle):
    angle = math.radians(angle)
    cos, sin = np.cos(angle), np.sin(angle)
    h, w = image.shape[0], image.shape[1]

    # La nueva imagen debería ser más grande para que la rotación tenga suficiente lugar
    new_h = int(np.round(np.abs(image.shape[0]*cos)+np.abs(image.shape[1]*sin))+1)
    new_w = int(np.round(np.abs(image.shape[1]*cos)+np.abs(image.shape[0]*sin))+1)

    original_centre = np.array([np.round(((h+1)/2)-1), np.round(((w+1)/2)-1)])
    new_centre = np.array([np.round(((new_h+1)/2)-1), np.round(((new_w+1)/2)-1)])

    # Por si la imagen es a color
    if len(image.shape) > 2:
      size = (new_h, new_w, image.shape[2])
    else:
      size = (new_h, new_w)

    output = np.zeros(size, dtype=np.uint8)
    
    # Create coordinate arrays for the output image
    x = np.arange(new_w)
    y = np.arange(new_h)
    mesh_2xn = meshgrid_vectorized(x, y)
    coords = np.vstack((mesh_2xn, np.ones(mesh_2xn.shape[1])))

    # Matriz R de rotacion
    R_ext = np.array([[cos, -sin, 0],
                      [sin, cos, 0],
                      [0, 0, 1]])
    
    centrar = np.array([[1,0,-original_centre[0]],
                         [0,1,-original_centre[1]],
                          [0,0,1]])
    
    re_centrar = np.array([[1,0,new_centre[0]],
                           [0,1,new_centre[1]],
                           [0,0,1]])

    # Transform coordinates
    resultado = re_centrar @ (R_ext @ (centrar @ coords))
    resultado = np.round(resultado).astype(np.int16)
    x_rotated = resultado[0].reshape(new_h, new_w)
    y_rotated = resultado[1].reshape(new_h, new_w)

    # Create mask for valid coordinates
    valid_coords_mask = (x_rotated >= 0) & (x_rotated < w) & (y_rotated >= 0) & (y_rotated < h)

    # Map valid coordinates to output image
    output[valid_coords_mask] = image[y_rotated[valid_coords_mask], x_rotated[valid_coords_mask]]
  
    return output


if __name__ == "__main__":
    img_path = 'labo3/super_mario.jpg'
    angle = 15
    image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2GRAY)

    # Rotate
    output = rotate(image, angle)

    # Plot
    fig, ax = plt.subplots(1,2, figsize=(8, 10))
    ax[0].imshow(image, cmap='gray')
    ax[1].imshow(output, cmap='gray')
    plt.show()

