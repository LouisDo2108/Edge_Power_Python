import numpy as np
import cv2
from skimage.segmentation import find_boundaries
from copy import deepcopy
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def show_visualize(imgray, maskgray, thick, normal_vector, edge_element, grid_z0, grid_z1, grid_z2):
    plt.subplot(241)
    plt.imshow(imgray, cmap='gray')
    plt.title('image')
    plt.subplot(242)
    plt.imshow(maskgray, cmap='gray')
    plt.title('mask')
    plt.subplot(243)
    plt.imshow(thick, cmap='gray')
    plt.title('boundary')
    plt.subplot(244)
    plt.imshow(cv2.convertScaleAbs(normal_vector,
               alpha=255/normal_vector.max()), cmap='gray')
    plt.title('normal_vector')
    plt.subplot(245)
    temp_edge_element = deepcopy(edge_element)
    temp_edge_element[~thick.astype('bool')] = 0
    plt.imshow(temp_edge_element, cmap='gray')
    plt.title('edge_element')
    plt.subplot(246)
    plt.imshow(grid_z0.T, extent=(0, 255, 0, 255), origin='lower', cmap='gray')
    plt.title('Nearest')
    plt.subplot(247)
    plt.imshow(grid_z1.T, extent=(0, 255, 0, 255), origin='lower', cmap='gray')
    plt.title('Linear')
    plt.subplot(248)
    plt.imshow(grid_z2.T, extent=(0, 255, 0, 255), origin='lower', cmap='gray')
    plt.title('Cubic')
    plt.gcf().set_size_inches(16, 8)
    plt.show()


def calculate_edge_power(points, grid_z2):
    values = []
    edge_vector = grid_z2
    for boundary_pixel in points:
        x, y = boundary_pixel
        value = edge_vector.T[256-y][x]
        values.append(value)
    values = np.array(values)
    edge_power_mean = np.mean(np.square(values))
    edge_power_sum = np.sum(np.square(values))
    return edge_power_mean, edge_power_sum


def interpolate_edge_vector(points, values):
    grid_x, grid_y = np.mgrid[0:255:256j, 0:255:256j]
    grid_z0 = griddata(points, values, (grid_x, grid_y),
                       method='nearest', fill_value=0, rescale=True)
    grid_z1 = griddata(points, values, (grid_x, grid_y),
                       method='linear', fill_value=0, rescale=True)
    grid_z2 = griddata(points, values, (grid_x, grid_y),
                       method='cubic', fill_value=0, rescale=True)

    return grid_z0, grid_z1, grid_z2


def get_edge_vectors(thick, edge_element):
    points = []
    values = []
    xx, yy = np.nonzero(thick)
    for x, y in list(zip(xx, yy)):
        points.append((y, 256-x))
        values.append(edge_element[x][y])
    points = np.array(points)
    values = np.array(values)
    return points, values


def get_edge_element(thick, normal_vector):
    full_gradient_rms = np.sqrt(
        np.mean(np.square(normal_vector[thick.astype('bool')])))
    edge_element = deepcopy(normal_vector)
    edge_element[thick.astype('bool')] = edge_element[thick.astype(
        'bool')] / full_gradient_rms

    return edge_element


def correct_normal_vector_orientation(dx, dy, normal_vector):
    orientation = np.arctan2(dy, dx) * (180 / np.pi)
    orientation[orientation < 0] = -1
    orientation[orientation >= 0] = 1
    normal_vector = np.abs(normal_vector)*orientation
    return normal_vector


def read_image(img_path):
    mask_path = img_path[:-3] + 'png'
    image = cv2.imread(img_path)
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgray = cv2.resize(imgray, (256, 256))
    mask = cv2.imread(mask_path)
    maskgray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    maskgray = cv2.resize(maskgray, (256, 256))
    thick = find_boundaries(maskgray, mode='thick')
    return imgray, maskgray, thick

def get_edge_power(img_path, filter='scharr', visualize=True):

    # Read image and its mask
    imgray, maskgray, thick = read_image(img_path)

    # Get the image's normal vector
    if filter == 'steering':
        dx_steering_kernel = np.array(
            [[0.36787944,  0., -0.36787944],
             [0.60653066, -0., -0.60653066],
             [0.36787944, -0., -0.36787944]])
        dy_steering_kernel = dx_steering_kernel.T
        dx = cv2.filter2D(imgray, cv2.CV_64F, dx_steering_kernel)
        dy = cv2.filter2D(imgray, cv2.CV_64F, dy_steering_kernel)
    else:
        dx = cv2.Scharr(imgray, cv2.CV_64F, dx=1, dy=0)
        dy = cv2.Scharr(imgray, cv2.CV_64F, dx=0, dy=1)
    normal_vector = np.sqrt(np.power(dx, 2) + np.power(dy, 2))

    # Correct the normal vector direction ###
    normal_vector = correct_normal_vector_orientation(dx, dy, normal_vector)

    # Calculate edge element
    edge_element = get_edge_element(thick, normal_vector)

    # Get all edge vectors
    points, values = get_edge_vectors(thick, edge_element)

    # Interpolate edge vectors
    grid_z0, grid_z1, grid_z2 = interpolate_edge_vector(points, values)

    edge_power_mean, edge_power_sum = calculate_edge_power(points, grid_z2)

    if visualize:
        show_visualize(imgray, maskgray, thick, normal_vector,
                       edge_element, grid_z0, grid_z1, grid_z2)
    print('mean:', edge_power_mean)
    print('sum:', edge_power_sum)