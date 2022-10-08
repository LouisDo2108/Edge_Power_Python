import numpy as np
import cv2
from skimage.segmentation import find_boundaries
from copy import deepcopy
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import json


def show_visualize(imgray, maskgray, thick, normal_vector, edge_element, grid, interpolate):
    plt.subplot(231)
    plt.imshow(imgray, cmap='gray')
    plt.title('image')
    plt.subplot(232)
    plt.imshow(maskgray, cmap='gray')
    plt.title('mask')
    plt.subplot(233)
    plt.imshow(thick, cmap='gray')
    plt.title('boundary')
    plt.subplot(234)
    plt.imshow(cv2.convertScaleAbs(normal_vector,
               alpha=255/normal_vector.max()), cmap='gray')
    plt.title('normal_vector')
    plt.subplot(235)
    temp_edge_element = deepcopy(edge_element)
    temp_edge_element[~thick.astype('bool')] = 0
    plt.imshow(temp_edge_element, cmap='gray')
    plt.title('edge_element')
    plt.subplot(236)
    plt.imshow(np.flip(grid.T, 0), extent=(0, 255, 0, 255), cmap='gray')
    plt.title(interpolate)
    plt.gcf().set_size_inches(12, 8)
    plt.show()


def calculate_edge_power(thick, grid_z2):
    values = thick*np.flip(grid_z2.T, 0)
    edge_power_mean = np.mean(np.square(values))
    edge_power_sum = np.sum(np.square(values))
    return edge_power_mean, edge_power_sum


def interpolate_edge_vector(points, values, interpolate='cubic'):
    grid_x, grid_y = np.mgrid[0:255:256j, 0:255:256j]
    grid =  griddata(points, values, (grid_x, grid_y),
                        method=interpolate, fill_value=0, rescale=True)
    return grid


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
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgray = cv2.resize(imgray, (256, 256))
    mask = cv2.imread(mask_path)
    maskgray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    maskgray = cv2.resize(maskgray, (256, 256))
    thick = find_boundaries(maskgray, mode='thick')
    return img, imgray, maskgray, thick


def get_edge_power(img_path, filter='scharr', interpolate='cubic', color=False, visualize=True):

    # Read image and its mask
    img, imgray, maskgray, thick = read_image(img_path)
    
    if np.sum(maskgray) == 0:
        return 0, 0, 0

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
    grid = interpolate_edge_vector(points, values, interpolate)

    if visualize:
        if color:
            image = img
        else:
            image = imgray
        show_visualize(image, maskgray, thick, normal_vector,
                       edge_element, grid, interpolate)
    # if interpolate == 'cubic':
    #     grid = grid_z2
    # elif interpolate == 'linear':
    #     grid = grid_z1
    # else:
    #     grid = grid_z0
    edge_power_mean, edge_power_sum = calculate_edge_power(thick, grid)
    print('{}: mean: {:.4f}, sum: {:.4f}, overall: {:.4f}'.format(
        interpolate, edge_power_mean, edge_power_sum, edge_power_mean*edge_power_sum))

    return edge_power_mean, edge_power_sum, edge_power_mean*edge_power_sum

def get_edge_power_json(p):
    cat = [x for x in p.iterdir() if x.is_dir()]

    result = {}

    for c in cat:
        if str(c) not in result:
            result[str(c)] = {}
        subcat = [x for x in c.iterdir() if x.is_dir()]
        for sc in subcat:
            if str(sc) not in result[str(c)]:
                result[str(c)][str(sc)] = []
            images = sc.glob('**/*.jpg')
            for img in images:
                m, s, o = get_edge_power(str(img), filter='scharr', color=True, visualize=False)
                result[str(c)][str(sc)].append({
                'source':images,
                'mean':m,
                'sum':s,
                'overall':o
                })
                
    with open('edge_power.json', 'w') as fp:
        json.dump(result, fp)
        
    return result
