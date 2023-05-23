import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.ndimage import binary_dilation
from scipy.ndimage import gaussian_filter
from skimage.draw import polygon
import matplotlib.pyplot as plt
from noise import snoise3
from PIL import Image
import math

map_seed = 7552


biome_names = [
    "desert",
    "savanna",
    "tropical_woodland",
    "tundra",
    "seasonal_forest",
    "rainforest",
    "temperate_forest",
    "temperate_rainforest",
    "boreal_forest"
    ]
biome_colors = [
    [255, 255, 178],
    [184, 200, 98],
    [188, 161, 53],
    [190, 255, 242],
    [106, 144, 38],
    [33, 77, 41],
    [86, 179, 106],
    [34, 61, 53],
    [35, 114, 94]
    ]

# why this process
def my_voronoi(points, size):
    edge_points = size * np.array([[-1, -1], [-1, 2], [2, -1], [2,2]])
    new_points = np.vstack([points, edge_points])
    voronoi = Voronoi(new_points)
    
    return voronoi

def voronoi_to_2darray(vor, size):
    vor_map = np.zeros((size, size), dtype=np.uint32)
    
    for i, region in enumerate(vor.regions):
        if len(region) == 0 or -1 in region : continue
        # in voronoi, vertices store in (y, x) format, due to historical qhill from fortrain
        x, y = np.array([vor.vertices[i][::-1] for i in region]).T
        rr, cc = polygon(x, y)
        # Remove pixels out of image bounds
        in_box = np.where((0 <= rr) & (rr < size) & (0 <= cc) & (cc < size))
        rr, cc = rr[in_box], cc[in_box]
        # Paint image
        vor_map[rr, cc] = i
        
    return vor_map

def lloyd_relaxation(points, size, k = 10):
    new_points = points.copy()
    
    for _ in range(k):
        vor = my_voronoi(new_points, size)
        new_points = []
        for i, region in enumerate(vor.regions):
            if len(region) == 0 or -1 in region: continue
            vertices = np.array([vor.vertices[i] for i in region])
            new_points.append(vertices.mean(axis=0))
        new_points = np.array(new_points).clip(0, size)
    
    return new_points

def perlin_2darray(size, res, seed, octaves=1, persistence=0.5, lacunarity=2.0):
    scale = size/res
    # TODO: don't really understand the concept of map_seed + seed means
    return np.array([[
        snoise3(
            (x+0.1)/scale,
            y/scale,
            seed+map_seed,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity
        )
        for x in range(size)]
        for y in range(size)
    ])
    
#Images in scikit-image are represented by Numpy ndarrays.
from skimage import exposure

# img: numpy ndarray
def histeq(img,  alpha=1):
    img_cdf, bin_centers = exposure.cumulative_distribution(img)
    img_eq = np.interp(img, bin_centers, img_cdf)
    img_eq = np.interp(img_eq, (0, 1), (-1, 1))
    return alpha * img_eq + (1 - alpha) * img
    
def quantize(data, n):
    bins = np.linspace(-1, 1, n+1)
    return (np.digitize(data, bins) - 1).clip(0, n-1)     
    
def average_cells(vor, data):
    """Returns the average value of data inside every voronoi cell"""
    size = vor.shape[0]
    count = np.max(vor)+1

    sum_ = np.zeros(count)
    count = np.zeros(count)

    for i in range(size):
        for j in range(size):
            p = vor[i, j]
            count[p] += 1
            sum_[p] += data[i, j]

    average = sum_/count
    average[count==0] = 0

    return average

def fill_cells(vor, data):
    size = vor.shape[0]
    image = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            p = vor[i, j]
            image[i, j] = data[p]

    return image

def color_cells(vor, data, dtype=int):
    size = vor.shape[0]
    image = np.zeros((size, size, 3))

    for i in range(size):
        for j in range(size):
            p = vor[i, j]
            image[i, j] = data[p]

    return image.astype(dtype)

def color_single_cell(vor, index):
    size = vor.shape[0]
    value = np.zeros((size, size, 3))
    
    for i in range(size):
        for j in range(size):
            if (vor[i,j] == index):
                value[i,j] = (0.5,0.5, 0.5)
            else:
                value[i,j] = (0.0,0.0, 0.0)
    return value

def bezier(x1, y1, x2, y2, a):
    p1 = np.array([0, 0])
    p2 = np.array([x1, y1])
    p3 = np.array([x2, y2])
    p4 = np.array([1, a])

    return lambda t: ((1-t)**3 * p1 + 3*(1-t)**2*t * p2 + 3*(1-t)*t**2 * p3 + t**3 * p4)    

from scipy.interpolate import interp1d

def bezier_lut(x1, y1, x2, y2, a):
    t = np.linspace(0, 1, 256)
    f = bezier(x1, y1, x2, y2, a)
    curve = np.array([f(t_) for t_ in t])

    return interp1d(*curve.T)

def filter_map(h_map, smooth_h_map, x1, y1, x2, y2, a, b):
    f = bezier_lut(x1, y1, x2, y2, a)
    output_map = b*h_map + (1-b)*smooth_h_map
    output_map = f(output_map.clip(0, 1))
    return output_map


def expand_image(array, scale):
    new_shape = (array.shape[0]*scale, array.shape[1]*scale, array.shape[2])
    
    print(new_shape)
    out = np.zeros(new_shape)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i][j] = array[math.floor(i/scale)][math.floor(j/scale)]
            
    return out

def get_biome_map(image_file_path):
    im = np.array(Image.open(image_file_path))[:, :, :3]
    #im = expand_image(im, 8)
    im = np.repeat(im, 8, axis=0)
    im = np.repeat(im, 8, axis=1)    
    biomes = np.zeros((256, 256))
    
    for i, color in enumerate(biome_colors):
        indices = np.where(np.all(im == color, axis=-1))
        biomes[indices] = i
        
    biomes = np.flip(biomes, axis=0).T
    return biomes

def get_boundary(vor_map, size, kernel=1):
    boundary_map = np.zeros_like(vor_map, dtype=bool)
    n, m = vor_map.shape
    
    clamp = lambda x: max(0, min(size-1, x))
    def check_for_mult(a):
        b = a[0]
        for i in range(len(a)-1):
            if a[i] != b: return 1
        return 0
    
    for i in range(n):
        for j in range(m):
            boundary_map[i, j] = check_for_mult(vor_map[
                clamp(i-kernel):clamp(i+kernel+1),
                clamp(j-kernel):clamp(j+kernel+1),
            ].flatten())
            
    return boundary_map

def process(size=1024, n=256):
    
    points = np.random.randint(0, size, (514, 2))
    points = lloyd_relaxation(points, size)
    vor = my_voronoi(points, size)
    vor_map = voronoi_to_2darray(vor, size)

    #plt.scatter(*new_points.T, s=1)
    #fig = plt.figure(dpi = 150, figsize=(8,4))
    #plt.imshow(map)
    #fig = voronoi_plot_2d(vor)
    #plt.scatter(*new_points.T, s=1)
    #test_map= color_single_cell(map, 100)

    #fig = plt.figure(dpi=150, figsize=(4, 4))

    #plt.scatter(*new_points.T, s=1)
    #fig = plt.figure(dpi = 150, figsize=(8,4))

    temperature_map = perlin_2darray(size, 2, 10)
    precipitation_map = perlin_2darray(size, 2, 20)
    temperature_map_uniform = histeq(temperature_map, 0.33)
    precipitation_map_uniform = histeq(precipitation_map, 0.33)

    temperature_cells = average_cells(vor_map, temperature_map)
    precipitation_cells = average_cells(vor_map, precipitation_map)

    temperature_map = fill_cells(vor_map, temperature_cells)
    precipitation_map = fill_cells(vor_map, precipitation_cells)
    
    quantize_temperature_cells = quantize(temperature_cells, n)
    quantize_precipitation_cells = quantize(precipitation_cells, n)

    quantize_temperature_map = fill_cells(vor_map, quantize_temperature_cells)
    quantize_precipitation_map = fill_cells(vor_map, quantize_precipitation_cells)

    height_map = perlin_2darray(size, 4, 0, octaves=6, persistence=0.5, lacunarity=2)
    land_mask = height_map > 0
    # plt.subplot(1,2, 1)
    # plt.title("temperature")
    # plt.imshow(temperature_map, cmap='rainbow')
    # plt.subplot(1,2, 2)
    # plt.title("precipitation")
    # plt.imshow(precipitation_map, cmap='Blues')
    # plt.show()

    biomes = get_biome_map("biome_image.png")
    
    n = len(quantize_temperature_cells)
    biome_cells = np.zeros(n, dtype=np.uint32)

    for i in range(n):
        temp, precip = quantize_temperature_cells[i], quantize_precipitation_cells[i]
        biome_cells[i] = biomes[temp, precip]
            
    biome_map = fill_cells(vor_map, biome_cells).astype(np.uint32)
    biome_color_map = color_cells(biome_map, biome_colors)
    
    sea_color = np.array([12, 14, 255])
    mountain_color = np.array([0, 0, 0])
    land_mask_color = np.repeat(land_mask[:, :, np.newaxis], 1, axis=-1)
    masked_biome_color_map = land_mask_color*biome_color_map + (1-land_mask_color)*mountain_color
    
    boundary = get_boundary(biome_map, size, 10)
    boundary_voronoi = get_boundary(vor_map, size, 1)
    boundary_all = np.logical_or(boundary, boundary_voronoi)
    
    #new_image = biome_color_map * (1 - boundary) + boundary
    
    fig = plt.figure(dpi=150, figsize=(4, 4))
    plt.imshow(biomes)
    plt.title("Temperature–Precipitation graph")
    
    fig = plt.figure(figsize=(5, 5), dpi=150)
    plt.imshow(biome_color_map)
    
    # fig = plt.figure(figsize=(5, 5), dpi=150)
    # plt.imshow(masked_biome_color_map)
    
    
    #fig = plt.figure(figsize=(5, 5), dpi=150)
    fig = plt.figure(figsize=(5, 5), dpi=150)
    plt.imshow(boundary_all)
    
    loose_river_mask = binary_dilation(boundary, iterations=8)
    rivers_height = gaussian_filter(boundary.astype(np.float64), sigma=2)*loose_river_mask
    
    fig = plt.figure(figsize=(5, 5), dpi=150)
    plt.imshow(loose_river_mask)
    fig = plt.figure(figsize=(5, 5), dpi=150)
    plt.imshow(rivers_height)
    
    land_mask_color = np.repeat(boundary[:, :, np.newaxis], 3, axis=-1)
    rivers_biome_color_map = (1-land_mask_color)*biome_color_map + land_mask_color*mountain_color

    plt.figure(figsize=(5,5), dpi=150)
    plt.imshow(rivers_biome_color_map)
    
    
    plt.show()
    
process()    