__author__ = 'Robin Vandaele'

import numpy as np # handling arrays and general math
import dionysus # computing topological persistence
from scipy.stats import skew, kurtosis, iqr # summary statistics of data
import scipy # working with graphs
import SimpleITK as sitk # handling dicom images
import imageio # handling images
from skimage.measure import marching_cubes # lesion mask to surface mesh



def read_dcm(path):
    
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    vol = reader.Execute()
    
    return vol


def read_nifti(path):
    
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    vol = reader.Execute()
    
    return vol


def np_from_sitk(vol):

    return sitk.GetArrayFromImage(vol)


def point_cloud_from_lesion(vol, lesion_np):
    
    verts = marching_cubes(lesion_np)[0]
    verts = np.array([vol.TransformContinuousIndexToPhysicalPoint(list(map(float, list(v)))) for v in verts])

    return verts


def mesh_from_lesion(vol, lesion_np):
    
    verts, faces, normals, values = marching_cubes(lesion_np)
    verts = np.array([vol.TransformContinuousIndexToPhysicalPoint(list(map(float, list(v)))) for v in verts])

    return verts, faces


def slice_and_stack(path):
    
    slices = imageio.imread(path)
    m, n = slices.shape[1], slices.shape[1] 
    num_slices = int(slices.shape[0] / m)
    
    stacked = np.zeros((num_slices, m, n)).astype("int")
    for idx in range(num_slices):
        stacked[num_slices - idx - 1,:,:] = slices[range(idx * m, (idx + 1) * m),:]
    return stacked


def bbox_3D(img):

    x = np.any(img, axis=(1, 2))
    y = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return xmin, xmax, ymin, ymax, zmin, zmax


def segment_3Darray(segmentation, array, scale=None, width=None, fillnan=True):
    
    x1, x2, y1, y2, z1, z2 = bbox_3D(segmentation)
    
    if not width is None:
        widthx = np.min([width[0], array.shape[0]])
        widthy = np.min([width[1], array.shape[1]])
        widthz = np.min([width[2], array.shape[2]])
        
        x1 = int(x1 - (widthx - (x2 - x1)) / 2)
        if x1 < 0:
            x1 = 0
        x2 = x1 + widthx
        if x2 > array.shape[0]:
            x2 = array.shape[0]
            x1 = x2 - widthx
            
        y1 = int(y1 - (widthy - (y2 - y1)) / 2)
        if y1 < 0:
            y1 = 0
        y2 = y1 + widthy
        if y2 > array.shape[1]:
            y2 = array.shape[1]
            y1 = y2 - widthy
            
        z1 = int(z1 - (widthz - (z2 - z1)) / 2)
        if z1 < 0:
            z1 = 0
        z2 = z1 + widthz
        if z2 > array.shape[2]:
            z2 = array.shape[2]
            z1 = z2 - widthz
    
    segmented_array = array[x1:x2, y1:y2, z1:z2].astype("float")
    if fillnan:
        segmented_array[segmentation[x1:x2, y1:y2, z1:z2] == 0] = np.nan
    
    if scale:
        to_scale = ~np.isnan(segmented_array)
        m, M = np.min(segmented_array[to_scale]), np.max(segmented_array[to_scale])
        segmented_array[to_scale] = np.round((((segmented_array[to_scale] - m) * (scale[1] - scale[0])) / (M - m)) + scale[0])
    
    return(segmented_array)


def geodesic_distances(verts, faces):
    
    edges = np.zeros((faces.shape[0] * 3, 2)).astype("int")
    edge_idx = 0
    for face_idx in range(faces.shape[0]):
        for vert_idx in [0, 1, 2]:
            edges[edge_idx,:] = np.delete(faces[face_idx,:], vert_idx)
            edge_idx += 1
    
    nverts = np.max(edges) + 1
    values = [np.linalg.norm(verts[e[0]] - verts[e[1]]) for e in edges]
    values = values + values
    
    adjacency = scipy.sparse.coo_matrix((values, (np.append(edges[:,0], edges[:,1]), np.append(edges[:,1], edges[:,0]))),
                                        shape=(nverts, nverts))
    del values, edges
    
    return scipy.sparse.csgraph.shortest_path(adjacency, directed=False, unweighted=False)


def lower_star_img3D(img3D):
    """
    Compute persistent homology of a lower star filtration on an image

    Parameters
    ----------
    img: ndarray (K, M, N)
        An array of single channel 3D image data
        np.nan entries correspond to empty pixels

    Returns
    -------
    L: list of three ndarrays (T, 2)
        A list of three persistence diagrams corresponding to the sublevelset filtration
        The index in the list corresponds to the dimension of the diagram
    """
    empty_pixel = np.isnan(img3D)
    some_pixels_empty = np.sum(empty_pixel) > 0
    if some_pixels_empty:
        max_pixel = np.max(img3D[~empty_pixel])
        img3Dcopy = img3D.copy() 
        img3Dcopy[empty_pixel] = max_pixel + 1
        f_lower_star = dionysus.fill_freudenthal(img3Dcopy)
        
    else:
        f_lower_star = dionysus.fill_freudenthal(img3D)
    
    ph = dionysus.homology_persistence(f_lower_star)
    dgms = list()
    for idx1, dgm in enumerate(dionysus.init_diagrams(ph, f_lower_star)):
        if idx1 > 2:
            break
        dgm_array = np.zeros([len(dgm), 2])
        for idx2, dgm_point in enumerate(dgm):
            d = dgm_point.death
            if some_pixels_empty:
                if d > max_pixel + 1 / 2:
                    d = np.inf
            dgm_array[idx2,:] = [dgm_point.birth, d]
        dgms.append(dgm_array)
    
    return dgms


def persistence_statistics(dgm):
    
    infinite_lifespans = np.where(dgm[:,1] == np.inf)[0]
    dgm = np.delete(dgm, infinite_lifespans, axis=0)
    
    if dgm.shape[0] == 0:
        return {"min_birth": np.nan,
                "no_infinite_lifespans": len(infinite_lifespans),
                "no_finite_lifespans": 0, 
                "mean_finite_midlifes": np.nan, 
                "mean_finite_lifespans": 0,
                "std_finite_midlifes": np.nan, 
                "std_finite_lifespans": 0,
                "skew_finite_midlifes": np.nan, 
                "skew_finite_lifespans": 0,
                "kurtosis_finite_midlifes": np.nan, 
                "kurtosis_finite_lifespans": 0,
                "median_finite_midlifes": np.nan, 
                "median_finite_lifespans": 0,
                "Q1_finite_midlifes": np.nan, 
                "Q1_finite_lifespans": 0,
                "Q3_finite_midlifes": np.nan, 
                "Q3_finite_lifespans": 0,
                "IQR_finite_midlifes": np.nan, 
                "IQR_finite_lifespans": 0,
                "entropy_finite_lifespans": 0}
    
    lifespans = dgm[:,1] - dgm[:,0]
    sum_lifespans = np.sum(lifespans)
    midlifes = (dgm[:,0] + dgm[:,1]) / 2
    
    return {"min_birth": np.min(dgm[:,0]),
            "no_infinite_lifespans": len(infinite_lifespans),
            "no_finite_lifespans": dgm.shape[0], 
            "mean_finite_midlifes": np.mean(midlifes), 
            "mean_finite_lifespans": np.mean(lifespans),
            "std_finite_midlifes": np.std(midlifes), 
            "std_finite_lifespans": np.std(lifespans),
            "skew_finite_midlifes": skew(midlifes), 
            "skew_finite_lifespans": skew(lifespans),
            "kurtosis_finite_midlifes": kurtosis(midlifes), 
            "kurtosis_finite_lifespans": kurtosis(lifespans),
            "median_finite_midlifes": np.median(midlifes), 
            "median_finite_lifespans": np.median(lifespans),
            "Q1_finite_midlifes": np.percentile(midlifes, 25), 
            "Q1_finite_lifespans": np.percentile(lifespans, 25),
            "Q3_finite_midlifes": np.percentile(midlifes, 75), 
            "Q3_finite_lifespans": np.percentile(lifespans, 75),
            "IQR_finite_midlifes": iqr(midlifes), 
            "IQR_finite_lifespans": iqr(lifespans),
            "entropy_finite_lifespans": -np.sum((lifespans / sum_lifespans) * np.log(lifespans / sum_lifespans))}
