import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom as dicom
import os
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2 as cv
import vtk
from vtk.util import numpy_support

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# take folder of a single lung and preprocess and segments the lungs



INPUT_FOLDER = '../../dataset/manifest-1674660649243'
NLST = f'{INPUT_FOLDER}/NLST flatten'

expFolder = f'{NLST}/100004/01-02-2001-NA-NLST-LSS-01308/1.000000-2OPATOAQUL4C410.22-06725'
cases_root_folder = f'{NLST}/100004/01-02-2001-NA-NLST-LSS-01308/'
# an array including path of all the scan folders
cases_path = [cases_root_folder + fileName for fileName in os.listdir(cases_root_folder)]

class preprocess:
    def __init__(self, input_folder, dilation_kernel_size = 2, dilation_iteration = 2):
        self.dilation_kernel_size = dilation_kernel_size
        self.dilation_iteration = dilation_iteration
        self.input_folder = input_folder
        self.loaded_scans = self.load_scan(self.input_folder)
        self.scan_HU_values = self.get_pixels_hu(self.loaded_scans)
        self.resampled_3d_image, self.spacing = self.resample(self.scan_HU_values, self.loaded_scans, [1,1,1])
        self.segmented_lung_mask = self.segment_lung_mask(self.resampled_3d_image, True, dilation = '3d', kernel_size = self.dilation_kernel_size, iteration = self.dilation_iteration)
        self.segmented_grayscale_scan = self.hu_transform(self.resampled_3d_image, self.segmented_lung_mask, lung_window= [-1200, 600])

    def hu_transform(self, scan_HU_value, mask, lung_window = [-1200, 600]):
        
        mask = mask.astype(np.int16)
        
        scaled_scan = (scan_HU_value-lung_window[0])/(lung_window[1]-lung_window[0])
        scaled_scan[scaled_scan<0]=0
        scaled_scan[scaled_scan>1]=1

        trans_scaled_scan = (scaled_scan*255).astype('uint8')
        segmented_trans_scan = mask * trans_scaled_scan
        
        return segmented_trans_scan
        

    def load_scan(self, path):
        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(
                slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(
                slices[0].SliceLocation - slices[1].SliceLocation)

        for s in slices:
            s.SliceThickness = slice_thickness

        return slices


    def get_pixels_hu(self, slices):
        image = np.stack([s.pixel_array for s in slices])
        # Convert to int16 (from sometimes int16),
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)

        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        image[image == -2000] = 0

        # Convert to Hounsfield units (HU)
        for slice_number in range(len(slices)):

            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope

            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)

            image[slice_number] += np.int16(intercept)

        return np.array(image, dtype=np.int16)

    def resample(self, image, scan, new_spacing=[1, 1, 1]):
        # Determine current pixel spacing
        spacing = np.array([scan[0].SliceThickness] +
                           list(scan[0].PixelSpacing), dtype=np.float32)

        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor

        image = ndimage.zoom(image, real_resize_factor, mode='nearest')

        return image, new_spacing


    def largest_label_volume(self, im, bg=-1):
        vals, counts = np.unique(im, return_counts=True)

        counts = counts[vals != bg]
        vals = vals[vals != bg]

        if len(counts) > 0:
            return vals[np.argmax(counts)]
        else:
            return None



    def dilate(self, img, kernel_size, iteration, three_d=False, shape = 'circular'):

        img = img.copy()
        if three_d:
            if shape == 'circular' :
                kernel = morphology.ball(kernel_size).astype(np.uint8)
            elif shape == 'square' :
                kernel = np.ones((kernel_size, kernel_size, kernel_size), np.uint8)
            img = ndimage.binary_dilation(img, kernel, iterations=iteration)
        else:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            for i, slide in enumerate(img):
                img[i] = ndimage.binary_dilation(
                    slide, kernel, iterations=iteration)

        return img


    def segment_lung_mask(self, image, fill_lung_structures=True, dilation='3d', kernel_size=3, iteration=5):

        # 1 in dilate does 2d dilation and 2 does 3d
        # not actually binary, but 1 and 2.
        # 0 is treated as background, which we do not want
        binary_image = np.array(image > -320, dtype=np.int8)+1
        # plot_3d(binary_image - 1, 0, 'thresholding')
        labels = measure.label(binary_image)

        # Pick the pixel in the very corner to determine which label is air.
        #   Improvement: Pick multiple background labels from around the patient
        #   More resistant to "trays" on which the patient lays cutting the air
        #   around the person in half
        background_labels = labels[0:2, 0:2, 0:2]
        background_label = np.median(background_labels)

        # Fill the air around the person
        binary_image[background_label == labels] = 2

        # Method of filling the lung structures (that is superior to something like
        # morphological closing)
        if fill_lung_structures:
            # For every slice we determine the largest solid structure
            for i, axial_slice in enumerate(binary_image):
                axial_slice = axial_slice - 1
                labeling = measure.label(axial_slice)
                l_max = self.largest_label_volume(labeling, bg=0)

                if l_max is not None:  # This slice contains some lung
                    binary_image[i][labeling != l_max] = 1

        binary_image -= 1  # Make the image actual binary
        binary_image = 1-binary_image  # Invert it, lungs are now 1

        # Remove other air pockets insided body
        labels = measure.label(binary_image, background=0)
        l_max = self.largest_label_volume(labels, bg=0)
        if l_max is not None:  # There are air pockets
            binary_image[labels != l_max] = 0

        if dilation == '2d':
            binary_image = self.dilate(binary_image, kernel_size, iteration)
        elif dilation == '3d':
            binary_image = self.dilate(binary_image, kernel_size, iteration, True)

        return binary_image


#scan = preprocess(cases_path[0])
#print(scan.segmented_lung_mask)
# print(scan.segmented_lung_mask)


class vtk_visualization:

    colors = vtk.vtkNamedColors()

    def __init__(self, structure):
        self.structure = structure 
        self.imData = self.numpyToImageDate(structure)
        self.volumeMapper, self.volumeProperty = self.imDataToVolumeMapper(self.imData)
        self.vtkVolume = vtk.vtkVolume()
        self.volume = self.vMapperToVolume(self.vtkVolume, self.volumeMapper, self.volumeProperty)
        self.ren = self.renderVol(self.volume)

    def numpyToImageDate(self, numpyScan):
        scan = numpyScan.transpose(2,0,1)
        imData = vtk.vtkImageData()
        vtk_data = numpy_support.numpy_to_vtk(scan.ravel(order= 'F'), deep=True, array_type=vtk.VTK_DOUBLE)

        imData.SetDimensions(scan.shape)
        imData.SetSpacing([.1,.1,.1])
        imData.SetOrigin([0,0,0])
        imData.GetPointData().SetScalars(vtk_data)

        return imData


    def imDataToVolumeMapper(self, imData ):
        
    
        opacity = vtk.vtkPiecewiseFunction()
        opacity.AddPoint(0, 0)
        opacity.AddPoint(1, .5)
    

        color = vtk.vtkColorTransferFunction()
        r, g, b = vtk_visualization.colors.GetColor3ub('black')
        color.AddRGBPoint(0, r, g, b)
        r, g, b = vtk_visualization.colors.GetColor3ub('white')
        color.AddRGBPoint(255, r, g, b)

        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(color)
        volumeProperty.SetScalarOpacity(opacity)
        volumeProperty.SetInterpolationTypeToLinear()
        volumeProperty.SetIndependentComponents(2)


        volumeMapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
        volumeMapper.SetInputData(imData)
        #volumeMapper.SetBlendModeToAverageIntensity()
        return volumeMapper, volumeProperty 


    def vMapperToVolume(self, volume, mapper, property):
        volume.SetMapper(mapper)
        volume.SetProperty(property)
        return volume


    def renderVol(self, volume):

        ren = vtk.vtkRenderer()
        ren.AddVolume(volume)
        ren.SetBackground(vtk_visualization.colors.GetColor3ub('black'))
        return ren

    def visualize(self, ren):

        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(ren)
        render_window.SetSize(400, 400)


        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)
        render_window_interactor.Initialize()

        ren.ResetCamera()
        ren.GetActiveCamera().Zoom(1.5)

        render_window.Render()
        render_window_interactor.Start()

scan = preprocess(cases_path[1])

#print(scan.hu_trans_segmented)
#print(f'{type(scan.resampled_3d_image)} {scan.resampled_3d_image.dtype} {scan.resampled_3d_image.dtype.type} shape of hu values : {scan.resampled_3d_image.shape}')
#print(f'{type(scan.segmented_lung_mask)} {scan.segmented_lung_mask.dtype} {scan.segmented_lung_mask.dtype.type} shape of mask : {scan.segmented_lung_mask.shape} {scan.segmented_lung_mask.astype(np.int16).dtype}')



#print(scan.resampled_3d_image.astype(np.int16))
scan_vis = vtk_visualization(scan.segmented_grayscale_scan)
scan_vis.visualize(scan_vis.ren)