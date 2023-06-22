# Lung cancer’s malignant nodule visualization and detection using CNN
visualizing potential cancerous nodules in ct scans and predicting lung cancer using CNN

## installation and setup

1. Download CUDA toolkit
2. run the installer
3. set Environment Variables
4. install PyTorch
5. run :
  ```
  python3 -m pip install -r requirements.txt
  ```

## Description


Lung cancer is a significant cause of cancer-related deaths worldwide, and accurate and timely detection is crucial for successful treatment outcomes. The study aims to explore the potential of CNNs in detecting and visualizing lung cancer. CT scans of the lungs serve as input data for the CNN model, and I train it to identify cancerous regions. I utilize Grad-CAM to visualize these regions and provide insights into the model's decision-making process. Our findings show that the proposed approach can accurately detect cancerous regions in lung CT scans with high precision and recall rates. Moreover, the visualization of these regions using Grad-CAM can provide valuable insights into the areas of the lungs that may require further investigation and treatment. This research highlights the potential of CNNs and Grad- CAM in detecting and visualizing lung cancer, improving the accuracy of lung cancer detection, and leading to timely treatment and better patient outcomes. The proposed approach serves as a basis for further research in developing automated lung cancer detection systems using deep learning techniques.


### Dataset

1. NLST
2. Lung Nodule Analysis (LUNA16) 
3. The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI) 

### Preprocessing

The overall preprocessing procedure is brought here step by step:

Thresholding: CT scans are saved as DICOM files, which contain Hounsfield Units (HU) that represent the density of tissues in the body [11]. To segment the lungs, we first need to extract the HU values that correspond to the lungs. This is done by applying a thresholdto the HU values, which essentiallysets a cutoff point for the values that we consider part of the lungs. In this case, a threshold of -350 HU was used, which is a good value for this approach.

Connected Component Analysis: After thresholding, the image is binarized, which means that we have a binary image where each pixel is either 0 or 1. The next step is to identify the
 
air around the person in the scan, which is done by applying connected component analysis to the binary image. This essentially groups together the pixels that are connected to each other and assigns a label to each group. The label that corresponds to the air around the person is determined, and all pixels with that label are filled with 1s in the binary image.

Keep Only the Largest Air Pocket: After the lungs are segmented, we want to keep only the largest air pocket, which correspondsto the lungs themselves. The humanbodyhas other pockets of air, such as the stomach and intestines, which we want to exclude from the lung mask. This is done by keeping only the largest connected component in the binary image.

Dilation: Most nodulesin the lungsare located nearthe body of the lungs. To make sure that these nodules are included in the lung mask, dilation operations are used. Dilationessentially expands the lung mask by a few pixels in all directions, which helps to include any nodules that may be near the edge of the lung.

Zero Centering and Normalization: After segmentation, we want to preprocess the lung images to make them easier for the CNN to process. This is done by zero centering and normalization. Zero centering means that we subtract the mean value of the HU values from all pixels in the image. Normalization means that we divide all pixel values by the standard deviation of the HU values. This helps to standardize the range of pixel values in the image and makes it easier for the CNN to convolve and process the images.

Changing the Range of Values: Lastly, for visualization purposes, we change the range of all values from 0 to 255. This is done so that the lung images can be displayed as grayscale images with a clear contrast between the lung tissue and the surrounding air

### Visualization 

For visualizingthe lungs, I used the VTK library for volumetric rendering. Volumetric renderingis a technique that allows usto visualize a 3D object by rendering it as a collection of small volume elements or voxels. Unlike other rendering techniques like marching cubes or isosurfaces, volumetric rendering can show the intensity of the object, which can provide useful information about the model. I wrote a code that takes in the structure of the lungs and converts it into image data using the numpy library. This image data is then used to create a volume mapper and a volume property using the VTK library. The volume mapper is responsible for mapping the image data onto the volume and the volume property defines the appearance of the volume. Next, I created a VTK volume using those and that was then rendered using the render volume function from the VTK library. Ichose volumetric rendering over other rendering techniques because it allows us to see the interior of the lungs and provides more information about the intensity of the cancerousareas. Thisinformation can be valuablein diagnosing and treating lung cancer. Additionally, volumetric rendering is computationally efficient and works well with large datasets, making it a good choice for visualizing medical imaging data.

Transfer function plays a crucial role in the visualization of lungs using volume rendering techniques. It allows the user to map scalar values to color and opacity, which is essential for revealing important information about the internal structures of the lungs. In this research, I used different transfer functions to enhance the visibility of the inside structures of the lungs and nodules, making them more distinguishable to the human eye. One of the transfer functions I used was the gradient transfer function, which assigns higher intensity values to regions with a higher gradient. This allowed me to visualize the edges of nodules more clearly, making them stand out from the surrounding lung tissue. The transfer function also had a part in differentiating between different types of nodules, as they can have varying gradients. In addition, I used other transfer functions to adjust the color and opacity of different regions of the lungs, depending on their intensity values. By tweaking these functions, I was able to create a visual representation of the lungs that highlighted the areas of interest, such as the nodules, while still providing context for their location within the lungtissue. Belowyoucansee a visualizaiont oflungsusing vtk library.


In addition to the gradient transfer function, other transfer functions were used to adjust the color and opacity of different regions of the lungs, depending on their intensity values. Particularly, a novel approach was implemented to visualize lungs by analyzing the intensity diagram of the lung structure. This intensity diagram was observed to exhibit two distinct Gaussian-like distributions (Fig. 2). The first distribution was observed to be present in the intensity range of 5 to 80, whereas the second distribution spanned from 120 to 200. The experimental findings suggested that the first distribution primarily corresponded to the intensity values that were irrelevant to the lung structure, while the latter distribution was mainly associated with the intensity values of nodules present in the lungs. To enhance the visualization of the nodules and distinguish them from the surrounding lung tissue, Otsu's thresholding method was utilized. Otsu's thresholding method is an image segmentation technique that determines the optimal threshold value to separate the foreground (nodules) from the background (lung tissue) based on the histogram of intensity values [20]. The threshold value was then utilized in a linear transfer function to make the internal structure of the lungs, corresponding to the first distribution, more transparent. By doing so, the second distribution, which corresponded to the nodules, became more opaque, thus emphasizing their visibility.

<div align=”center”>
  
![Histogram ](https://github.com/Dev-Erf/Lung-Cancer-Prediction-and-visualization/assets/85780796/2b59f535-6d76-47d4-8f0f-07e571d59f70)

</div> 

Otsu's thresholding method is a commonly used algorithm in image processing for determining the optimal threshold value for image segmentation. The algorithm works by finding the threshold that minimizes the variance between two classes of pixels, which are separated by the threshold. In the case of the intensity diagram of lungs, the algorithm was used to separate the two Gaussian-like distributions. This method allowed for greater visualization of the internal structures of the lungs, particularly the nodules, which are often difficult to detect due to their similarity in appearance to surrounding lung tissue. By utilizing Otsu's thresholding method, the research was able to distinguish between the foreground and background of the image, making it easier to identify the nodules.


![ezgif com-video-to-gif(1)](https://github.com/Dev-Erf/Lung-Cancer-Prediction-and-visualization/assets/85780796/49d39307-a2c1-4f54-984f-a2f0913b04c1)


### Model Architechture

In this study, the primary objective was to visualize cancerous parts of the lungs using CT scans. For this purpose, a 3D convolutional neural network (CNN) architecture was used. CNNs are commonly used for image classification tasks and are known to perform well on tasks such as image recognition,	segmentation,	and	detection.	The	model architecture consisted of three 3D convolutional layers, which were connected to three fully connected layers. Due to processing power limitations, the number of layers was kept relatively low. This decision was made keeping in mind that,for the purpose of visualizing cancerous areas in the lungs, a high accuracy rate is not necessarily the most important factor. Even models with lower accuracy rates, such as those between 70% to 80%, can still provide good results in visualizing cancerous areas in the lungs. To improve the generalization performance ofthemodel, poolinganddropout layerswere also incorporated into the architecture. Pooling layers help in reducing the spatial dimensions of the output volume of the previous layer. This helps to reduce the number of parameters and hence, the processing power required. Dropout layers, on the other hand, help in preventing overfitting of the model to the training data. Overfitting is a common problem in deep learning where the model becomes too specialized in recognizing the training data, but performs poorly on new data. To train the model, binary cross-entropy (BCE) loss function was used [12]. BCE loss is a commonly used loss function for binary classification tasks and is known to perform well compared to other loss functions such as cross-entropy.below you can see the area under curve of the architecture.


### Visualizing the Potential Cancerous Areas 


Now that we have trained the model and have its weights, we can use Grad-CAM (Gradient-weighted Class Activation Mapping) [2] to visualize the cancerous parts in the lungs. Grad-CAM is a technique for generating heatmaps that highlight the regions of an image that were important for the classification decision made by a neural network. It is an example of an "explanatory" and "interpretable" AI technique, because it helps us understand how the network is making its decisions. In the medical field, interpretability is crucial for understanding how the AI system arrives at its decision, especially for clinical decision making. For instance, in lung cancer detection, explainable AI can help radiologists and physicians to see which regions of the lung are contributing to the model's decision, andthushelp to confirm orchallenge their initial diagnosis. To use Grad-CAM in our project, we use the last convolutional layer of our 3D CNN. We take the output of this layer, which is a tensor of shape (batch_size, num_filters, conv_output_width, conv_output_height, conv_output_depth), and calculate the gradient of the output with respect to the final

 
layer of the network. This gives us a tensor of shape (batch_size, num_filters), where each element corresponds to the gradient of the output with respect to the activation of a particular filter. we then perform a global average pooling operation on this tensor, resulting in a vector of length num_filters. We multiply this vector by the weights of the final layer, which gives us a weighted sum of the activations of each filter. This weighted sum represents the importance of each filter for the classification decision. Finally, we calculate the gradient of this weighted sum with respect to the activations of the last convolutional layer. This gives us a tensor of shape (batch_size,	num_filters,	conv_output_width, conv_output_height, conv_output_depth), which we use to generate a heatmap that highlights the important regions of the input image.

To make the heatmap more interpretable, we rescale it to the size of the original input image. We then apply a color map to the heatmap to make the cancerous regions more visible. We add this heatmap to the same volume rendering of the lungs, allowing us to visualize the cancerous regions in the context of the entire lung. Below you can see the final result.

![Screen_Recording_2023-04-05_at_02 03 11](https://github.com/Dev-Erf/Lung-Cancer-Prediction-and-visualization/assets/85780796/72d4c8af-0d9e-4682-a9c7-e1404fca1b61)


### Future Work

There is an area that has yet to be explored in the field of lung cancer detection using CT scans. Specifically, this tool can be utilized to investigate any patterns in the lungs preceding lung cancer. Typically, these patterns are hidden from radiologists, but neural network models and pattern recognition could be used to uncover these hidden patterns. By training a neural network to recognize these patterns, potentially can be detected lung cancer earlier and improve the chances of successful treatment. This could have a significant impact on patient outcomes and could potentially save lives. Additionally, this approach could lead to the development of new screening methods for lung cancer, which could help to reduce the overall incidence of the disease. There is still much work to be done in this area, but the potential benefits are significant.



## Support 


## Licence 

## Inspired By 

