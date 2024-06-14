# DermAi (Skincare Recommendation Systems)

Lately, there's been a lot of buzz around recommender systems, which are crucial in both business and academic circles. One area that's been in the spotlight is the 'facial skincare recommendation system.' This system essentially acts as a personalized skincare advisor, helping people pick the right facial products based on their specific skin type, issues, and preferences. However, existing models, whether they use collaborative or content-based filtering, haven't quite hit the mark in addressing all user concerns. To bridge this gap, there's a new approach on the horizon: a hybrid method that combines different techniques like KNN, CNN, Transfer Learning of EfficientNet B0, and content-based filtering. By taking into account user inputs such as skin tone, type, and acne severity, the algorithm aims to provide tailored product recommendations. Tests have shown that this hybrid model outperforms others, offering a more comprehensive and personalized skincare solution. With the incorporation of EfficientNet B0, the model achieves impressive validation and training accuracies, promising even better precision and efficiency in skincare recommendations.

# Architecture Diagram
![image](https://github.com/vinit714/A-Recommendation-system-for-Facial-Skin-Care-using-Machine-Learning-Models/assets/52816788/6971ee9a-4108-43bd-bed7-1687422baecb)

The system architecture diagram serves as a visual representation of how the different components of our application work together. It provides a clear overview of how the system operates, making it easier to understand how each component connects and contributes to the overall functionality.

In our application's architecture, when a user interacts with the system, their facial image is captured. This image is then processed by the model, and combined with additional inputs provided by the user, such as skin type and concerns, to generate personalized product recommendations.

From a software engineering perspective, it's crucial to consider design specifics, particularly regarding the user interface. The graphical user interface (GUI) plays a vital role in ensuring customer satisfaction, as its transparency directly impacts user experience. Additionally, our application is built as a web application, ensuring accessibility across various platforms and devices, thereby making it available to users virtually anywhere. This foundational choice enhances the application's usability and reach.

# ML Models
The proposed model serves the purpose of evaluating acne severity as well as determining skin type and tone. To achieve this, it incorporates several established algorithms such as K-Means and EfficientNet, along with a content-based recommendation model, ensuring the delivery of accurate results. Below, we break down the individual modules and approaches encompassed within the model.
## Recommendation for Skin Tone
Determining skin tone involves the initial step of identifying and isolating the pixels constituting the skin. Subsequently, these pixels are categorized into respective skin tone groups based on their color values. The process of skin detection comprises three fundamental operations: initial segmentation, prediction of skin pixels, and k-means clustering.

![image](https://github.com/vinit714/A-Recommendation-system-for-Facial-Skin-Care-using-Machine-Learning-Models/assets/52816788/c9c3f04f-169f-4d04-a93f-f7ed96e765c9)

The initial segmentation process relies on a threshold value, which is calculated as the average of TOTSU and TMAX. These values are derived from the histogram of the grayscale image.
## Recommendation according to Skin Type
For skin type classification, the image undergoes analysis and categorization using convolutional neural networks (CNN). This categorization divides facial skin into standard, oily, and dry types. By employing transfer learning with EfficientNet B0, the model aims to enhance its accuracy. Currently, the model achieves a training accuracy of 87.10% and a validation accuracy of 80%.

![image](https://github.com/vinit714/A-Recommendation-system-for-Facial-Skin-Care-using-Machine-Learning-Models/assets/52816788/e9f51e82-7cda-4b02-aabe-50a0d795effd)

The table above illustrates the total number of layers incorporated into the EfficientNet-B0 architecture. Images with a resolution of 224 by 224 pixels can seamlessly be fed into the network. The term "MBConv" denotes a depth-wise separable convolution layer featuring an inverted linear bottleneck. In the equation below, where x represents the input image, f1 through f7 denote the layers of the neural network, and y signifies the output classification label or probability distribution across classes, the EfficientNet-B0 design is encapsulated in the following formula.

$𝑦 = 𝑓7(𝑓6(𝑓5(𝑓4(𝑓3(𝑓2(𝑓1(𝑥)))))))$

## Acne Severity
The skin metric referred to as the "acne concern level" is categorized into three levels: Low, Moderate, and Severe. Through the utilization of transfer learning within the model's framework, an accuracy of 68% has been attained across both the training and validation image datasets. Similar to the Skin Types CNN model, this model's architecture is structured akin to the primary EfficientNet-B0 network. It incorporates MobileNetV2 inverted bottleneck residual blocks along with squeeze-and-excitation blocks.

## Working of the Proposed Recommender System
In order to recommend products tailored to the user's skin features, the model relies on identifying the top similarity values between the user's skin vector and the product vector within the dataset. This process is depicted in the figure. By searching for products whose features align closely with the user's skin measurements and concerns, the system can make intelligent recommendations. The automated cosine similarity between the user's skin attribute vector and the product feature vector serves as a measure of likeness, allowing the system to effectively match products to the user's specific needs and preferences.

![image](https://github.com/vinit714/A-Recommendation-system-for-Facial-Skin-Care-using-Machine-Learning-Models/assets/52816788/a95ff28c-8e8f-4fd8-aee6-283c5185d89c)

# EXPERIMENTAL RESULTS
Below are six categorized lists of skin tones: Fair skin is described as having 
+ Light eyes and hair, is easily burned, and rarely tans
+ Brown or hazel eyes, light brown hair, and skin that is light to medium in tone with occasional burning but potential for gradual tanning
+ Brown eyes and dark brown hair are complemented by skin that is medium to olive in tone, rarely burns, and tans easily
+ Brown eyes and black hair complement dark brown complexion, which rarely burns and tans quickly
+ Black eyes and black hair complement the deeply pigmented brown complexion, which never burns and tans easily
+ Black skin: Skin that tans readily, never burns, and has black hair and eyes

![Efficient_Net-based_Expert_System_for_Personalized_Facial_Skincare_Recommendations (1)](https://github.com/vinit714/A-Recommendation-system-for-Facial-Skin-Care-using-Machine-Learning-Models/assets/52816788/bd53c8a3-3646-4a79-aa31-9cf36b3a0089)

The data training process is conducted using the Python programming language along with TensorFlow library (version 8). For plotting and other data processing tasks, MATLAB is utilized. 

To determine the skin type of individuals, measurements are performed at precise locations while gently moving the sensor head around the region. Subsequently, a Convolutional Neural Network (CNN) analysis is employed, dividing the facial image into three categories: standard, oily, and dry. The model's accuracy is enhanced through transfer learning with EfficientNet B0, currently achieving an accuracy of 87.10% compared to a validation accuracy of 80%. 

EfficientNet's capabilities, as outlined in Table II, make it an ideal choice for precise categorization of the skin spectrum. To establish the superiority of the proposed model over existing methods, a comparison is conducted between it and other methodologies.

![Efficient_Net-based_Expert_System_for_Personalized_Facial_Skincare_Recommendations (1)](https://github.com/vinit714/A-Recommendation-system-for-Facial-Skin-Care-using-Machine-Learning-Models/assets/52816788/60eaa9ca-a701-4580-8ab1-0fb35c863a6d)

The metric concerning skin, termed the "acne concern level," is segmented into three levels: Low, Moderate, and Severe. Despite these categorical labels, it's deemed suitable to represent them with conventional numeric values for consistency: 0 - No Acne, 1 - Clear, 2 - Almost Clear, 3 - Mild, 4 - Moderate, and 5 - Severe. This numeric representation facilitates uniformity and clarity in interpreting and analyzing the acne severity levels.

![image](https://github.com/vinit714/A-Recommendation-system-for-Facial-Skin-Care-using-Machine-Learning-Models/assets/52816788/4e595c22-c4cd-4b5d-9f96-0626f14d386f)

The figure illustrates the distribution of images categorized into various groups based on their acne severity levels. There's an observed imbalance in the distribution of image classes, with Class 3, representing Mild acne, being predominant. Both the training and testing datasets comprise images with different severity levels of acne. However, this endeavor was challenging due to the presence of noisy picture labels provided by dermatologists. Additionally, it was noted that the training image dataset contained numerous identical or nearly identical images, further complicating the training process.
