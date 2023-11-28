# TM DataEnhancer

## Introduction
TM DataEnhancer utilizes Tsetlin Machines to improve data quality by focusing on significant clauses and their corresponding convolutional windows in images. This approach aims to address challenges such as tracking moving objects in images by concentrating on their key features.

## Methodology
- **Selection of Convolution Windows**: The first Tsetlin Machine identifies important convolution windows from the original images.

  ![Matching Patches](https://github.com/vHalenka/TM_DataEnhancer/assets/148200081/53542f39-3992-443c-9ae8-3176262b4946)

- **Anchor Window Identification**: The convolutional window with the highest weight is chosen as the 'Anchor'. All other windows are then related to this anchor to enhance the data.

  ![img to Anchor](https://github.com/vHalenka/TM_DataEnhancer/assets/148200081/537abf09-62ed-483d-87d0-c8e6fc5eb3d3)

## Testing
I used a custom dataset where dice are placed in random positions in images to test the effectiveness of this method.
![image](https://github.com/vHalenka/TM_DataEnhancer/assets/148200081/79924ca6-fd41-40b5-a8a7-9bbcd6bd4be5)

### Test Results
- **First Test**: Displayed noticeable improvement in data quality.

  <img width="1131" alt="image" src="https://github.com/vHalenka/TM_DataEnhancer/assets/148200081/52078f01-26ff-413d-b496-495a6ed36979">

- **Second Test**: Confirmed the consistency of the improvement on a very fast and low effort run on the complete dataset.

  ![bilde](https://github.com/vHalenka/TM_DataEnhancer/assets/148200081/ff8ae0f6-09e6-4348-bf29-2ad81a131872)

- **Third Test**:

![image](https://github.com/vHalenka/TM_DataEnhancer/assets/148200081/7af940ba-e28e-49fc-835b-c26bfc3da678)

This behaviour can be explained by the essence of the dataset and property of this approach. The current approach finds all matches of a particular patch and notes their average coordinate as the anchor point. Meaning, in the Enhancers view, the anchors are mostly put into exactly the same position, the middle of the die. What is strange
