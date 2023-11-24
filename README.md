# TM DataEnhancer

The Data Enhancer concept is to take the learnt clauses, with the highest weights, from the first Tsetlin Machine and find their fitting convolutional windows in all images. Then map all other convolutional windows to that most important, and hence enhance the data.

The concept should tacke directly the issue of an object moving around in an image, by centering on its most important part.



## Method:
We select important convolution windows with the first Tsetlin Machine and find them in the original images:

![Matching Patches](https://github.com/vHalenka/TM_DataEnhancer/assets/148200081/53542f39-3992-443c-9ae8-3176262b4946)

Then we take the convolutional window with the highest weight and select it as an Anchor, and relate all other convolutional windows to it.
![img to Anchor](https://github.com/vHalenka/TM_DataEnhancer/assets/148200081/537abf09-62ed-483d-87d0-c8e6fc5eb3d3)




## Test:

To show the main benefit of this method, a customary dataset has been created:

![Dice set](https://github.com/vHalenka/TM_DataEnhancer/assets/148200081/03f35d25-3c46-41ce-971c-2684797ac74a)

Where a dice is randomly placed in an image.

### 1st Test:

