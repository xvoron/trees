# Trees Detection from Image

## Goal:
Detect trees from image and calculate number of trees. Available only RGB
images.

### Image information
![img](doc/trees_rgb.jpg)
- **parameters:** 5000x5000 px, RGB


## Solutions

1. [Local maxima filtering](trees_local_maxima.py): Done
2. [Template matching](trees_template_match.py): Not working
3. [Watershed algorithm](trees_watershed.py): TODO
4. [Deepforest library](df_trees.py): Not good

### Local Maxima outcome
![img](doc/trees_detected.jpg)

## Sources
- https://www.intechopen.com/chapters/49851
- https://realpython.com/python-opencv-color-spaces/
- https://hal.archives-ouvertes.fr/search/index/?q=guillaume+perrin+xavier+descombes
- https://github.com/weecology/DeepForest
