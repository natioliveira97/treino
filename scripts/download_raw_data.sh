#############################################
# Created by: Anshuman Dewangan
# Date: 2021
#
# Description: Downloads latest raw wildfire images from HPWREN website
#############################################

# wget -r -nH --cut-dirs=2 --no-parent --reject="index.html*" http://hpwren.ucsd.edu/HPWREN-FIgLib/HPWREN-FIgLib-Data/ --directory-prefix=data/new_data/raw_images/

wget -r -nH --cut-dirs=2 --no-parent --reject="index.html*" http://hpwren.ucsd.edu/HPWREN-FIgLib/Labels/HPWREN-BB/VOC/ --directory-prefix=data/new_data/labels/