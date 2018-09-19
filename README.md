# PAC 2018 competition model - Team utastoot

Attempt to detect major depression using sMRI data.

### About

This is a tidied-up version of the source code of our competition model in
[PAC 2018](https://www.photon-ai.com/pac) predictive analytics competition.

Our model achieved 64% accuracy on an unseen test set and the shared second
position in the competition (:tada:).

The version history has been removed since our original Git repository contained
Jupyter notebooks showing parts of the competition dataset, which we are not
allowed to publish. However, this version of the code should reproduce our final
submission if trained and evaluated with the competition data.

Team "utastoot" consists of Satu Immonen and Otto Seiskari.

### The model

Our model reduces the sMRI data to two continuous features computed as the sums
of the voxel values in two areas defined by thresholding voxel-wise t-test
scores. Then we use **logistic regression** on these features combined with age
and gender.

In more detail, the feature extraction proceeds as follows

 1. Pre-process:
    * Crop to remove (almost) always empty edge regions
    * GMV normalization: divide voxel values by the mean in each image
    * Down-sampling: partitioning to blocks of size 2x2x2 and take the mean

 2. Compute per-voxel t-test scores between the depressed and control groups.

 3. Threshold the resulting "t-brain image" with two different values, which
    yields two boolean 3D voxel images, masks, defining areas that seem
    to be associated with depression.

 4. Construct two new continuous features for each subject as the inner
    product between the (preprocessed) voxel data and each mask, that is,
    the sum of the voxel values in the masked areas.

### Setup

Install all dependencies:

    pip3 install -r requirements.txt

Tested with Python 2.7.13 and the packages listed in `pip-freeze.txt`.

### Cross-validation

    python src/run_cross_validation.py /path/to/PAC2018_Covariates_Upload.csv

The training meta data file needs to be converted from XLSX to CSV. It should
begin with the header `PAC_ID,Label,Age,Gender,TIV`. The training brain images
`PAC2018_DDDD.nii` are assumed to be in the same folder with the CSV file.

### Train and classify test data

This is expected to take about 3 minutes if the data is stored on an SSD.

    python src/classify_test_data.py \
      /path/to/PAC2018_Covariates_Upload.csv \
      /path/to/PAC2018_Covariates_Testset.csv \
      answers.csv

The training and test meta data files need to be converted from XLSX to CSV.
The training data `PAC2018_DDDD.nii` files are assumed to be in the same folder
with `PAC2018_Covariates_Upload.csv` and the test data files should be in the
same folder with `PAC2018_Covariates_Testset.csv`. The latter file should
begin with the header `PAC_ID,Scanner,Age,Gender,TIV`.
