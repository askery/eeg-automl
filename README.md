# Automated Machine Learning for drowsiness detection with EEG signals

In this repository, we make available the code and best models associated with the manuscript `Automated Machine Learning for drowsiness detection with EEG signals` 

```diff
- ((TO-DO: arxiv link))
```

The repository is organized as follows:

```
.
├── data/
├── best_models/
│   ├── C4_1_FE_RD_tpot_pipeline_0.9536.py
│   ├── xxx-5
│   ├── C4_6_FE_RD_tpot_pipeline_acc_0.9992.py
│   ├── xxx-8
│   ├── xxx-10
│   ├── xxx-all
├── autokeras_experiments/
│   ├── exordial_TS_features_autoKeras.ipynb
│   ├── v1_TS_rawdata_autoKeras.ipynb
├── tpot_experiments/
│   ├── exordial_TS_features_TPOT.ipynb
│   ├── v1_TS_rawdata_TPOT.ipynb
├── utils/
│   ├── classical_MLP.py
│   ├── data_parser_and_brief_EDA.ipynb
│   ├── feature_importance.ipynb
├── confusion_matrices_mlp/
├── LICENSE
└── README.md
```

```diff
- ((TO-DO: edit with the actual names of the files/folders for the autokeras/tf models))
```
______________

In the folder `data` you can find all the data used in the study, in `xlsx`, `csv` and `parquet` formats. Each file contains data corresponding to one of the 5 subjects used in the study (those which alternate the classification of alert to drowsy from the first PVT to the third PVT - i.e., subjects 1, 5, 6, 8 and 10). As discussed in the manuscript, we only used the data from the C4 EEG channel, sampled at 512 Hz. Thus, all file names start with `C4_ID`, where `ID` identifies the corresponding subject. For each subject, we have 3 different types of data, according to the respective indication in the data file name:

- `rawdata`: This is the rawdata, containing the EEG measurements of the C4 channel from a 100-point EEG time window. Each row corresponds to ~0.2 seconds of data, given the sampling frequency of 512 Hz. Each set of 100 points is represented in a single column, compressed in a list-like structure. There is also a column for the target label (0 or 1);

- `rawdata_open`:` This contains the same raw data described above, but the data points are each one open into 100 columns, which is more adequate to serve as input to the machine learning pipelines. These datasets are produced by the code in the notebook `data_parser_and_brief_EDA.ipynb`;

- `features`: This contains the 3 features calculated from the EEG raw data, as discussed in Sec. III-B of the manuscript.

```diff
- ((TO-DO: please check if the data description is accurate, and if the data in the repo is indeed the data used in the experiments!!!))
```
______________

In the folder `best_models` you find the model objects for each one of the best models found for each subject, as highlighted in Table II of the manuscript: `TABLE II: MLP and AutoML results, in %; RD: raw data, F: features; best performance per subject highlighted in gray`. The file names indicate the subject, input data arrangement, type of AutoML solution, and the respective model accuracy:

- `C4_1_FE_RD_tpot_pipeline_0.9536.py`: best model for subject 1: pipeline using TPOT, yielding 95.36% accuracy;

- `xxx-5`: best model for subject 5: tensorflow model built using AutoKeras, yielding 83.23% accuracy;

- `C4_6_FE_RD_tpot_pipeline_acc_0.9992.py`: best model for subject 6: pipeline using TPOT, yielding 99.92% accuracy;

- `xxx-8`: best model for subject 8: tensorflow model built using AutoKeras, yielding 95.35% accuracy;

- `xxx-10`: best model for subject 10: tensorflow model built using AutoKeras, yielding 98.37% accuracy;

- `xxx-all`: best generalist model: tensorflow model built using AutoKeras, yielding 82.54% accuracy.

```diff
- ((TO-DO: edit with the actual names of the files/folders for the autokeras/tf models)).
```
______________

In the folder `autokeras_experiments` you find notebooks with the code used in the AutoKeras experiments:

- `v1_TS_rawdata_autoKeras.ipynb`: AutoKeras experiments using raw data (RD);

- `exordial_TS_features_autoKeras.ipynb`: AutoKeras experiments using feature vectors.

______________

In the folder `tpot_experiments` you find notebooks with the code used in the TPOT experiments:

- `v1_TS_rawdata_TPOT.ipynb`: TPOT experiments using raw data (RD);

- `exordial_TS_features_TPOT.ipynb`: TPOT experiments using feature vectors.

______________

In the folder `utils` you can find some ancillary code, namely:

- `classical_MLP.py`: code for the classical MLP models, used as benchmark;

- `data_parser_and_brief_EDA.ipynb`: notebook with the code used for parsing the data into the final formats, and a brief exploratory data analysis;

- `feature_importance.ipynb`: notebook with feature importance analysis for the feature vectors.

______________

In the folder `confusion_matrices_mlps` you can find the confusion matrices for each one of the fine-tuned MLP models, for each subject and each one of the three different input data arrangements. 

```diff
- (((TO-DO: check if this description is correct))).
```
______________

## Configuration of the environment

To run the code in this repository, you will need standard data science and machine learning Python libraries (pandas, numpy, scikit-learn, etc.), as well as TPOT and AutoKeras for the AutoML experiments.

TPOT can be installed via:

``` pip install TPOT ``` or  ``` conda install -c conda-forge tpot ```

AutoKeras can be installed via:

``` pip install autokeras```

______________