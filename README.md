# CV-Sign: Sign Language Recognition using TensorFlow Object Detection

An end-to-end computer vision project focused on training a custom object detection model utilizing the TensorFlow Object Detection API. This repository handles everything from image collection, data annotation, and preprocessing to model training and evaluation.

## 📁 Repository Structure

```bash
CV-SIGN/
│
├── Tensorﬂow/
│   ├── labelImg-master/          # Graphical image annotation tool for labeling images
│   ├── scripts/                  # Utility scripts for data preprocessing
│   │   └── generate_tfrecord.py  # Converts XML annotations into TFRecord format
│   │
│   └── workspace/                # Active training and configuration workspace
│       ├── annotations/          # Contains label maps and compiled dataset records
│       │   ├── label_map.pbtxt   # Defines IDs and string names for custom classes
│       │   ├── test.record       # TFRecord binary file for evaluation data
│       │   └── train.record      # TFRecord binary file for training data
│       │
│       ├── images/               # Raw and split dataset components
│       │   ├── collectedimages/  # Original images gathered from source/webcam
│       │   ├── test/             # Testing subset with corresponding XML labels
│       │   └── train/            # Training subset with corresponding XML labels
│       │
│       ├── models/               # Training checkpoints and custom configurations
│       │   └── my_ssd_mobnet/    # Log files, pipelines, and checkpoints for fine-tuning
│       │
│       └── pre-trained-models/   # Base architectures downloaded from TF Model Zoo
│           └── ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/
│
├── Tutorial.ipynb                # Main notebook walking through the entire pipeline step-by-step
└── README.md                     # Project documentation

```

---

## 🚀 Workflow & Pipeline

### 1. Image Collection & Labeling

* Raw images are collected and stored inside `Tensorflow/workspace/images/collectedimages`.
* Images are labeled manually using the integrated `labelImg` tool, producing Pascal VOC XML annotation files.
* Data is split into `train` and `test` subdirectories for model calibration.

### 2. Generating Training Data

To feed data into the TensorFlow Object Detection API, the XML annotations must be converted into the binary `.record` format.
Run the script provided in the repository:

```bash
python Tensorflow/scripts/generate_tfrecord.py -x Tensorflow/workspace/images/train -l Tensorflow/workspace/annotations/label_map.pbtxt -o Tensorflow/workspace/annotations/train.record
python Tensorflow/scripts/generate_tfrecord.py -x Tensorflow/workspace/images/test -l Tensorflow/workspace/annotations/label_map.pbtxt -o Tensorflow/workspace/annotations/test.record

```

### 3. Model Fine-Tuning

* **Base Model:** SSD MobileNet V2 FPNLite (320x320), optimized for real-time edge deployment and lightweight performance.
* Custom configurations (learning rate, batch size, and class limits) are adjusted inside the `pipeline.config` within `Tensorflow/workspace/models/my_ssd_mobnet/`.

---

## 🛠️ Prerequisites & Installation

To set up the workspace local environment, clone this repository and install the mandatory dependencies:

1. Clone the official TensorFlow Object Detection API inside the `Tensorflow` directory if it's missing:
```bash
cd Tensorflow && git clone [https://github.com/tensorflow/models](https://github.com/tensorflow/models)

```


2. Install the required libraries via the provided `Tutorial.ipynb` or manually:
```bash
pip install tensorflow matplotlib opencv-python object-detection

```



## 📈 Next Steps

* Open `Tutorial.ipynb` in your Jupyter environment.
* Follow the step-by-step execution cells to set up paths, download pre-trained checkpoints, trigger local training loops, and deploy the webcam inferencing module.

