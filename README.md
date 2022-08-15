# LINKS: A dataset of a hundred million planar linkage mechanisms for data-driven kinematic design

<img width="100%" src="https://i.ibb.co/BCPTFt0/overview-1.jpg" alt="overview">

Here you will find the code we used to generate the dataset and simulate it. A reduced version of the dataset is also publicly available. To gain access to the full dataset send your request to us in the link provided below.

## Required packages

- tensorflow > 2.4.0
- sklearn
- numpy
- matplotlib
- tqdm
- svgpath2mpl


## Dataset
A reduced dataset of 1,000,000 Mechanisms are provided publicly without any limits. If you are interested in recieving the complete 100,000,000 mechanisms please send your request in the link below:

<a href="https://forms.gle/Xx8ZJiZGPjDPVBFu7">Full Dataset Request Form</a>

### Dataset Description
The dataset is provided in 4 files. The mechanisms in the "dataset" file, the numerical simulation data for each mechanism in the "simulation_dataset" file and the dataset of the normalized curves in the "normalized_dataset" file, and the curated normalized curves in the "curated_dataset" file.

For more details on the structure of the data and use of our utilty functions see the jupyter notebook in the Dataset folder.

### Code Details
The code provided here includes all the parts needed to generate your own dataset. To do this simply run:

```bash
   python train_models.py --N 1000 --save_name dataset
```

This will generate 5000 (5 variations on 1000 topoloies) and save the dataset in files called "dataset", "simulation_dataset", "noralized_dataset", "curated_dataset" with structures identical to the dataset provided.

Note that the GPU solvers are also included in the sim.py but not used directly in the code.

### Webdemo And Project Page Coming August 22nd
Web demo and project webpage will go live august 22nd, 2022.

Project Webpage:
<a href="https://decode.mit.edu/projects/LINKS">Project Page</a>

### More research on inverse kinematics coming soon! Keep an eye out:

Sneak Peak:
[![Watch the video]](https://filebin.net/puxmtnqjsbvyuf6l/IBM_Final.mp4)

