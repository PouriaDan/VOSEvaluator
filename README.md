# VOSEvaluator
An Ipython-widgets based monitoring tool for Semi-Supervised Video Object Segmentation models


https://user-images.githubusercontent.com/15379497/138335288-74af451b-4ee0-4765-ab78-18e8904f4ab8.mov


The evaluation code is based on [DAVIS 2017 Semi-supervised and Unsupervised evaluation package](https://github.com/davisvideochallenge/davis2017-evaluation)

# Usage:

In your conda environemt, run `monitoring_vos_model.ipynb`

Specify paths to your data in:
```bash
frame_path = 'Path/To/JPEGImages' #Replace with frames dir
gt_path = 'Path/To/Annotations' #Replace with annotations dir
results_path = 'Path/To/Results' #Replace with results dir
saved_evals = 'Path/To/SavedEvaluation' #Replace with saved_evaluation dir
```

And run the Cell
