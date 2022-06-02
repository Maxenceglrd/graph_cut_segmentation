# Graph-cut algorithm for image segmentation -  Alpha-expansion extension

### MVA / CentraleSupélec 2022: Graphical Models - Discrete Inference and Learning

### Author: Maxence Gélard

This repository contains the code for my project for the course
Graphical Models - Discrete Inference and Learning.

I have been exploring graph cut algorithms for image 
segmentation, evaluating them on the COCO dataset from Microsoft.

It is organized as follows:

- `main.py`: main file from which we can run the different algorithm.
It comes with an argument parser than uses the following argument:
  - `algorithm`: `binary`or `alpha-expansion
  - `image-index`: index of the image (in the case of looking at
    only one image), from the COCO dataset (25 images in total)
  - `unary-cost`: `normal` or `l2_dist`
  - `sigma`: float, bandwidth for the binary potentials
  - `lmbda`: float, weight of the unary potentials
  - `weight-binary`: float, weight of the binary potentials
  - `mode`: `one` (only test on one image) or `all` (benchmark on all the datasets)
    
- `graph_cut_runner.py`: contains the main function called by `main.py` to run
the experiment
- `graph_cut_base.py`: code shared by the different algorithm on graph cut
  (namely the interactive annotations)
- `graph_cut_binary.py`: code for the binary segmentation using graph cut
- `graph_cut_alpha.py`: code for the multi-label segmentation using graph cut (alpha-expansion)
- `ford_fulkerson.py`: code containing a hand-made implementaiton of the Ford-Fulkerson
algorithm used for graph cut (N.B: as mentionned in some part of the code in `graph_cut_binary.py` for 
  example, we end up using the networkx library for minimum cut for time performance issues.)
- `coco_dataset.py`: contains a helping class to help loading the coco dataset
- `evaluation.py`: evaluation metric (IoU)
- `mask_rcnn.py`: implementation of a mask RCNN to compare our method
with this Deep Learning architecture
- `utils.py`: file containing the list a predictable classes for the mask RCNN.  

Moreover, we will find:
- `coco-dataset` folder: with the downloaded 25 samples from the
validation split of the Microsoft's COCO dataset.
- `precomputed_annotation` folder: with pickles containing the interactive
annotation already done for the COCO dataset (I did them to facilitate experiments)


Instructions on the image annotation: when running the algorithm, a window
will open with the image. To start annotating for a new label, click one time in the image
(release the left button), and move your mouse on the pixels you want
to annotate. Press another time the left button to validate this label. Iterate
until you reached the desired number of labels. Press the escape button to finalize the annotation process
Note that, in order to be consistent with the evaluation pipeline I set up, you need to make sure that:
- For the binary segmentation problem, you start by annotating the "object" (so the cat or the dog) and then the background.
- For the multi-label segmentation problem (so there are both cats and dogs), you start by the dog, then the background and finally the cat.

Acknowledgement to some tutorials / repositories that helped me for this project:

- https://sandipanweb.wordpress.com/2018/02/11/interactive-image-segmentation-with-graph-cut/
- https://profs.etsmtl.ca/hlombaert/energy/
- https://github.com/NathanZabriskie/GraphCut

(In addition to the course main page: https://lear.inrialpes.fr/people/alahari/disinflearn/index.html)
