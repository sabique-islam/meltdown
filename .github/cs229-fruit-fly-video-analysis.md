# Real-time Detailed Video Analysis of Fruit Flies

**CS229 Fall 2018 Final Project**
**Steven Herbst** — sherbst@stanford.edu

Source: [cs229.stanford.edu/proj2018/report/61.pdf](https://cs229.stanford.edu/proj2018/report/61.pdf)
Code and dataset: [github.com/sgherbst/cs229-project](https://github.com/sgherbst/cs229-project.git)

## Abstract

The report describes a real-time image-processing pipeline for fruit-fly video. It estimates each fly’s position, orientation, and sex, and, for the male, both wing angles. The models are a decision tree, linear regression, logistic regression, and principal component analysis (PCA). Features for the later stages come from a Histogram of Oriented Gradients (HOG) descriptor.

On 1530×1530 grayscale frames, without a GPU, the pipeline runs at **84 frames per second** and is accurate on the reported metrics.

## 1. Introduction

Fruit flies are common in neurobiology because they can be genetically modified with established tools and because whole-brain imaging in vivo is feasible. Experiments often need each fly’s position and orientation over time, plus leg and wing positions.

Existing automated trackers are usually too slow for real time. That blocks closed-loop behavior experiments, and it makes video processing the bottleneck in long recordings such as circadian studies.

This project targets one experiment from Prof. Tom Clandinin’s lab at Stanford: courtship between one male and one female. The input is a grayscale video of the two flies. The outputs are position, orientation, and sex for each fly, plus the male’s wing angles, which are used to measure the rapid wing motion that produces courtship song.

The report covers related tools (Section 2), the dataset (Section 3), a four-stage machine-learning pipeline (Section 4), results (Section 6), and a conclusion (Section 7).

## 2. Related work

**FlyTracker** (Eyjolfsdottir et al., 2014) is treated as a gold standard. It is a MATLAB workflow from Caltech’s Computational Vision Lab that records body position, leg position, body angle, and wing angle for many interacting flies, using thresholding and morphological transforms. Informal tests found it about **10× slower than real time**.

**Klibaite et al. (2017)** used unsupervised learning on pairs of flies. Inputs were two egocentrically aligned videos: each fly was centered, cropped, and rotated to a nominal heading. Segmentation used a Gaussian mixture model on the pixel-intensity histogram to set thresholds, then assigned appendages by pixel distance from the body. When the bodies themselves could not be split, a watershed algorithm was used, which sometimes masked out wings or legs. Speed was not reported.

**DeepLabCut** (Mathis et al., 2018) and **LEAP** (Pereira et al., 2018) both use deep networks.

- DeepLabCut uses transfer learning from an ImageNet-pretrained feature network so fewer labeled frames are needed. Reported speed: **30 Hz** on 682×540 images with a GPU.
- LEAP uses a simpler custom network aimed at higher throughput: **185 Hz** on smaller images, with about one hour of training, both with a GPU.

This project keeps supervised learning on labeled data, unlike FlyTracker’s hand-crafted rules, but uses much simpler models than DeepLabCut and LEAP so that training stays fast and large frames (1530×1530) can be processed in real time on a CPU.

## 3. Dataset

The source is a **15-minute grayscale video** of one male and one female, provided by Dr. Ryan York as an example of footage from the lab’s experimental rig. The video was cropped to a **1530×1530** frame that exactly contains the circular well.

**326 frames** were hand-annotated in LabelMe with heads, abdomens, and a point inside each body. Male frames also include points for wing angles. About one hundred frames were chosen at random. Because flies move in bursts, that sample was not very varied, so the remaining ~200 frames were picked by hand for hard cases: crawling on the wall, flies in contact, and extensive wing motion.

The report notes that LEAP first reduces frames with PCA and then clusters them to find distinct poses, but that method assumes frames are already egocentrically aligned.

Further preprocessing and data augmentation are described with each pipeline stage.

## 4. Methods

The pipeline has four machine-learning stages:

1. Extract contours and keep those that contain an individual fly.
2. When there are two single-fly contours, decide which is male and which is female.
3. Estimate each fly’s orientation over a full 0–360° range.
4. For the male, estimate the left and right wing angles.

Implementation is Python with scikit-learn, opencv-python, NumPy, Matplotlib, imutils, joblib, and tqdm.

| Stage | What it does |
| --- | --- |
| I/O | Read a frame from the video. |
| FlyCount | Threshold the image, extract contours, and classify each contour as 0, 1, or 2 flies. |
| Male vs. female | If there are two one-fly contours, identify which is male and which is female. |
| Orientation | For both flies, estimate body orientation over 0–360°. |
| WingAngle | For the male, estimate each wing’s angle over 0–90° relative to the body’s major axis. |

### 4.1 FlyCount

Thresholding removes the background and the appendages, leaving the body. Contours are extracted from that mask. A contour can contain zero flies (noise or a background object), one fly, or two flies (the animals are touching).

Each contour’s **area** is the only feature. A **decision tree** classifies it as zero, one, or two flies. Splits are chosen to minimize **Gini impurity**:

$$
J = \sum_{k=1}^{K} p_k (1 - p_k)
$$

where $p_k$ is the proportion of examples of class $k$ in the split and $K = 3$.

The learned tree is easy to read. From the trained tree in the report (759 training contours: 309 neither, 416 one fly, 34 both):

- area ≤ **1413.5** → neither
- area ≤ **10442.5** → one fly
- otherwise → both

### 4.2 Male vs. female

If there are two one-fly contours, the next stage decides which is male and which is female. Classification is joint: the model sees both contours and predicts whether the list is ordered male–female or female–male. Females are usually larger, but the difference is harder when a fly is climbing a wall.

Features (four total): **area** and **aspect ratio** of each contour. Aspect ratio comes from image-moment analysis. Features are rescaled to zero mean and unit variance (`StandardScaler` in the figure), then passed to **logistic regression**:

$$
h_\theta(x) = \frac{1}{1 + \exp(-\theta^\top x)}
$$

with the single-example update

$$
\theta := \theta + \alpha \,(y - h_\theta(x))\, x
$$

Rescaling matters because area and aspect ratio have very different magnitudes, so the update would otherwise behave badly. Data augmentation swaps the order of the two contours and flips the label.

### 4.3 Orientation

The paper numbers this stage as Section 5. Orientation over 0–360° is reduced to a binary classification.

Preprocessing masks the image to one fly’s body. Image moments then give an angle with a forward/backward ambiguity:

$$
\theta \approx \frac{1}{2} \tan^{-1}\!\left(\frac{2\mu_{11}}{\mu_{20} - \mu_{02}}\right)
$$

$$
\mu_{pq} = \sum_x \sum_y (x - \bar{x})^p (y - \bar{y})^q f(x, y)
$$

where $(\bar{x}, \bar{y})$ is the center of mass and $f(x, y)$ is pixel intensity. The moment angle cannot tell front from back, so a second model decides whether to add 180°.

That model:

1. Rotates the fly image by the moment angle $\theta$.
2. Computes a **HOG** descriptor.
3. Projects it onto **15 principal components**.
4. Uses **logistic regression** to decide whether the 180° correction is needed.

HOG builds a histogram of gradient directions on 8×8 patches. Each patch has 64 pixels. Gradient magnitude and direction are computed at each pixel, then binned into **9** angular ranges, weighted by magnitude. Patch histograms are flattened into one vector.

The 15 components are the eigenvectors for the 15 largest eigenvalues of $\sum_i x^{(i)}{x^{(i)}}^\top$, where $x^{(i)}$ is the HOG descriptor of training example $i$. Those 15 components explain about **60%** of the feature variance. Figure 8 shows that even the first two components already separate forward-facing from backward-facing flies (shown for females). Data augmentation rotates images 180° and inverts the labels.

The HOG input at this stage is a **128×64** image, producing a **3780×1** descriptor before PCA.

### 4.4 Wing angle

The paper numbers this stage as Section 5.1. Left and right wing angles of the male are estimated separately.

An ROI is built around the male and its wings, then rotated upright using the orientation stage. Median blurring, adaptive thresholding, and erosion keep the wings and remove the legs. The upright fly image is split into two halves, one per wing.

For each half:

1. Compute a HOG descriptor. The crop is a **96×80** image, giving a **3564×1** descriptor.
2. Project onto **40 principal components** (chosen to reduce test error).
3. **Linear regression** predicts the wing angle.

Figure 9 shows that wing angle is already roughly linear in the first principal component alone.

Linear regression minimizes mean squared error. The closed-form solution is

$$
\theta = (X^\top X)^{-1} X^\top y
$$

and the prediction for features $x$ is $\theta^\top x$.

## 6. Results

One third of the data was held out and never used in training. The first three stages are classifiers, so error is misclassification rate. FlyCount makes no errors. Sex and orientation errors are rare and mostly happen when a fly climbs a wall: the camera’s depth of field is small, so climbing flies become blurry enough that even a person has trouble.

WingAngle is a regression. Mean error is small (**0.592°**). Test error is reported as a standard deviation of **2.92°**, which the report reads as about a **95% interval of ±6°**. Wings only sweep about 90°, so that error is not negligible, but overlays on video look qualitatively good ([example clip](https://bit.ly/2SGMXTQ)).

### Table 1. Model accuracy

| Model | Train error | Test error | N train | N test |
| --- | --- | --- | --- | --- |
| FlyCount | 0.0% | 0.0% | 759 | 253 |
| Male/female order | 0.2% | 0.7% | 423 | 141 |
| Male orientation | 0.0% | 0.0% | 358 | 120 |
| Female orientation | 0.8% | 0.0% | 259 | 87 |
| WingAngle | σ = 2.06° | σ = 2.92° | 354 | 118 |

Orientation is scored as a classifier because logistic regression only decides whether the moment-based heading needs a 180° flip.

### Table 2. Timing

Measured on a 2.8 GHz Intel Core i7 with 16 GB RAM, no GPU. Mean throughput is **84.0 FPS**, which is **2.4×** the source video rate.

| Step | Runtime | Share of total |
| --- | --- | --- |
| I/O | 3.9 ms | 32.5% |
| FlyCount | 2.2 ms | 18.3% |
| Male vs. female | 0.7 ms | 5.8% |
| Orientation | 1.9 ms | 15.8% |
| WingAngle | 3.3 ms | 27.5% |

Reading the frame is the slowest step, followed by WingAngle, which has the heaviest preprocessing.

Figure 10 plots male wing angle over about 16 seconds. The two wings tend to move in opposite directions; their cross-correlation is **−0.64**. Large oscillations are the courtship songs from rapid wing motion.

## 7. Conclusion

The pipeline takes a grayscale video of a pair of flies and returns position, orientation, and sex for each fly, plus male wing angles. The four stages use a decision tree, logistic regression, linear regression, and PCA. On 1530×1530 frames it reaches 84 FPS on a CPU with the accuracy in Table 1.

HOG plus PCA worked well in the orientation and wing-angle stages even with one or two components. The report attributes that to preprocessing that increases HOG variance from the quantity of interest (heading or wing angle) and decreases variance from legs, the other wing, the other fly, and background texture.

Future directions named in the report:

- Leg positions. Early tests suggest leg tips are found reliably by a keypoint detector such as SIFT, so this could be a keypoint-classification problem.
- GPU-accelerated OpenCV on a CUDA machine, to see how far throughput can go.
- With Dr. York, unsupervised methods such as t-SNE on the feature vectors this pipeline already produces.

## References

1. G. Bradski. 2000. The OpenCV Library. *Dr. Dobb’s Journal of Software Tools*.
2. Navneet Dalal and Bill Triggs. 2005. Histograms of Oriented Gradients for Human Detection. *CVPR ’05*. [doi:10.1109/CVPR.2005.177](https://doi.org/10.1109/CVPR.2005.177)
3. J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. 2009. ImageNet: A Large-Scale Hierarchical Image Database. *CVPR 2009*.
4. Eyrun Eyjolfsdottir, Steve Branson, Xavier P. Burgos-Artizzu, Eric D. Hoopfer, Jonathan Schor, David J. Anderson, and Pietro Perona. 2014. Detecting Social Actions of Fruit Flies. *ECCV 2014*, 772–787. [doi:10.1007/978-3-319-10605-2_50](https://doi.org/10.1007/978-3-319-10605-2_50)
5. John D. Hunter. 2007. Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering* 9(3), 90–95. [doi:10.1109/mcse.2007.55](https://doi.org/10.1109/mcse.2007.55)
6. Ugne Klibaite, Gordon J. Berman, Jessica Cande, David L. Stern, and Joshua W. Shaevitz. 2017. An unsupervised method for quantifying the behavior of paired animals. *Physical Biology* 14(1), 015006. [doi:10.1088/1478-3975/aa5c50](https://doi.org/10.1088/1478-3975/aa5c50)
7. David G. Lowe. 2004. Distinctive Image Features from Scale-Invariant Keypoints. *Int. J. Comput. Vision* 60(2), 91–110. [doi:10.1023/B:VISI.0000029664.99615.94](https://doi.org/10.1023/B:VISI.0000029664.99615.94)
8. Laurens van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-SNE. *Journal of Machine Learning Research* 9, 2579–2605.
9. Alexander Mathis et al. 2018. DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. *Nature Neuroscience* 21(9), 1281–1289. [doi:10.1038/s41593-018-0209-y](https://doi.org/10.1038/s41593-018-0209-y)
10. Travis Oliphant. NumPy: A guide to NumPy. [numpy.org](http://www.numpy.org/)
11. F. Pedregosa et al. 2011. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research* 12, 2825–2830.
12. Talmo D. Pereira et al. 2018. Fast animal pose estimation using deep neural networks. [doi:10.1101/331181](https://doi.org/10.1101/331181)
13. Raphael Candelier. 2018. Tracking object orientation with image moments. [raphael.candelier.fr](http://raphael.candelier.fr/?blog=Image%20Moments)
14. Bryan C. Russell, Antonio Torralba, Kevin P. Murphy, and William T. Freeman. 2008. LabelMe: A Database and Web-Based Tool for Image Annotation. *Int. J. Comput. Vision* 77(1–3), 157–173. [doi:10.1007/s11263-007-0090-8](https://doi.org/10.1007/s11263-007-0090-8)
15. Wikipedia contributors. 2018. Decision tree learning. [Wikipedia](https://en.wikipedia.org/w/index.php?title=Decision_tree_learning&oldid=871969828)
16. Wikipedia contributors. 2018. Image moment. [Wikipedia](https://en.wikipedia.org/w/index.php?title=Image_moment&oldid=859704100)
