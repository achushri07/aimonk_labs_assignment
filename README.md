# AIMonk Multilabel Classification Problem

## Problem Overview

This project solves a multilabel image classification task where each image may contain multiple attributes. The dataset consists of:

* `images/` → folder containing all images
* `labels.txt` → annotation file with 4 attributes per image

Label meanings:

* `1` → Attribute present
* `0` → Attribute absent
* `NA` → Attribute information not available

The objective is to train a deep learning model using pretrained weights, properly handle missing labels, address class imbalance, generate a training loss plot, and provide inference capability.

---

## Model Architecture

* Backbone: **ResNet50 (ImageNet pretrained)**
* `include_top=False`
* GlobalAveragePooling2D
* Dropout (0.3)
* Dense layer with 4 output logits

The model is fine-tuned on top of pretrained ImageNet weights (not trained from scratch).

---

## Handling Missing Labels (NA)

The dataset contains partially labeled samples. Instead of discarding images with missing attributes, a **masking strategy** was used:

* If attribute = `1` → label = 1, mask = 1
* If attribute = `0` → label = 0, mask = 1
* If attribute = `NA` → label = 0 (dummy), mask = 0

During training:

```python
loss = loss * mask
```

Only known attributes contribute to the loss. This allows the model to learn from partially labeled images without introducing label noise.

---

## Handling Class Imbalance

The dataset is skewed across attributes. To address this:

* Computed positive class weights:

  ```
  pos_weight = negative_count / positive_count
  ```

* Used **weighted binary cross entropy loss**

* Applied gradient clipping for numerical stability

This improves learning for rare attributes.

---

## Training Configuration

* Image size: 224 × 224
* Batch size: 16
* Optimizer: Adam
* Learning rate: 1e-5
* Epochs: 8
* Loss: Masked + Weighted Binary Cross Entropy

### Outputs

* Saved model file: `aimonk_multilabel_resnet50.h5`
* Training loss plot:

  * X-axis → `iteration_number`
  * Y-axis → `training_loss`
  * Title → `Aimonk_multilabel_problem`

Note: Since loss is plotted per mini-batch iteration, fluctuations are expected.

---

## Inference

The notebook includes a prediction function:

```python
probs, present = predict_image(model, image_path, threshold=0.5)
```

It returns:

* Probabilities for each attribute
* List of predicted attributes above threshold

Threshold tuning per attribute can further improve performance.

---

## Possible Improvements

Due to time constraints, the following enhancements were not implemented but could further improve results:

* Data augmentation (flip, rotation, brightness adjustment)
* Two-stage fine-tuning (unfreezing deeper ResNet layers)
* Focal Loss for stronger imbalance handling
* Per-class threshold optimization
* Learning rate scheduling
* Early stopping
* Validation metrics such as F1-score, Precision, Recall
* Cross-validation

---

## Conclusion

This implementation:

* Uses a pretrained deep learning architecture
* Handles partial annotations using masking
* Addresses class imbalance using weighted loss
* Produces required training plot and inference pipeline

The solution follows best practices for multilabel classification with incomplete annotations and imbalanced data.
