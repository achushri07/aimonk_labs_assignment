Aimonk Multilabel Classification Problem

Problem Overview
This is a multilabel image classification task where each image may contain multiple attributes. The dataset consists of an images/ folder and a labels.txt file. Labels include 1 (present), 0 (absent), and NA (unknown).

Model Architecture
ResNet50 pretrained on ImageNet was used as the backbone (include_top=False). A GlobalAveragePooling layer, Dropout (0.3), and a Dense layer with 4 output logits were added. The backbone was used as a feature extractor and fine-tuned on the dataset.

Handling Missing Labels
NA values were handled using a masking strategy. For unknown attributes, a dummy label (0) was assigned but excluded from loss computation using a binary mask. This allows learning from partially labeled images without introducing noise.

Handling Class Imbalance
The dataset is imbalanced. Positive class weights were computed as (negative_count / positive_count) and used in a weighted binary cross-entropy loss. Gradient clipping was applied for numerical stability.

Training Details
Image size: 224x224
Batch size: 16
Optimizer: Adam
Learning rate: 1e-5
Epochs: 8
Loss: Masked + Weighted Binary Cross Entropy

Training outputs:

* Saved model file: aimonk_multilabel_resnet50.h5
* Loss plot with iteration_number (x-axis) and training_loss (y-axis), titled "Aimonk_multilabel_problem"

Inference
The notebook includes a prediction function that takes an image path as input and outputs attribute probabilities along with the list of predicted attributes (threshold = 0.5).

Possible Improvements
Future improvements could include data augmentation, unfreezing deeper layers for fine-tuning, focal loss for imbalance, per-class threshold tuning, learning rate scheduling, early stopping, and validation metrics such as F1-score.

This implementation follows best practices for multilabel classification with partial annotations and class imbalance.
