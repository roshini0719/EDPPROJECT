# Handwritten Digit Recognizer (MNIST)


A learning project that explores image preprocessing and handwritten-digit classification using the MNIST dataset. The MNIST portion includes a preprocessing exercise, a scikit-learn neural network, and a TensorFlow convolutional neural network (CNN).

## Project Structure

```text
week4/
  mnist.npz
  mnsit_preprossesing.py
week_5/
  mnist_neural_network.py
week_6/
  mnist_cnn.py
week_1/
  data_cleaning.py
  titanic.csv
week_2/
  linear_regression.py
  Student_DataSet.csv
week_3/
  spam_detection.py
  spam_sms.csv
```

The Week 1–3 scripts are separate data-analysis and machine-learning exercises. The main project is the MNIST work in Weeks 4–6.

## Requirements

- Python 3
- NumPy
- scikit-learn
- TensorFlow (for the Week 6 CNN)
- pandas and matplotlib (for the Week 1–3 exercises)

From the project root, create and activate a virtual environment in Windows PowerShell, then install the packages:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy scikit-learn tensorflow pandas matplotlib
```

If PowerShell prevents activation, you can run the scripts using the environment's Python directly: `.\.venv\Scripts\python.exe`.

## Run the MNIST Scripts

Run these commands from the project root:

```powershell
python week4/mnsit_preprossesing.py
python week_5/mnist_neural_network.py
python week_6/mnist_cnn.py
```

The MNIST scripts load `week4/mnist.npz` from the project. The preprocessing code scales image pixels from 0–255 to 0–1. The Week 5 model flattens each 28×28 image into 784 features and trains an MLP classifier. The Week 6 model keeps the image dimensions and trains a CNN with convolution, max-pooling, dropout, and a 10-class output layer.

The Week 5 script prints accuracy, a classification report, and a confusion matrix. The Week 6 script prints test loss, test accuracy, and predictions for the first ten test images.

## Results

Record the results from your own run here. Do not fill in an expected score in place of measured output.

| Model | Test accuracy | Notes |
| --- | --- | --- |
| Week 5 MLP | Accuracy       : 97.88 % | `week_5/mnist_neural_network.py` |
| Week 6 CNN | Test accuracy: 99.11 % | `week_6/mnist_cnn.py` |

The project plan also calls for visualizing misclassified examples in Weeks 6–7. Add that visualization and describe what you observe here when it is implemented; the current Week 6 script reports metrics and sample predictions, but does not yet plot misclassifications.

## Earlier Weekly Exercises

Each command should be run from its own week folder because these scripts load their CSV files using relative paths:

```powershell
cd week_1
python data_cleaning.py

cd ..\week_2
python linear_regression.py

cd ..\week_3
python spam_detection.py
```



## GitHub Checklist

- [ ] Add measured model results and complete the misclassification visualization section.
- [ ] Replace the contribution placeholders with the team's actual contributions.
- [ ] Confirm the dataset may be redistributed before committing `week4/mnist.npz` and the CSV datasets. If not, document where to obtain them instead of uploading them.
- [ ] Review the repository for private information and remove generated files such as `__pycache__`.
- [ ] Create a GitHub repository, add this project, and verify that the README's setup and run commands work from a fresh clone.