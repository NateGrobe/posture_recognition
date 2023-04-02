# Posture Recognition

### Run the App:
Install dependencies from requirements.txt

```bash
pip install -r requirements.txt
```

To run, execute src/main.py

```bash
python src/main.py
```

### Explanation of Training Utilities
```bash
take_photos.py # saves frames from webcam, overlays media pipe points and saves them to the folder specified by "title" variable
```

```bash
create_csv.py # reads in all jpg files from src/training folder and creates csv file with media pipe points for each image
```

```bash
train_models.py # reads coords.csv file containing training data and generates machine learning models
```

```bash
evaluate_models.py # uses contents of evaluation data to create separate set of validation data and determines each models accuracy
```

```bash
rename.py # mass renaming utility, used when first setting up training data
```

#### To train your own models:

Change title variable in take_photos to the name of the pose you want to create. Run the program and stand in the desired pose.

To execute:
```bash
cd src/training
python take_photos.py
```

Next, execute train_models program. This will automatically train new models based on the training data generated.

To execute:
```bash
cd src/training
python train_models.py
```
