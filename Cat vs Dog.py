import os
import cv2
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

train_dir = r"C:\Users\tahseen\OneDrive\Desktop\Prodigy InfoTech Internship\PRODIGY_ML_03-main\PRODIGY_ML_03-main\Sample train"
test_dir = r"C:\Users\tahseen\OneDrive\Desktop\Prodigy InfoTech Internship\PRODIGY_ML_03-main\PRODIGY_ML_03-main\Sample test"

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'): 
            img_path = os.path.join(folder, filename)
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.resize(image, (64, 64)) 
                images.append(image)
                label = 1 if 'dog' in filename else 0  
                labels.append(label)
    return np.array(images), np.array(labels)

X_train, y_train = load_images_from_folder(train_dir)

X_train_flat = X_train.reshape(X_train.shape[0], -1)

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_flat, y_train, test_size=0.2, random_state=42)

model = svm.SVC(kernel='linear')
model.fit(X_train_split, y_train_split)

X_test = []
test_filenames = []
for filename in os.listdir(test_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(test_dir, filename)
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.resize(image, (64, 64))
            X_test.append(image)
            test_filenames.append(filename.split('.')[0])  

X_test_flat = np.array(X_test).reshape(len(X_test), -1)

predictions = model.predict(X_test_flat)

submission = pd.DataFrame({
    'id': test_filenames,
    'label': predictions
})


submission.to_csv('Sample Submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")
