# Knot-Contactmap-ML
This is the Convolutional Neural Network model developed to identify the types and locations of knots from contact maps.

# How to Use This Program
1. You need to install the required packages.

```bash
$ pip install -r requirements.txt
```

2. Prepare the Data
 Place the data files in the ```augmented_data``` folder. The data should be in the following format:

   ```train_image.npy```: Training data
   
   ```test_image.npy```: Test data
   
   ```valid_image.npy```: Validation data

3. Train the Model
   To train the model, run ```main.py```. Enter the following command:
```bash
$ pip python main.py
```

4. Test the Model
   
   You can evaluate the trained model using the test data. Run ```test.py``` to check the model's performance:
```bash
$ pip python test.py
```

5. Check the Results
   Training and test results will be saved in the ```history_metrics.dat``` and ```test.dat``` files. You can check the performance metrics in these files.
