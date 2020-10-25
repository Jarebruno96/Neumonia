# Pneumonia

## Context

Pneumonia is an infection that inflames the air sacs in one or both lungs. Pneumonia can range in severity from mild to life threatening. It is most serious in infants and young children, people over 65, and people with health problems or weakened immune systems

Among the most prominent symptoms can be mentioned

- Fever
- Cough and expectoration
- chest pain
- Chills and sweating
- Headaches

Inhalation pneumonia is due to microorganisms that survive long enough while suspended in the air to be transported far from their source. These are smaller than 1 micron in size so that the aerosolized particles transport an inoculum, and thus avoid the host's defensive mechanisms.

<br></br>
## Objective

The aim is to create a model capable of predicting from a chest X-ray image whether the patient has pneumonia or not. For this, the public Kaggle dataset will be used:  https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

<br></br>
## Model creation

To create the model, use the file ** main.py **. The correct way to use this file is attached below

python main.py

Parameters are:

  - -h --help: Shows help message
  - -s --source: To Write and read data from local source or google cloud platform.
    - local (default): To Write and read data from local. 
    - gcp: To Write and read data from Google Cloud Platform.
  - -n --normalize: Normalize pixel values from 0 to 1.
    - yes (default): Normalize pixel values from 0 to 1.
    - no: Don't normalize pixel values from 0 to 1.
  - -H --height: Height image size to be resized at preprocessing.
  - -w --width: Widtht image size to be resized at preprocessing.
  - -e --epochs: Number of epochs at trainning. 10 by default.
  - -b --batch_size: Batch size at trainning. 32 by default.
  - -v --validationSplit: Percetange of validation at trainning from 0 to 1. 0.2 by default.
  - -c --channel: Number of channels to be read from images.
    - gray: Read images in gray scale.
    - rgb (default): Read images in rgb scale.
  - -k --gcpkey: Google key with Google Cloud Storage, at least with read and write permission. If it is not set, the virtual machine where the progam is running must have the permission
  - -u --bucket: Google Cloud Storage Bucket where data is stored
  - -f --folder: Local folder o GCP folder where the model and its configuration are going to be stored

After executing the previous file, a new directory will be created in which the model will be stored, as well as its information.

<br></br>
## Predictions

### Web service

In order to make predictions of images that the model has never seen before, a web service has been developed using Flask. This web service corresponds to the file ** webservice.py **. The correct way to activate the service is as follows:

python webservice.py
  - -h --help: Shows help message
  - -f --folder specifies model´s folder to be loaded
  - -s --source: To read data from local source or google cloud platform.
    - local (default): To read data from local. 
    - gcp: To read data from Google Cloud Platform.
  - -k --gcpkey: Google key with Google Cloud Storage, at least with read permission. If it is not set, the virtual machine where the progam is running must have the permission
  - -u --bucket: Google Cloud Storage Bucket where data is stored

The accessible methods of this web service are as follows:

- / status allows to know the status of the service. If it works properly, a ** OK ** will be returned

- / predict allows that, after passing an image of a chest x-ray of a patient, if he has pneumonia or not. This method will return the following information:
  - original: Information about the original image
  - proccessed: Information about the preprocessed image and provided to the model to be able to make its prediction
  - prediction: Value between 0 and 1 where:
    - 0: Indicates that the patient ** does not ** have pneumonia
    - 1: Indicates that the patient ** yes ** has pneumonia
  - predictionResult: Result of the prediction where:
    - Normal: Indicates that the patient ** does not ** have pneumonia
    - Pneumonia: Indicates that the patient ** yes ** has pneumonia


### Web Page


A web page ** web page / index.html ** has been developed to make use of the aforementioned web service in a simple way. On this website, an image can be uploaded by the user of the model in order to know if the selected image represents pneumonia or not.
