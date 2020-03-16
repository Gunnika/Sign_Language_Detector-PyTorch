# Sign_Language_Detector-PyTorch
Recognition of hand gestures in 3D space using a single low resolution camera for converting American Sign Language into any spoken language.

## Inspiration
There are only 250 certified sign language interpreters in India, translating for a deaf population of between 1.8 million and 7 million.

![percentage](https://user-images.githubusercontent.com/34855465/76789152-42404700-67e2-11ea-8e96-718ba4ae0a36.png)

We need to devise a solution that allows inclusion of the deaf and mute people in normal conversations. Our application allows any user to point the camera towards a mute person (with consent, ofcourse) and effectively understand what he/she's trying to say.

### American Sign Language (ASL)
American Sign Language (ASL) is a visual language. With signing, the brain processes linguistic information through the eyes. The shape, placement, and movement of the hands, as well as facial expressions and body movements, all play important parts in conveying information. 
![ASL](https://user-images.githubusercontent.com/34855465/76790591-28ecca00-67e5-11ea-990d-b6540acb9a1b.png)


## Dataset used
Sign Language MNIST (https://www.kaggle.com/datamunge/sign-language-mnist)
Each training and test case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because of gesture motions).

## Working
![steps](https://user-images.githubusercontent.com/34855465/76790048-1625c580-67e4-11ea-9fcb-77339e2c4658.png)

Autocompletion and Word Suggestion simplify and accelerate the process of information transmission. The user can select one out of the top 4 suggestions or keep making more gestures until the desired word is obtained. 


## Use Cases
1. Deaf people can have a common classroom by asking their questions/doubts without any hesitation
2. Inclusion of this community in normal schools.
3. Tourist Guides can communicate better using sign language

## Set Up Instructions

The `requirements.txt` file should list all Python libraries that your notebooks
depend on, and they will be installed using:

```
pip install -r requirements.txt
```

To run the web application:
```
python app.py
```
It will run your app on http://localhost:8888/

## Screenshots
### H
![H](https://user-images.githubusercontent.com/34855465/76798612-eda6c700-67f5-11ea-974e-514a82c8c5c5.png)

### A
![A](https://user-images.githubusercontent.com/34855465/76798664-044d1e00-67f6-11ea-9b41-0a4ca9f625e1.png)

