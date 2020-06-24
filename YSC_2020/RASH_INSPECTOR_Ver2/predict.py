import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

img_width, img_height = 128, 128
model_path = 'RASH_INSPECTOR_Ver2/models/arm_densenet121_model.h5'
model_weights_path = 'RASH_INSPECTOR_Ver2/models/arm_densenet121_weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

def predict(file):
  x = load_img(file, target_size=(img_width, img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print('Label: Atopic Dermatitis')
  elif answer == 1:
    print('Label: Other')
 
  return answer

atopic_t = 0
atopic_f = 0
other_t = 0
other_f = 0

for i, ret in enumerate(os.walk('Test_Data/AtopicDermatitis')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    #print("Label: Atopic")
    result = predict(ret[0] + '/' + filename)
    if result == 0:
        atopic_t += 1
    else:
        atopic_f += 1

for i, ret in enumerate(os.walk('Test_Data/Other')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    #print("Label: Other")
    result = predict(ret[0] + '/' + filename)
    if result == 1:
        other_t += 1
    else:
        other_f += 1

"""
Check metrics
"""
print("True Atopic Dermatitis: ", atopic_t)
print("False Atopic Dermatitis: ", atopic_f)
print("True Other: ", other_t)
print("False Other: ", other_f)