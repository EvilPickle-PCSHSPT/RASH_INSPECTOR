import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

img_width, img_height = 128, 128
model1_dir = 'models_and_weights/model_m1.h5'
model1_weights_dir = 'models_and_weights/weights_m1.h5'
model1 = load_model(model1_dir)
model1.load_weights(model1_weights_dir)

model2_dir = 'models_and_weights/model_m2.h5'
model2_weights_dir = 'models_and_weights/weights_m2.h5'
model2 = load_model(model2_dir)
model2.load_weights(model2_weights_dir)

def predict(file):
  x = load_img(file, target_size=(img_width, img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model1.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print('Label: Atopic Dermatitis')
  elif answer == 1:
    print('Label: Other')
    array2 = model2.predict(x)
    result2 = array2[0]
    answer = np.argmax(result2)
    if answer == 0:
      print('Label: class1')
    elif answer == 1:
      print('Label: class2')
    elif answer == 2:
      print('Label: class3')
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

# for i, ret in enumerate(os.walk('Test_Data/Other')):
#   for i, filename in enumerate(ret[2]):
#     if filename.startswith("."):
#       continue
#     #print("Label: Other")
#     result = predict(ret[0] + '/' + filename)
#     if result == 1:
#         other_t += 1
#     else:
#         other_f += 1

"""
Check metrics
"""
print("True Atopic Dermatitis: ", atopic_t)
print("False Atopic Dermatitis: ", atopic_f)
print("True Other: ", other_t)
print("False Other: ", other_f)