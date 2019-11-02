import cv2
import random

def my_gen(limit):
  data = 1
  while data < limit:
    yield data
    data += 2
    
gen_obj = my_gen(5)
print(next(gen_obj))
print(next(gen_obj))

ROOT_PATH = './'
image_size = 224

def random_brightness(image):
    # Convert 2 HSV colorspace from BGR colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Generate new random brightness
    rand = random.uniform(0.3, 1.0)
    hsv[:, :, 2] = rand*hsv[:, :, 2]
    # Convert back to BGR colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img

def zoom(image):
    zoom_pix = random.randint(0, 10)
    zoom_factor = 1 + (2*zoom_pix)/image_size
    image = cv2.resize(image, None, fx=zoom_factor,
                       fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
    top_crop = (image.shape[0] - image_size)//2
    left_crop = (image.shape[1] - image_size)//2
    image = image[top_crop: top_crop+image_size,
                  left_crop: left_crop+image_size]
    return image

def get_image(index, data, should_augment):
    # Read image and appropiate traffic light color
    image = cv2.imread(os.path.join(
        ROOT_PATH, data['path'].values[index].strip()))
    color = data['class'].values[index]
    should_chng_brightness = random.randint(0, 1)
    should_flip = random.randint(0, 1)
    should_zoom = random.randint(0, 1)
    if should_augment:
        if should_chng_brightness == 1:
            image = random_brightness(image)
        if should_flip == 1:
            image = cv2.flip(image, 1)
        if should_zoom == 1:
            image = zoom(image)

    return [image, color]