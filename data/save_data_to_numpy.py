#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import tensorflow as tf
import numpy as np
import os
import cv2

#==================================================Load Data from Image Folders and Save to Numpy Files==================================================
def save_data(base_dir, save_path_source, save_path_target, num_classes, num_images_per_class):
  # Set the base directory for the RealWorld and ClipArt folders
  base_dir = base_dir

  #GPU info
  physical_devices = tf.config.list_physical_devices('GPU')
  print("GPU: ", physical_devices)

  # Pre-Process data
  #
  # 1.   create 5-d array source_domain: `[class,image_no,height,width,channels]`
  # 2.   create 5-d array target_domain: `[class,image_no,height,width,channels]`
  #
  # -----
  # Note: Images are read using cv2 hence are in the BGR format

  # Path to Real World folder
  realworld_folder = os.path.join(base_dir, 'Real World')
  # Path to Clipart folder
  clipart_folder = os.path.join(base_dir, 'Clipart')
  # Numeric class labels
  class_labels = []
  # Initialize a 5-dimensional array with the appropriate shape
  num_classes = num_classes
  # Careful all classes don't have same number of images in OfficeHome Real World and Clipart
  # For standardization we will take 20 images per class
  num_images_per_class = num_images_per_class#len(os.listdir(os.path.join(realworld_folder, str(0))))
  num_channels = 3  # assuming the images are RGB
  image_height = 224
  image_width = 224
  source_domain = np.empty((num_classes, num_images_per_class, image_height, image_width, num_channels))
  target_domain = np.empty((num_classes, num_images_per_class, image_height, image_width, num_channels))

  # Load the images from the Real World folder
  real_world=sorted(os.listdir(realworld_folder))
  print(real_world)
  for i, folder in enumerate(real_world):
    # Stop loading data if required num_classes are selected
    if i < num_classes:
      class_folder = os.path.join(realworld_folder, folder)
      # Print number of images in the class
      print(class_folder, len(os.listdir(class_folder)))
      for j, image_path in enumerate(os.listdir(class_folder)):
        # If the current iteration is greater than num_images_per_class then break the inner loop and continue with the next class
        if j==num_images_per_class:# as indices start from 0
          break
        # Load the image and store it in the array
        image = cv2.imread(os.path.join(class_folder,image_path))
        # Convert images to 224*224
        image = cv2.resize(image, dsize=(image_height, image_width), interpolation=cv2.INTER_CUBIC)
        source_domain[i, j, :, :, :] = image
        # Append class labels
      class_labels.append(i)
    else:
      break

  # Load the images from the Clipart folder
  clipart=sorted(os.listdir(clipart_folder))
  print(clipart)
  for i, folder in enumerate(clipart):
    # Stop loading data if required num_classes are selected
    if i < num_classes:
      class_folder = os.path.join(clipart_folder, folder)
      # Print number of images in the class
      print(class_folder, len(os.listdir(class_folder)))
      for j, image_path in enumerate(os.listdir(class_folder)):
        # If the current iteration is greater than num_images_per_class then break the inner loop and continue with the next class
        if j==num_images_per_class:# as indices start from 0
          break
        # Load the image and store it in the array
        image = cv2.imread(os.path.join(class_folder,image_path))
        # Convert images to 224*224
        image = cv2.resize(image, dsize=(image_height, image_width), interpolation=cv2.INTER_CUBIC)
        target_domain[i, j, :, :, :] = image
    else:
      break

  np.save(f'{save_path_source}/source_{num_classes}.npy', source_domain)
  np.save(f'{save_path_target}/target_{num_classes}.npy', target_domain)