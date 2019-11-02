# haar-dir-creation
Takes a root directory of images and saves the cumulative result of a haar cascade into a new directory of images.
For example, takes a directory of images of people and a frontal face haar cascade and creates a new directory filled with images of just faces. Can be useful for specific dataset creation using a broader base dataset. Allows for custom random cropping. 

Example Usage:

<code>
  python3 haar_dataset_creation --root_dir ./people_dataset --save_dir ./face_dataset --max_images 1000
 </code>
