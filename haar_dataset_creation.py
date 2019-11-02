import cv2
import glob
import random
import sys
import argparse
import os


def detect_faces(cascade, img, scale_factor=1.1, min_neighbors=4, resize_min=.75, resize_max=1.5, offset_max=.5, min_image_size=128, save_name="TEST.jpg"):

    image_copy = img.copy()
    files_created = 0
    rect = cascade.detectMultiScale(image_copy, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    height, width, _ = image_copy.shape

    for (x, y, w, h) in rect:

        # creates random cropping

        resize = random.uniform(resize_min, resize_max)

        padding = int(w * resize)

        x += int((w * random.uniform(-offset_max, offset_max)))
        y += int((y * random.uniform(-offset_max, offset_max)))

        x -= int(padding / 2)
        y -= int(padding / 2)
        w += int(padding)
        h += int(padding)

        #checks random cropping boundries

        if x < 0:
            w += x
            x = 0
        if y < 0:
            h += y
            y = 0

        if x + w > width:
            w = width - x
        if y + h > height:
            h = height - y


        # creates new image

        if w > min_image_size and h > min_image_size:
            new_image = image_copy[y:y+h, x:x+w, :]
            cv2.imwrite(save_name, new_image)
            files_created += 1

        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return files_created


def main(args):

    parser = argparse.ArgumentParser(description='Save cascading images from directory')
    parser.add_argument('-rd', '--root_dir', default='', type=str, help='where the images are stored')
    parser.add_argument('-sd', '--save_dir', default='', type=str, help='where the cascaded images are saved')
    parser.add_argument('-sf', '--scale_factor', default=1.1, type=float, help='scale factor')
    parser.add_argument('-rmi', '--resize_min', default=.75, type=float, help='min % increase in cascade size')
    parser.add_argument('-rma', '--resize_max', default=1.5, type=float, help='max % increase in cascade size')
    parser.add_argument('-oma', '--offset_max', default=.5, type=float, help='max % x and y offset')
    parser.add_argument('f', "--sampling_frequency", default=100, type=int, help='verbose frequency')
    parser.add_argument('c', "--cascade_dir", default='haarcascade_frontalface_default.xml', type=str, help="if custom = True, cascade_dir is absolute path to xml cascade file, else looks for cascade_dir in cv2.data.haarcascades library")
    parser.add_argument('cu', '--custom', default=False, type=bool, help="whether or not custom xml is specified in cascade_dir")
    parser.add_argument('mn', '--min_neighbors', default=4, type=int, help="number of minimum neighbors - the lower the value the more detections (but also more false positives)")
    parser.add_argument('mc', '--max_images', default=100000, type=int, help="max number of images created")
    parser.add_argument("mis", '--min_image_size', default=128, type=128, help="default resolution to be saved -- makes sure only good images are saved")
    args = parser.parse_args()

    save_dir = args.save_dir
    sampling_frequency = args.sampling_frequency

    print("Loading cascade and globbing root dir")

    # loads custom haar classifier if given

    if not args.custom:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + args.cascade_dir)
    else:
        cascade = cv2.CascadeClassifier(args.cascade_dir)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # globs root directory

    images = glob.glob(args.root_dir + "/*")

    print("Total images to be processed: {}".format(str(len(images))))

    image_count = 0
    files_created_count = 0

    # creates haar images until all images processed or max images created is reached

    for image in images:
        image_count += 1
        img = cv2.imread(image)

        save_name = save_dir + "/_" + str(image_count) + ".jpg"
        height, width, _ = img.shape

        # scales to 500px to create consistent cascading

        if not (height < 500 and width < 500):
            div = height / 500
            img = cv2.resize(img, (int(width/div), int(height/div)))
        files_created_count += detect_faces(cascade, img, scale_factor=args.scale_factor, min_neighbors=args.min_neighbors, resize_min=args.resize_min, resize_max=args.resize_max, offset_max=args.offset_max, min_image_size=args.min_image_size, save_name=save_name)
        if files_created_count > args.max_images:
            print("Reached max. Files created: {}".format(str(files_created_count)))
            break

        if image_count % sampling_frequency == 0 and image_count != 0:
            print("Images Processed: {}/{}".format(str(image_count), str(len(images))))
            print("Cascaded Images Created {}".format(str(files_created_count)))


if __name__ == "__main__":
    from sys import argv

    try:
        main(argv)
    except KeyboardInterrupt:
        pass
    sys.exit()

