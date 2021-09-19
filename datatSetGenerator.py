import numpy as np
import cv2
from PIL import Image as im
import pandas as pd
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

'''
images = pd.read_csv("sign_mnist_train.csv")

images_numpy = pd.DataFrame.to_numpy(images)
for image in images_numpy:
    name = image[0]
    image_arr = (np.array(image[1:785])).reshape(28, 28)
    print(image_arr)
    rgb_img = np.array([[[0, 0, 0] for i in range(0,28)] for j in range(0,28)])
    for i in range(0, 28):
        for j in range(0, 28):
#            print(image_arr[i][j])
            rgb_img[i][j] = [image_arr[i][j], image_arr[i][j], image_arr[i][j]]
#    detector = handDetector()
    img = im.fromarray(rgb_img.astype(np.uint8))
    img.save("test.png")
#    markedImg = detector.findHands(img)
#    markedImg("test2.png")
    break
'''


IMAGE_FILES = ["IMG_1140.jfif"]
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    im.fromarray(annotated_image.astype(np.uint8)).save('test2.png')