import cv2
import matplotlib.pyplot as plt


plt.figure("black_box_vs_white_box")
black_box_img = cv2.imread("black_box.png")
white_box_img = cv2.imread("white_box.png")
plt.subplot(2, 1, 1)
plt.xticks([])
plt.yticks([])
plt.imshow(black_box_img[:, :, ::-1])

plt.subplot(2, 1, 2)
plt.xticks([])
plt.yticks([])
plt.imshow(white_box_img[:, :, ::-1])

plt.show()