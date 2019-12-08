from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
img=Image.open('C:/dataset/train/face/face00001.pgm')
plt.figure("test")
plt.imshow(img)
currentAxis = plt.gca()
rect = patches.Rectangle((12, 0), 4, 1, linewidth=1, edgecolor='b', facecolor='black')
rect1 = patches.Rectangle((12, 1), 4, 1, linewidth=1, edgecolor='b', facecolor='white')
currentAxis.add_patch(rect)
currentAxis.add_patch(rect1)

plt.show()
