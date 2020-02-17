import cv2
from pure_detector import PuReDetector

# read image as color and grayscale
img = cv2.imread("../eye.png", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# run detector
d = PuReDetector()
result, debug = d.detect_debug(gray)

# draw result on color image
# cv excepts tuples of ints and semi-axes
ellipse = result["ellipse"]
cv2.ellipse(
    img,
    center=tuple(int(v) for v in ellipse["center"]),
    axes=tuple(int(v / 2) for v in ellipse["axes"]),
    angle=ellipse["angle"],
    startAngle=0,
    endAngle=360,
    color=(0, 0, 255),  # BGR
)

# show images, press any key to continue (and exit)
cv2.imshow("img", img)
cv2.imshow("debug", debug)
cv2.waitKey(-1)
