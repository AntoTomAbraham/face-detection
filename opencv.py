import cv2
image_path=r"C:\Users\91854\PycharmProjects\sample\hg.jpg"
xml_path=r"C:\Users\91854\PycharmProjects\sample\hgg.xml"
j=cv2.CascadeClassifier(xml_path)
img=cv2.imread(image_path)

grey_colour=cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

k = j.detectMultiScale(
    grey_colour,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(20,20),
    flags=cv2.CASCADE_SCALE_IMAGE,
)


print("found {0} face".format(len(k)))

for(x,y,w,h) in k:
    cv2.rectangle(img,(x,y),(x+w , y+h),(0,0,255),1)

cv2.imshow("FACE DECTECTED",img)
cv2.waitKey(0)