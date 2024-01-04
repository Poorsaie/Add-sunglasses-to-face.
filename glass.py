import cv2

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image for sunglasses
sunglasses_image = cv2.imread('sunglasses.png', cv2.IMREAD_UNCHANGED)

# Prompt the user to enter the path to the image
image_path = input("Enter the path to the image: ")

# Read the image
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Process each detected face
for (x, y, w, h) in faces:
    # Calculate the position and size of the eyes region
    eyes_x = int(x + w / 6)
    eyes_y = int(y + h / 3.5)
    eyes_w = int(w * 2 / 3)
    eyes_h = int(h / 3)

    # Resize the sunglasses image to match the eyes region
    sunglasses_resized = cv2.resize(sunglasses_image, (eyes_w, eyes_h))

    # Calculate the coordinates for the sunglasses overlay
    x_offset = eyes_x
    y_offset = eyes_y

    # Overlay the sunglasses image on the original image
    for c in range(3):
        image[y_offset:y_offset+eyes_h, x_offset:x_offset+eyes_w, c] = (
            sunglasses_resized[:, :, c] * (sunglasses_resized[:, :, 3] / 255.0) +
            image[y_offset:y_offset+eyes_h, x_offset:x_offset+eyes_w, c] * (1.0 - sunglasses_resized[:, :, 3] / 255.0)
        )

# Display the modified image
cv2.imshow("Sunglasses", image)
cv2.waitKey(0)
cv2.destroyAllWindows()