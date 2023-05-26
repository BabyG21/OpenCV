import cv2
import numpy as np
import openai

openai.api_key = "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"

# Load the class labels from disk
rows = open('class_labels.txt').read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# Load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Initialize the video stream
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize the frame to 300x300 pixels (required by the model)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence > 0.2:
            # Extract the index of the class label from the `detections`
            idx = int(detections[0, 0, i, 1])

            # Draw the prediction on the frame
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Display the prediction
            label = "{}: {:.2f}%".format(classes[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Pass the detected object to GPT-3
            message = f"I see a {classes[idx]} with a confidence of {confidence * 100}%"
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": message}
                ])
            print(completion.choices[0].message['content'])

    # Display the resulting frame
    cv2.imshow("Frame", frame)
    
    # Break the loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
