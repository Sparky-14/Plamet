import cv2
import numpy as np
import os
import imutils
from tensorflow.keras.models import load_model
import pytesseract
import re
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from image_to_text import predict_number_plate
from paddleocr import PaddleOCR

def run_algorithm(directory):
    
    print("Running algorithm on directory:", directory)

    messagebox.showinfo("Success", "Algorithm ran successfully on directory: " + directory)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


    model = load_model('helmet-nonhelmet_cnn.h5')
    print('model loaded!!!')

    cap = cv2.VideoCapture(directory)
    COLORS = [(0,255,0),(0,0,255)]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter('output.avi', fourcc, 5,(888,500))

   #pytesseract.pytesseract.tesseract_cmd = r'/Users/junaid/Documents/coding/miniproject/lib/python3.10/site-packages/pytesseract'

    def helmet_or_nohelmet(helmet_roi):
        try:
            helmet_roi = cv2.resize(helmet_roi, (224, 224))
            helmet_roi = np.array(helmet_roi,dtype='float32')
            helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
            helmet_roi = helmet_roi/255.0
            return int(model.predict(helmet_roi)[0][0])
        except:
                pass

    ret = True

    while ret:

        ret, img = cap.read()
        img = imutils.resize(img,height=500)
        height, width = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        confidences = []
        boxes = []
        classIds = []
        e=0
        e += 1

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)

                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    classIds.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                color = [int(c) for c in COLORS[classIds[i]]]
                
                if classIds[i]==0: #bike
                    helmet_roi = img[max(0,y):max(0,y)+max(0,h)//4,max(0,x):max(0,x)+max(0,w)]
                else: #number plate
                    x_h = x-60
                    y_h = y-350
                    w_h = w+100
                    h_h = h+100
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 7)
                    if y_h>0 and x_h>0:
                        h_r = img[y_h:y_h+h_h , x_h:x_h +w_h]
                        c = helmet_or_nohelmet(h_r)
                        plate_roi = img[y-10:y+h+10, x-10:x+w+10]
                        # crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                        # # Sharpening
                        # crop = cv2.GaussianBlur(crop, (3,3), 0)
                        # crop = cv2.addWeighted(crop, 1.5, blur, -0.5, 0)
                        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                        crop = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        
                        plate_filename = f"captured_plates/plate_{e}.jpg"
                        if c:
                            cv2.imwrite(plate_filename, plate_roi)
                            try:
                                vechicle_number, conf = predict_number_plate(gray, ocr)
                                if vechicle_number and conf:
                                    print(vechicle_number)
                            except TypeError:
                                text = pytesseract.image_to_string(gray, config='--psm 11')
                                number_plate_text = re.sub('[^A-Za-z0-9]','', text)
                                print(number_plate_text)
                        cv2.putText(img,['helmet','no-helmet'][c],(x,y-100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)                
                        cv2.rectangle(img, (x_h, y_h), (x_h + w_h, y_h + h_h),(255,0,0), 10)

        writer.write(img)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) == 27:
            break

    writer.release()
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def browse_directory():
    
    directory = file_path= filedialog.askopenfilename(title = "Select A File", filetypes = (("mov files", ".mov"), ("mp4", ".mp4"), ("wmv", ".wmv"), ("avi", ".avi")))
    directory_entry.delete(0, tk.END)  
    directory_entry.insert(0, directory)  

def on_submit():
    directory = directory_entry.get()
    if directory:
        run_algorithm(directory)
    else:
        messagebox.showwarning("Warning", "Please select a directory.")


root = tk.Tk()
root.title("Helmet Detection App")


frame = tk.Frame(root)
frame.pack(pady=20)


directory_entry = tk.Entry(frame, width=50)
directory_entry.pack(side=tk.LEFT, padx=(0, 10))


browse_button = tk.Button(frame, text="Browse", command=browse_directory)
browse_button.pack(side=tk.LEFT)


submit_button = tk.Button(root, text="Submit", command=on_submit)
submit_button.pack(pady=10)


root.mainloop()
