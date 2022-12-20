
            encoded = b64encode(img_data)
            encoded = encoded.decode('ascii')
            mime = "image/jpeg"
            uri = "data:%s;base64,%s" % (mime, encoded)
            from PIL import Image
            import pytesseract
            import numpy as np
            import cv2
            import os
            import base64

            pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR/tesseract.exe'
            
            img = np.fromstring(img_data, np.uint8)
            img = cv2.imdecode(img,cv2.IMREAD_UNCHANGED)
            h,w,_ = img.shape
            boxes = pytesseract.image_to_boxes(img)

            for b in boxes.splitlines():
                b = b.split(' ')
                img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

            bb= cv2.imencode('.jpg', img)[1]
            akhir = b64encode(bb)
            akhir = akhir.decode('ascii')
            
            mime = "image/jpg"
            uriii = "data:%s;base64,%s" % (mime, akhir)
           
            
            image = np.fromstring(img_data, np.uint8)
            image = cv2.imdecode(image,cv2.IMREAD_UNCHANGED)
            # get grayscale image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #noise removal
            #noise=cv2.medianBlur(gray,3)
            # thresholding# converting it to binary image by Thresholding
            # this step is require if you have colored image because if you skip this part
            # then tesseract won’t able to detect text correctly and this will give incorrect #result
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            frame_buff= cv2.imencode('.jpg', thresh)[1]
            akhir = b64encode(frame_buff)
            akhir = akhir.decode('ascii')
            
            mime = "image/jpg"
            urii = "data:%s;base64,%s" % (mime, akhir)
            #Configuration
            config = ('-l eng --oem 3 --psm 3')
            # pytessercat
            text = pytesseract.image_to_string(thresh,config=config)
            print(text)


            

            from django.db.models import F
            qs = Translate.objects.all()

            
            qs = qs.filter(Indonesia=text)