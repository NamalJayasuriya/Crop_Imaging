
# from flirpy.camera.lepton import Lepton
# import cv2
#
# cam = Lepton()
#
# while True:
#     image = cam.grab()
#     print(image)
#     # print(cam.frame_count)
#     # print(cam.frame_mean)
#     # print(cam.ffc_temp_k)
#     print(cam.fpa_temp_k)
#     cv2.imshow('Lepton', image)
#     k = cv2.waitKey(33)
#     if k == 27:  # Esc key to stop
#         break
# cv2.destroyAllWindows()
# cam.close()



import cv2
import numpy as np
from flirpy.camera.lepton import Lepton

with Lepton() as camera:
    while True:
        img = camera.grab().astype(np.float32)


        # Rescale to 8 bit
        #img = 255 * (img - img.min()) / (img.max() - img.min())
        print(img.min(), img.max())
        # Apply colourmap - try COLORMAP_JET if INFERNO doesn't work.
        # You can also try PLASMA or MAGMA
        img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_INFERNO)
        #img_col = cv2.applyColorMap(cv2.convertScaleAbs(img, alpha=0.09), cv2.COLORMAP_INFERNO)
        #img_col = np.asanyarray(img)
        img_col = cv2.resize(img_col, (1280, 720))
        cv2.imshow('Lepton', img_col)
        if cv2.waitKey(1) == 27:
            break  # esc to quit

cv2.destroyAllWindows()
camera.close()