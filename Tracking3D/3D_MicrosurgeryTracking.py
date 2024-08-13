import cv2
import os
import numpy as np
from ultralytics import YOLO

# Register of centroids, use "ant" if there are no markers in the frame
cen_x_ant = [0, 0, 0, 0]    # x -> [Green_CamL, Pink_CamL, Green_CamR, Pink_CamR]
cen_y_ant = [0, 0, 0, 0]

cen_x_ant_8n = [0, 0, 0, 0]    # x -> [Green_CamL, Pink_CamL, Green_CamR, Pink_CamR]
cen_y_ant_8n = [0, 0, 0, 0]

r_hand = [0, 0, 0]
l_hand = [0, 0, 0]


#  #################  Computing centroids  Mitracks3D #####################################################
def centroid_es(centres, i):
    # Centroids:
    global cen_x_ant
    global cen_y_ant
    if np.sum(centres) > 0:     # Just if there are markers
        M = cv2.moments(centres)
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])

        cen_x_ant[i] = x
        cen_y_ant[i] = y

    else:                       # If there are no markers, use the last know position
        x = cen_x_ant[i]
        y = cen_y_ant[i]
    return x, y


#  #################  Color segmentation and triangulation Mitracks3D  ####################################
def segment_colors(image1, image2):

    cen_x = [0, 0, 0, 0]
    cen_y = [0, 0, 0, 0]
    cx = 0
    cy = 0

    # Image from BGR to HSV color space
    hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

    # LEFT CAM Defining the range of values HSV for Green, Pink and Blue
    # GREEN
    cl_lower_G = (80, 80, 80)
    cl_upper_G = (92, 255, 255)
    # PINK
    cl_lower_P = (144, 70, 135)
    cl_upper_P = (162, 255, 255)
    # # AZUL
    # lower_B = (100, 200, 170)
    # upper_B = (110, 255, 255)

    # # RIGHT CAM Defining the range of values HSV for Green, Pink and Blue
    # GREEN
    cr_lower_G = (80, 60, 60)
    cr_upper_G = (95, 255, 255)
    # PINK
    cr_lower_P = (144, 60, 125)
    cr_upper_P = (162, 255, 255)

    # Getting just the markers in binary images
    cl_mask_G = cv2.inRange(hsv1, cl_lower_G, cl_upper_G)
    cl_mask_P = cv2.inRange(hsv1, cl_lower_P, cl_upper_P)
    cr_mask_G = cv2.inRange(hsv2, cr_lower_G, cr_upper_G)
    cr_mask_P = cv2.inRange(hsv2, cr_lower_P, cr_upper_P)

    kernel = np.ones((3, 3), np.uint8)  # 3,3
    cl_mask_G = cv2.morphologyEx(cl_mask_G, cv2.MORPH_OPEN, kernel)
    cl_mask_P = cv2.morphologyEx(cl_mask_P, cv2.MORPH_OPEN, kernel)
    cr_mask_G = cv2.morphologyEx(cr_mask_G, cv2.MORPH_OPEN, kernel)
    cr_mask_P = cv2.morphologyEx(cr_mask_P, cv2.MORPH_OPEN, kernel)

    #  Getting the centroids of the markers
    for i in range(0, 4):
        if i == 0:
            cx, cy = centroid_es(cl_mask_G, i)
        if i == 1:
            cx, cy = centroid_es(cl_mask_P, i)
        if i == 2:
            cx, cy = centroid_es(cr_mask_G, i)
        if i == 3:
            cx, cy = centroid_es(cr_mask_P, i)

        cen_x[i] = cx
        cen_y[i] = cy

    #  Show the centroids on the images
    camL_seg = cv2.circle(image1,   (cen_x[0], cen_y[0]), 15, (0, 181, 0),      2, cv2.LINE_AA)
    camL_seg = cv2.circle(camL_seg, (cen_x[1], cen_y[1]), 15, (255, 0, 255),    2, cv2.LINE_AA)
    camR_seg = cv2.circle(image2,   (cen_x[2], cen_y[2]), 15, (0, 181, 0),      2, cv2.LINE_AA)
    camR_seg = cv2.circle(camR_seg, (cen_x[3], cen_y[3]), 15, (255, 0, 255),    2, cv2.LINE_AA)

    #  #################  Triangulating (Taking y as x due the cameras' orientation)

    #       Main CAM LEFT
    # RIGHT hand
    zv = (3.1 * 1430) / ((cen_y[0] + 395 - 545) - (cen_y[2] + 75 - 545))   # D*B / (y+ShiftROI-CentralPoint_y)
    xv = (zv * (cen_y[0] + 395 - 545)) / 1430      # zv * (y + ShiftROI-CentralPoint_y) / focus
    yv = (zv * (cen_x[0] + 817 - 948)) / 1430      # (zv * x+ShiftROI_x- CentralPoint_x)

    # LEFT hand
    zp = (3.1 * 1430) / ((cen_y[1] + 395 - 545) - (cen_y[3] + 75 - 545))   # D*B / (y+ShiftROI-CentralPoint_y)
    xp = (zp * (cen_y[1] + 395 - 545)) / 1430      # zv * (y + ShiftROI-CentralPoint_y) / focus
    yp = (zp * (cen_x[1] + 817 - 948)) / 1430      # (zv * x+ShiftROI_x- CentralPoint_x)

    r_xyz = [xv, yv, zv]
    l_xyz = [xp, yp, zp]

    return camL_seg, camR_seg, r_xyz, l_xyz


def yolo_hsvL(image1):

    # Make the image from BGR to HSV color space
    hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)

    # LEFT CAM Defining the range of values HSV for Green, Pink and Blue
    # GREEN
    cl_lower_G = (80, 80, 80)
    cl_upper_G = (92, 255, 255)
    # PINK
    cl_lower_P = (144, 70, 135)
    cl_upper_P = (162, 255, 255)

    # Getting just the markers in binary images
    cl_mask_G = cv2.inRange(hsv1, cl_lower_G, cl_upper_G)
    cl_mask_P = cv2.inRange(hsv1, cl_lower_P, cl_upper_P)

    kernel = np.ones((3, 3), np.uint8)  # 3,3
    cl_mask_G = cv2.morphologyEx(cl_mask_G, cv2.MORPH_OPEN, kernel)
    cl_mask_P = cv2.morphologyEx(cl_mask_P, cv2.MORPH_OPEN, kernel)

    if np.count_nonzero(cl_mask_G) > np.count_nonzero(cl_mask_P):
        hand_id = 1
    elif np.count_nonzero(cl_mask_G) < np.count_nonzero(cl_mask_P):
        hand_id = 2
    else:
        hand_id = 0
        print('Lost marker')

    return hand_id


def yolo_hsvR(image2):
    # Make the image from BGR to HSV color space
    hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

    # # RIGHT CAM Defining the range of values HSV for Green, Pink and Blue
    # GREEN
    cr_lower_G = (80, 60, 60)
    cr_upper_G = (95, 255, 255)
    # PINK
    cr_lower_P = (144, 60, 125)
    cr_upper_P = (162, 255, 255)

    # Getting just the markers in binary images
    cr_mask_G = cv2.inRange(hsv2, cr_lower_G, cr_upper_G)
    cr_mask_P = cv2.inRange(hsv2, cr_lower_P, cr_upper_P)

    kernel = np.ones((3, 3), np.uint8)  # 3,3
    cr_mask_G = cv2.morphologyEx(cr_mask_G, cv2.MORPH_OPEN, kernel)
    cr_mask_P = cv2.morphologyEx(cr_mask_P, cv2.MORPH_OPEN, kernel)

    if np.count_nonzero(cr_mask_G) > np.count_nonzero(cr_mask_P):
        hand_id = 1
    elif np.count_nonzero(cr_mask_G) < np.count_nonzero(cr_mask_P):
        hand_id = 2
    else:
        hand_id = 0                 # If the tweezer was detected but there are not a marker, WARNING, solve this later
        print('Lost marker')

    return hand_id


#  #################  YOLOv8X  ######################################################################################

def yolo8n_tri(image1, image2):

    global cen_x_ant_8n
    global cen_y_ant_8n
    trh = ''

# Test:
    frame1 = image1
    frame2 = image2

    cx_g_cl = cen_x_ant_8n[0]
    cx_p_cl = cen_x_ant_8n[1]
    cx_g_cr = cen_x_ant_8n[2]
    cx_p_cr = cen_x_ant_8n[3]
    cy_g_cl = cen_y_ant_8n[0]
    cy_p_cl = cen_y_ant_8n[1]
    cy_g_cr = cen_y_ant_8n[2]
    cy_p_cr = cen_y_ant_8n[3]

#
    global frame_count_8n
    global threshold
    # hand_id = 0                 # Temporally use the marker to identify if the tweezer is left or right
    color0 = (255, 255, 255)

    classes = [1]

    for piv in range(0, 2):

        if piv == 0:
            frame = image1
        else:
            frame = image2

        results = model_8n.predict(frame, classes=classes, conf=0.4, iou=0.1)[0]  # Make som experiments modifying these parameters

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:

                if class_id == 1:  # Tweezer

                    # call to YOLO_HSV
                    im_roi = frame[int(y1):int(y2), int(x1):int(x2)]

                    if piv == 0:
                        hand_id = yolo_hsvL(im_roi)
                    else:
                        hand_id = yolo_hsvR(im_roi)

                    if hand_id == 1:        # Right hand detected
                        trh = 'RH_'
                        color0 = (0, 181, 0)

                        # Centroid
                        if piv == 0:
                            cx_g_cl = x1 + ((x2 - x1) / 2)
                            cy_g_cl = y1 + ((y2 - y1) / 2)
                        else:
                            cx_g_cr = x1 + ((x2 - x1) / 2)
                            cy_g_cr = y1 + ((y2 - y1) / 2)

                    elif hand_id == 2:      # Left hand detected
                        trh = 'LH_'
                        color0 = (255, 0, 255)

                        # Centroid
                        if piv == 0:
                            cx_p_cl = x1 + ((x2 - x1) / 2)
                            cy_p_cl = y1 + ((y2 - y1) / 2)
                        else:
                            cx_p_cr = x1 + ((x2 - x1) / 2)
                            cy_p_cr = y1 + ((y2 - y1) / 2)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color0, 2)
                cv2.putText(frame, trh + results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color0, 2, cv2.LINE_AA)

        # Just for the LEFT cam
        if piv == 0:
            frame_count_8n += 1
            frame1 = frame
        else:
            frame2 = frame

    # Here, we should have the 4 markers... OK
    cen_x = [cx_g_cl, cx_p_cl, cx_g_cr, cx_p_cr]  # x -> [Green_CamL, Pink_CamL, Green_CamR, Pink_CamR]
    cen_y = [cy_g_cl, cy_p_cl, cy_g_cr, cy_p_cr]

    cen_x_ant_8n = cen_x
    cen_y_ant_8n = cen_y

    # Triangulation
    #       Main CAM LEFT
    # RIGHT hand
    zv = (3.1 * 1430) / ((cen_y[0] + 395 - 545) - (cen_y[2] + 75 - 545))   # D*B / (y+ShiftROI-CentralPoint_y)
    xv = (zv * (cen_y[0] + 395 - 545)) / 1430      # zv * (y + ShiftROI-CentralPoint_y) / focus
    yv = (zv * (cen_x[0] + 817 - 948)) / 1430      # (zv * x+ShiftROI_x- CentralPoint_x)

    # LEFT hand
    zp = (3.1 * 1430) / ((cen_y[1] + 395 - 545) - (cen_y[3] + 75 - 545))   # D*B / (y+ShiftROI-CentralPoint_y)
    xp = (zp * (cen_y[1] + 395 - 545)) / 1430      # zv * (y + ShiftROI-CentralPoint_y) / focus
    yp = (zp * (cen_x[1] + 817 - 948)) / 1430      # (zv * x+ShiftROI_x- CentralPoint_x)

    r_xyz = [xv, yv, zv]
    l_xyz = [xp, yp, zp]

    return frame1, frame2, r_xyz, l_xyz

#  #################  Main definitions  ##############################################################################

# Path of the image files
path1 = os.path.dirname(os.path.abspath(__file__))  #             < --- THIS SCRIPT and example_trial1 folder TOGETHER
main_path = path1 + "\\example_trial1"

cl_folder_path = main_path + "\\left"
cr_folder_path = main_path + "\\right"

# Saving 3D data
save_3d_mitracks = main_path + "\\data_3D_mitracks.txt"
save_3d_YOLO8 = main_path + "\\data_3D_YOLO8.txt"

# VIDEOS
# Creating videos
#output_video_path_mt = main_path + "\\Mitracks3D_tracking.mp4"
output_video_path_8n = main_path + "\\Instruments_tracking.mp4"

height = 640
width = 640
# Define the videos codec and create the video files
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_out_8 = cv2.VideoWriter(output_video_path_8n, fourcc, 30, (width * 2, height * 2))

# cv2.namedWindow("Stereoscopic tracking", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Instruments tracking", cv2.WINDOW_AUTOSIZE)

# Create or open the files to start to register the 3D data
data_3dmt = open(save_3d_mitracks, 'w')
data_3d8n = open(save_3d_YOLO8, 'w')

# Load YOLO models
model_path_8n = os.path.join('.', 'best_Mitracks3D_YOLOv8n.pt')  # < --- OR best_Mitracks3D_YOLOv8m, or your own model
model_8n = YOLO(model_path_8n)  # load a custom model

threshold = 0.25    # original: 0.6


frame_count_8n = 0


# Get the list of files on the directory and sort them in numeric order; this is useful for the original images dataset
file_list = os.listdir(cl_folder_path)
file_list = sorted(file_list, key=lambda x: int(os.path.splitext(x)[0]))  # from 000000 to end

# Read and run through the image files
for filename in file_list:
    # Read the images
    cl_image_path = os.path.join(cl_folder_path, filename)
    cr_image_path = os.path.join(cr_folder_path, filename)

    cl_image = cv2.imread(cl_image_path)
    cr_image = cv2.imread(cr_image_path)

    # Check if there are images, and if they were well-read
    if cl_image is not None and cr_image is not None:

        cl_image_8n = np.copy(cl_image)
        cr_image_8n = np.copy(cr_image)

        # Find the markers and get the coordinates
        cam1, cam2, r_hand, l_hand = segment_colors(cl_image, cr_image)

        # Show the segmented images
        im_large = np.concatenate((cam1, cam2), axis=1)
        # cv2.imshow("Camera 1 and 2", im_large)

        # video_out_mt.write(im_large)

        print(filename)

        # Record the XYZ of mitracks on the txt file
        fila = '\t'.join("{:.3f}".format(elem) for elem in r_hand)
        data_3dmt.write(fila + '\t')
        fila = '\t'.join("{:.3f}".format(elem) for elem in l_hand)
        data_3dmt.write(fila + '\n')

        # now for YOLOv8X
        # Find the markers and get the coordinates
        cam1, cam2, r_hand, l_hand = yolo8n_tri(cl_image_8n, cr_image_8n)

        # Show the segmented images
        im_large2 = np.concatenate((cam1, cam2), axis=1)
        im_larger = cv2.vconcat([im_large, im_large2])

        # Formatting the video
        im_larger = cv2.line(im_larger, (0, 640), (640 * 2, 640), (0, 0, 0), 4, cv2.LINE_AA)
        im_larger = cv2.line(im_larger, (640, 0), (640, 640*2), (0, 0, 0), 2, cv2.LINE_AA)

        cv2.putText(im_larger, 'Mitracks3D', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(im_larger, 'YOLOv8 Tracking', (10, 640+50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (235, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Instruments tracking", im_larger)

        video_out_8.write(im_larger)   # Let's create a video to show the processed images.

        # Record the XYZ of mitracks on the txt file
        fila = '\t'.join("{:.3f}".format(elem) for elem in r_hand)
        data_3d8n.write(fila + '\t')
        fila = '\t'.join("{:.3f}".format(elem) for elem in l_hand)
        data_3d8n.write(fila + '\n')

        if cv2.waitKey(1) == ord('q'):
            break


cv2.destroyAllWindows()

video_out_8.release()

data_3dmt.close()
data_3d8n.close()
