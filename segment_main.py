"""

A program attempting to segment incisors in radiographs.

 Manual and Automatic settings are possible.

 Current setup does leave one out analysis

 Result is a binary image in which non zero values are incisor pixels.


"""

import cv2
import numpy as np

from time import time

import xray_code as xray
from landmark_code import load_landmarks, load_ground, load_all_landmarks
from tooth_code import Model
from procrustes_analysis import procrustes
from active_shape_model import Active_Shape_Model
from manual_selection_tool import manual_selection
from automatic_selection_process import automatic_selection
import plotting_code

def main(manual=True):

    # CODE SETTINGS:

    for x in range(1,15):

        print("radiograph: " + str(x))
        print("")
        x1 = time()

        leave_one_out=True

        leave_out_index=x

        radiographs_test_set_dir = './Data/Radiographs/'

        landmarks_dir = './Data/Landmarks/'

        train_images = xray.load_images(exclude=leave_out_index)

        test_image = xray.load_images(specific=leave_out_index)

        incisor_segmentations = []

        for incisor in range(1, 9):

            print("incisor: "+str(incisor))

            landmarks = load_landmarks(landmarks_dir, incisor, mirrored=True,exclude=leave_out_index)
            # landmarks = load_landmarks(landmarks_dir, incisor, mirrored=True)

            # testing multiple tooth model

            # landmarks = load_all_landmarks(landmarks_dir, exclude=2)

            tooth_model = Model(incisor)

            # PERFORM PROCRUSTES ANALYSIS

            mean_tooth, aligned_landmarks = procrustes(landmarks)

            # BUILD ACTIVE SHAPE MODEL

            a_s_model = Active_Shape_Model(mean_tooth, aligned_landmarks)


            # TRAIN MODEL ON TRAINING IMAGES + ACTIVE SHAPE MODEL

            tooth_model.train(landmarks, train_images, a_s_model)

            if manual:
                # MANUALLY SELECT TOOTH LOCATION
                manual_selected = manual_selection(a_s_model.mean_shape, test_image)
                selection = manual_selected
            else:
                # AUTOMATIC TOOTH LOCATOR
                automatic_selected = automatic_selection(a_s_model, test_image, incisor, show=False)
                selection = automatic_selected

            fit = tooth_model.fit(selection, test_image)

            incisor_segmentations.append(fit)

        plotting_code.plot_landmarks_on_image(incisor_segmentations,  test_image,show=False,save=True,index=leave_out_index, wait=True)

        if leave_one_out:

            evaluation_results=[]
            evaluation_landmarks = []

            for idx, pred in enumerate(incisor_segmentations):
                ground = load_ground(landmarks_dir, idx + 1, l_o_o=leave_out_index)
                F = calculate_f(test_image,leave_out_index,idx, pred)
                evaluation_results.append(F)
                evaluation_landmarks.append([ground,pred])

            plotting_code.plot_evaluations(evaluation_landmarks, test_image,show=False,save=True,index=leave_out_index, wait=True)

            for idx, itm in enumerate(evaluation_results):
                print(str(idx+1)+ " : " + str(itm))

            # print("")
            # print("F: " + str(sum(evaluation_results)/len(evaluation_results)))
            print("")

        results_output(test_image,incisor_segmentations, leave_out_index)

        print(time()-x1)

        print("Done!")


def calculate_f(test_image, test_index, index, predicted):

    # write out prediction to file

    height, width, _ = test_image.shape
    image2 = np.zeros((height, width), np.int8)
    mask = np.array([predicted.points], dtype=np.int32)
    cv2.fillPoly(image2, [mask], 255)
    maskimage2 = cv2.inRange(image2, 1, 255)
    out = cv2.bitwise_and(test_image, test_image, mask=maskimage2)
    cv2.imwrite('./Data/FinalOutput/Predict/%02d-%d.png' % ( test_index, index,), out)

    ground = cv2.imread('./Data/Segmentations/%02d-%d.png' % (test_index, index,), 0)
    (_, ground) = cv2.threshold(ground, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    fit = cv2.imread('./Data/FinalOutput/Predict/%02d-%d.png' % (test_index, index,), 0)
    (_, fit) = cv2.threshold(fit, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)



    TP = fit & ground
    FP = fit - ground
    FN = ground - fit

    return float(2 * (TP / 255).sum()) / (2 * (TP / 255).sum() + (FP / 255).sum() + (FN / 255).sum())


def results_output(test_image,incisor_segmentations, index):
    ## save tooth region segmented
    height, width, x = test_image.shape
    maskimage = 0
    for segmentation in incisor_segmentations:
        image2 = np.zeros((height, width), np.int8)
        mask = np.array([segmentation.points], dtype=np.int32)
        cv2.fillPoly(image2, [mask], 255)
        maskimage += cv2.inRange(image2, 1, 255)
    segmented = cv2.bitwise_and(test_image, test_image, mask=maskimage)
    # for i in range(segmented.shape[0]):
    #     for j in range(segmented.shape[1]):
    #         if (segmented[i,j] != [0, 0, 0]).all():
    #             segmented[i, j] = [255,255,255]
    cv2.imwrite('./Data/FinalOutput/result-%d.png' % (index), segmented)



if __name__ == '__main__':

    x0 = time()
    main(manual=False)

    print(time()-x0)



