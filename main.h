//
// Created by Feivel Ehren on 06/12/23.
//
#include <opencv2/ml.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv::ml;
using namespace cv;
using namespace std;

#ifndef PROJECT3002_MAIN_H
#define PROJECT3002_MAIN_H

#endif //PROJECT3002_MAIN_H

/*Configure path in main function before starting
 * Train images path
 * Train labels path
 * Test images path
 * Test labels path
 */

class model{
public:

    /*model Constructor
     * tr1 = training images path
     * tr2 = training labels path
     * te1 = test images path
     * te2 - test labels path*/
    model(String tr1, String tr2, String te1, String te2);


    /*
     * Reads training Images and training Labels
     * Loads into a Mat object
     */
    void load();


    /*public function train
     * After training images and labels are loaded into a Mat object
     * Uses trainImages and trainLabels to train KNearestModel
     * Saves the model in saveFile/saveModel
     */
    void train();


    /*
     * public function test
     * Loads testImages and testLabels into a Mat object
     * Uses the cv::ml::findNearest to get an 'expected Mat'
     * Compares expected(label) Mat with testLabel
     * Lists accuracy
     */
    void test();

    /*
     * Public function read
     * Reads a MNIST 28x28 image
     * Outputs expected digit based on KNN model
     */
    String read(String imagePath);

private:
    Ptr<KNearest> KNearestModel; //the KNN model
    String trainImagePath, trainLabelPath, testImagePath, testLabelPath; //Paths used to find the files
    Mat trainImages; //Mat object to store training images
    Mat trainLabels; //Mat object to store training labels
    Mat testImages; //Mat object to store test images
    Mat testLabels; //Mat object to store test labels

    int trainImageCount;
    int testImageCount;

};


void train();