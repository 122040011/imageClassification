

#include <opencv2/ml.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>


#include <iostream>
#include "main.h"

using namespace cv::ml;
using namespace cv;
using namespace std;


model::model(String tr1, String tr2, String te1, String te2) {
    //Creates model
    KNearestModel = KNearest::create();

    //Sets path to (train & test) (images & labels)
    trainImagePath = tr1;
    trainLabelPath = tr2;
    testImagePath = te1;
    testLabelPath = te2;
}

void model::load() {

    //READ Train Images Header
    ifstream trainImageFile(trainImagePath, ios::binary);

    char magicNum[4];
    trainImageFile.read(magicNum, 4);
    char imageByte[4];
    trainImageFile.read(imageByte,4);
    char rowByte[4];
    trainImageFile.read(rowByte,4);
    char colByte[4];
    trainImageFile.read(colByte,4);

    /*Gets number of images, row/image, column/image
     * no. of images = 60000
     * row/image = 28
     * col/image = 28
     */
    int noImage = (static_cast<unsigned char>(imageByte[0])<<24) | (static_cast<unsigned char>(imageByte[1])<<16) | (static_cast<unsigned char>(imageByte[2])<<8) | (static_cast<unsigned char>(imageByte[3])<<0);
    int noRow = (static_cast<unsigned char>(rowByte[0])<<24) | (static_cast<unsigned char>(rowByte[1])<<16) | (static_cast<unsigned char>(rowByte[2])<<8) | (static_cast<unsigned char>(rowByte[3])<<0);
    int noCol = (static_cast<unsigned char>(colByte[0])<<24) | (static_cast<unsigned char>(colByte[1])<<16) | (static_cast<unsigned char>(colByte[2])<<8) | (static_cast<unsigned char>(colByte[3])<<0);

    //Read Label Header
    ifstream trainLabelFile(trainLabelPath, ios::binary);

    char magicNum2[4];
    trainLabelFile.read(magicNum2,4);
    char imageByte2[4];
    trainLabelFile.read(imageByte2, 4);

    int noImage2 = (static_cast<unsigned char>(imageByte2[0])<<24) | (static_cast<unsigned char>(imageByte2[1])<<16) | (static_cast<unsigned char>(imageByte2[2])<<8) | (static_cast<unsigned char>(imageByte2[3])<<0);

    //Checks if no of images and labels are equal
    if(noImage!=noImage2) cout<<"Image & Label sets are not equal"<<endl;
    trainImageCount = noImage;

    //-------------------------------------------

    int pixel = noRow*noCol;
    trainImages = Mat::zeros(trainImageCount, 784,CV_32F);
    trainLabels = Mat::zeros(trainImageCount, 1, CV_32F);

    for(int i=0; i<noImage; i++){

        vector<unsigned char> image(pixel);
        vector<unsigned char> tempLabel(1);

        //Reads labels and images into vector
        trainLabelFile.read((char*)tempLabel.data(), 1);
        trainImageFile.read((char*)(image.data()), pixel);


        for(int j=0; j<image.size(); j++){
            //Store vector into Mat
            trainImages.at<float>(i, j) = image[j];
            if(j==756) break;
        }

        //Store vector into Mat
        trainLabels.at<float>(i, 0) = (int)tempLabel[0];

    }
    cout<<"Finished Loading Train Data"<<endl;
    trainImageFile.close();
    trainLabelFile.close();



}

void model::test(){

    int trial = 0;
    int success = 0;


    //READ Train Images Header
    ifstream testImageFile(testImagePath, ios::binary);

    char magicNum[4];
    testImageFile.read(magicNum, 4);
    char imageByte[4];
    testImageFile.read(imageByte,4);
    char rowByte[4];
    testImageFile.read(rowByte,4);
    char colByte[4];
    testImageFile.read(colByte,4);

    int noImage = (static_cast<unsigned char>(imageByte[0])<<24) | (static_cast<unsigned char>(imageByte[1])<<16) | (static_cast<unsigned char>(imageByte[2])<<8) | (static_cast<unsigned char>(imageByte[3])<<0);
    int noRow = (static_cast<unsigned char>(rowByte[0])<<24) | (static_cast<unsigned char>(rowByte[1])<<16) | (static_cast<unsigned char>(rowByte[2])<<8) | (static_cast<unsigned char>(rowByte[3])<<0);
    int noCol = (static_cast<unsigned char>(colByte[0])<<24) | (static_cast<unsigned char>(colByte[1])<<16) | (static_cast<unsigned char>(colByte[2])<<8) | (static_cast<unsigned char>(colByte[3])<<0);

    //Read Label Header (magicNum and imageByte)
    ifstream testLabelFile(testLabelPath, ios::binary);

    char magicNum2[4];
    testLabelFile.read(magicNum2,4);
    char imageByte2[4];
    testLabelFile.read(imageByte2, 4);

    int noImage2 = (static_cast<unsigned char>(imageByte2[0])<<24) | (static_cast<unsigned char>(imageByte2[1])<<16) | (static_cast<unsigned char>(imageByte2[2])<<8) | (static_cast<unsigned char>(imageByte2[3])<<0);

    if(noImage!=noImage2) cout<<"Image & Label sets are not equal"<<endl;
    testImageCount = noImage;

    //-------------------------------------------



    int pixel = noRow*noCol;
    testImages = Mat::zeros(testImageCount, 784,CV_32F);
    testLabels = Mat::zeros(testImageCount, 1, CV_32F);

    for(int i=0; i<testImageCount; i++){


        vector<unsigned char> image(pixel);
        vector<unsigned char> tempLabel(1);

        //Reads File into vector
        testLabelFile.read((char*)(tempLabel.data()), 1);    //reads Label
        testImageFile.read((char*)(image.data()), pixel);


        for(int j=0; j<image.size(); j++){

            //Stores vector into Mat (pixel by pixel)
            testImages.at<float>(i, j) = image[j];
            if(j==756) break;
        }

        //Stores vector into Mat
        testLabels.at<float>(i, 0) = (int)tempLabel[0];

    }
    cout<<"Finished Loading Test Data"<<endl;
    testImageFile.close();
    testLabelFile.close();

    //Generates Mat object to store predicted output
    Mat expected = Mat::zeros(testImageCount, 1, CV_32F);


    //Generates expected(predicted) output (Checks 3 points (KNN))
    cout<<"Generating Results"<<endl;
    KNearestModel->findNearest(testImages, 3, expected);

    //Checks Result
    cout<<"Checking Results"<<endl;
    for(int i=0; i<noImage; i++){
        //Uncomment the line below to check one by one
        //cout<<(int)expected.at<float>(i,0)<<" | "<<(int)testLabels.at<float>(i,0)<<endl;

        //Checks if predicted output and testLabel is equal
        if((int)expected.at<float>(i,0)==(int)testLabels.at<float>(i,0)) success++;
        trial++;
    }

    //Output Accuracy
    cout << "Accuracy= " << success << "/" << trial << " = " << ((float)success)/trial;

}

void model::train(){
    cout<<"Training..."<<endl;

    //Trains KNearestModel using trainImages and trainLabels
    KNearestModel->train(trainImages, ROW_SAMPLE, trainLabels);
    cout<<"Training Complete"<<endl;

    //Saves model in "saveFile/saveModel"
    KNearestModel->save("/Users/feivelehren/CLionProjects/Project3002/saveFile/saveModel");
    cout<<"Saved"<<endl;
}

String model::read(String imagePath){
    //Reads image into Mat file
    Mat image = imread(imagePath);

    //Show image to user
    imshow("test", image);
    waitKey(0);

    //Resize image
    resize(image, image, Size(784, 1));
    Mat resizedImage = Mat::zeros(1, 784, CV_32F);
    for(int i = 0; i<784; i++){
        resizedImage.at<float>(0, i) = (float)image.at<float>(0,i);
    }

    //Use the model to predict digit
    Mat expected = Mat::zeros(1, 1, CV_32F);
    KNearestModel->findNearest(resizedImage, 3, expected);

    //Return expected value
    return to_string((int) expected.at<float>(0,0));
}

int main() {

    String trainImagePath = "/Users/feivelehren/CLionProjects/Project3002/assets/train-images-idx3-ubyte";
    String trainLabelPath = "/Users/feivelehren/CLionProjects/Project3002/assets/train-labels-idx1-ubyte";
    String testImagePath = "/Users/feivelehren/CLionProjects/Project3002/assets/t10k-images-idx3-ubyte";
    String testLabelPath = "/Users/feivelehren/CLionProjects/Project3002/assets/t10k-labels-idx1-ubyte";

    //Create model and set path
    model model(trainImagePath, trainLabelPath, testImagePath, testLabelPath);

    //Load training images and labels
    model.load();

    //Train model using training images and labels
    model.train();

    //Read test.png and output digit
    cout<<"Digit in assets/test.png= "<<model.read("/Users/feivelehren/CLionProjects/Project3002/assets/test.png")<<endl;
    cout<<"Press any key to continue"<<endl;

    //Test model using test images and labels
    model.test();

    return 0;
}


