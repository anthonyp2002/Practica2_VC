#include <opencv2/opencv.hpp>
#include <string.h>
#include <iostream>

using namespace cv;
using namespace std;

VideoCapture vcap("http://166.247.77.253:81/mjpg/video.mjpg");
Mat frame,imgO,imgGryscale,imghist,imgCLAHE,imgRealce,imgDM;
Mat background;

Mat updateBackground(const Mat& currentFrame, const Mat& prevBg, double alpha) {
    Mat bg;
    addWeighted(currentFrame, alpha, prevBg, 1.0 - alpha, 0.0, bg);
    return bg;
}

int main(){
    int MAX_FRAMES = 1000;
    int THRESH = 60;
    int ASSIGN_VALUE = 255;
    double ALPHA = 0.1;
    double start_time = (double)getTickCount();
    int count = 0;
    while (true) {
        vcap.read(frame);
        resize(frame,imgO,Size(800,400));
        cvtColor(imgO,imgGryscale,COLOR_BGR2GRAY);

        count++;
        double elapsed_time = ((double)getTickCount() - start_time) / getTickFrequency();

        //FPS
        double fps = count / elapsed_time;
        String fpsText = "FPS: " + to_string(fps);
        

        //Detector de movimiento
        if (count == 1) {
            background = imgGryscale.clone();
        } else {
            Mat diff, motionMask;
            absdiff(background, imgGryscale, diff);
            threshold(diff, motionMask, THRESH, ASSIGN_VALUE, THRESH_BINARY);
            background = updateBackground(imgGryscale, background, ALPHA);

            imshow("Deteccion de Movimiento", motionMask);
        }

        putText(imgO,fpsText , Point(5,395), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 1.5);

        //Ecualizacion del Histograma
        equalizeHist(imgGryscale, imghist);

        //CLAHE
        Ptr<CLAHE> clahe = createCLAHE();
        clahe->setClipLimit(4); 
        clahe->apply(imgGryscale, imgCLAHE);
        clahe->setTilesGridSize(Size(37,37));
        

        // Define el kernel para el filtro de realce
        Mat kernel = (Mat_<float>(3,3) << 
                    0, -1, 0,
                    -1,  5, -1,
                    0, -1, 0);
                    
        filter2D(imgO, imgRealce, -1, kernel);

        //visualizar imagen 
        imshow("Video Original", imgO);
        imshow("Video con CLAHE", imgCLAHE);
        imshow("Video con ecualización de Histograma,", imghist);
        imshow("Video con Realce", imgRealce);


        if (waitKey(30) == 'q') {
                break;
            }
        }

    return 0;
}