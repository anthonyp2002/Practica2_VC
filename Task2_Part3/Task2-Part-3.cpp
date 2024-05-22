#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "iostream"

using namespace cv;
using namespace std;

String vectorImg[] = {"ImgM_1.png","ImgM_2.jpeg","ImgM_3.jpeg"};
int limite = (sizeof(vectorImg)/sizeof(vectorImg[0]));
int tamkernel = 5;
Mat element; 
Mat imgE,imgD,imgTH,imgBH,imgTBH;
Mat img;
bool running;

void morphological_operations(Mat img){
    resize(img,img,Size(250,350));
    cvtColor(img,img,COLOR_BGR2GRAY);
    element = getStructuringElement(MORPH_RECT, Size(tamkernel, tamkernel));
    //Erosion
    erode(img,imgE,element);
    //Dilation
    dilate(img,imgD,element);
    //Top Hat
    morphologyEx(img,imgTH ,MORPH_TOPHAT,element);
    //Black Hat
    morphologyEx(img,imgBH ,MORPH_BLACKHAT,element);
    // Imagen Original + (Top Hat – Black Hat)
    imgTBH = img + (imgTH - imgBH);

    Mat operationsImg(img.rows , img.cols * 6, img.type());
    img.copyTo(operationsImg(Rect(0,0,img.cols,img.rows)));
    imgE.copyTo(operationsImg(Rect(img.cols,0,img.cols,img.rows)));
    imgD.copyTo(operationsImg(Rect(img.cols*2,0,img.cols,img.rows)));
    imgTH.copyTo(operationsImg(Rect(img.cols*3,0,img.cols,img.rows)));
    imgBH.copyTo(operationsImg(Rect(img.cols*4,0,img.cols,img.rows)));
    imgTBH.copyTo(operationsImg(Rect(img.cols*5,0,img.cols,img.rows)));

    imshow("Imagen", operationsImg);
    int key = waitKey(1);
    if (key != -1) {
        running = false;
    }
}


void onTrackbarSlideKernel(int num, void *) {
    if (num < 1){
         num = 1;
    };
    if (num % 2 == 0) {
        num++;
    }
    tamkernel = num;
}



int main() {  
    namedWindow("Imagen", cv::WINDOW_AUTOSIZE);
    createTrackbar("Tamaño del Kernel","Imagen",nullptr,100,onTrackbarSlideKernel);
    for(int i=0;i<limite;i++){
        img = imread(vectorImg[i]);
        setTrackbarPos("Tamaño del Kernel","Imagen",tamkernel);
        running = true;
        while (running) {
            morphological_operations(img);
        }
        waitKey(0);
    }
    destroyWindow("Imagen");
    return 0;  
}
