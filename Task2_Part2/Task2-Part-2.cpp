#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;

VideoCapture captura("atardecer.mp4");
int tamKernel = 3;
int tamVariaciones = 50;
int umbralInferior = 50;
int umbralSuperior = 150;
int tamanoFiltroSobel = 3;
Mat imgOri,imgCRUdio,imgSalt,imgPepper,imgMedian,imgBlur,imgGaussian,imgCanny,imgLaplacian,imgCannySF,imgLaplacianSF;

void noise(Mat im, int n,int rows,int cols) {
    for (int k = 0; k < n; k++) {
        int i = rand() % cols;
        int j = rand() % rows;
        if (rand() % 2 == 1) {
            im.at<Vec3b>(j, i) = Vec3b(255, 255, 255);
        } else{
            im.at<Vec3b>(j, i) = Vec3b(0, 0, 0);
        }
    }
}

void noise_Salt(Mat im, int n,int rows,int cols) {
    for (int k = 0; k < n; k++) {
        int i = rand() % cols;
        int j = rand() % rows;
        if (rand() % 2 == 1) {
            im.at<Vec3b>(j, i) = Vec3b(255, 255, 255);
        } 
    }
}

void noise_Pepper(Mat im, int n,int rows,int cols) {
    for (int k = 0; k < n; k++) {
        int i = rand() % cols;
        int j = rand() % rows;
        if (rand() % 2 == 1) {
            im.at<Vec3b>(j, i) = Vec3b(0, 0, 0);
        } 
    }
}

void onTrackbarSlide(int num, void *) {
    if (num < 1) num = 1; 
        tamVariaciones = num;
}

void onTrackbarSlideKernel(int num, void *) {
    if (num < 1);
    if (num % 2 == 0) {
        num++;
    } 
        tamKernel = num;
}

void onTrackbarSlideUB(int num, void *) {
    if (num < 1) num = 1; 
        umbralInferior = num;
}

void onTrackbarSlideUA(int num, void *) {
    if (num < 1) num = 1; 
        umbralSuperior = num;
}

void onTrackbarSlideBS(int num, void *) {
    if (num < 3) num = 3;
    if (num % 2 == 0) {
        num++;
    }
    if (num > 7) num = 7;
    tamanoFiltroSobel = num;
}


int main() {  
    //Crear ventana Rudio
    namedWindow("Comparación de Ruidos", WINDOW_AUTOSIZE);
    namedWindow("Comparación de Filtros",WINDOW_AUTOSIZE);
    namedWindow("Deteccion de Bordes Canny",WINDOW_AUTOSIZE);

    //Trackbar 
    createTrackbar("Tamaño", "Comparación de Ruidos",nullptr, 5000, onTrackbarSlide);
    setTrackbarPos("Tamaño", "Comparación de Ruidos", tamVariaciones);
    createTrackbar("Tamaño del Kernel", "Comparación de Filtros",nullptr, 10, onTrackbarSlideKernel);
    setTrackbarPos("Tamaño del Kernel", "Comparación de Filtros", tamKernel);
    createTrackbar("Umbral Inferior", "Deteccion de Bordes Canny", nullptr, 150, onTrackbarSlideUB);
    setTrackbarPos("Umbral Inferior", "Deteccion de Bordes Canny", umbralInferior);
    createTrackbar("Umbral Superior", "Deteccion de Bordes Canny",nullptr, 150, onTrackbarSlideUA);
    setTrackbarPos("Umbral Superior", "Deteccion de Bordes Canny", umbralSuperior);
    createTrackbar("Tamaño del Filtro Sobel", "Deteccion de Bordes Canny", nullptr, 7, onTrackbarSlideBS);
    setTrackbarPos("Tamaño del Filtro Sobel", "Deteccion de Bordes Canny", tamanoFiltroSobel);

    while (true) {
        captura.read(imgOri);
        
        resize(imgOri,imgOri,Size(600,300));

        //Ruido
        imgSalt= imgOri.clone();
        noise_Salt(imgSalt,tamVariaciones,imgSalt.rows,imgSalt.cols);

        imgPepper = imgOri.clone();
        noise_Pepper(imgPepper,tamVariaciones,imgPepper.rows,imgPepper.cols);

        imgCRUdio = imgOri.clone();
        noise(imgCRUdio,5000,imgCRUdio.rows,imgCRUdio.cols);

        //Filtros
        imgMedian = imgCRUdio.clone();
        medianBlur(imgMedian,imgMedian,tamKernel);

        imgBlur = imgCRUdio.clone();
        blur(imgBlur,imgBlur,Size(tamKernel,tamKernel));

        imgGaussian= imgCRUdio.clone();
        GaussianBlur(imgGaussian,imgGaussian, Size( tamKernel, tamKernel ),0);

        //Deteccion de bordes sin filtro
        imgCanny = imgCRUdio.clone();
        cvtColor(imgCanny, imgCanny, cv::COLOR_BGR2GRAY);
        GaussianBlur(imgCanny,imgCanny, Size( 3, 3 ),0);
        Canny(imgCanny, imgCanny,umbralInferior,umbralSuperior,tamanoFiltroSobel);

        imgLaplacian = imgCRUdio.clone();
        cvtColor(imgLaplacian, imgLaplacian, cv::COLOR_BGR2GRAY);
        GaussianBlur(imgLaplacian,imgLaplacian, Size( 3, 3 ),0);
        Laplacian(imgLaplacian, imgLaplacian, CV_8U, tamKernel, 1, 0, BORDER_DEFAULT);
        convertScaleAbs(imgLaplacian, imgLaplacian);

        //Deteccion de Borde con filtro de la mediana
        imgCannySF = imgMedian.clone();
        cvtColor(imgCannySF, imgCannySF, cv::COLOR_BGR2GRAY);
        GaussianBlur(imgCannySF,imgCannySF, Size( 3, 3 ),0);
        Canny(imgCannySF, imgCannySF,umbralInferior,umbralSuperior,tamanoFiltroSobel);

        imgLaplacianSF = imgMedian.clone();
        cvtColor(imgLaplacianSF, imgLaplacianSF, cv::COLOR_BGR2GRAY);
        GaussianBlur(imgLaplacianSF,imgLaplacianSF, Size( 3, 3 ),0);
        Laplacian(imgLaplacianSF, imgLaplacianSF, CV_8U, tamKernel, 1, 0, BORDER_DEFAULT);
        convertScaleAbs(imgLaplacianSF, imgLaplacianSF);

        putText(imgPepper, "Ruido de Pimienta", Point(25, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 1.5);
        putText(imgSalt, "Ruido de Sal", Point(25, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 1.5);
        putText(imgCRUdio, "Imagen Con Ruido", Point(25, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 1.5);
        putText(imgMedian, "Filtro de la Mediana", Point(25, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 1.5);
        putText(imgBlur, "Filtro de la Blur", Point(25, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 1.5);
        putText(imgGaussian, "Filtro de Gaussian", Point(25, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 1.5);
        putText(imgCanny,"Deteccion de Borde con Canny sin filtro",Point(25, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255,255,255), 1.5);
        putText(imgLaplacian, "Deteccion de Borde Laplacian sin filtro", Point(25, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 1.5);
        putText(imgCannySF,"Deteccion de Borde con Canny con filtro",Point(25, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255,255,255), 1.5);
        putText(imgLaplacianSF, "Deteccion de Borde de Laplacian con filtro", Point(25, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 1.5);
        putText(imgOri,"Imagen Original",Point(25, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 1.5);


        Mat noiseImage(imgOri.rows , imgOri.cols * 3, imgOri.type());
        imgOri.copyTo(noiseImage(Rect(0, 0, imgOri.cols, imgOri.rows)));
        imgSalt.copyTo(noiseImage(Rect(imgOri.cols,0,imgOri.cols,imgOri.rows)));
        imgPepper.copyTo(noiseImage(Rect(imgOri.cols*2,0,imgOri.cols,imgOri.rows)));

        Mat combinedImage(imgOri.rows * 2, imgOri.cols * 2, imgOri.type());
        imgCRUdio.copyTo(combinedImage(Rect(0, 0, imgOri.cols, imgOri.rows)));
        imgMedian.copyTo(combinedImage(Rect(imgOri.cols, 0, imgOri.cols, imgOri.rows)));
        imgBlur.copyTo(combinedImage(Rect(0, imgOri.rows, imgOri.cols, imgOri.rows)));
        imgGaussian.copyTo(combinedImage(Rect(imgOri.cols, imgOri.rows, imgOri.cols, imgOri.rows)));

        imshow("Comparación de Filtros", combinedImage);
        imshow("Comparación de Ruidos", noiseImage);
        imshow("Deteccion de Bordes Canny sin Filtro", imgCanny);
        imshow("Deteccion de Bordes Sobel sin Filtro", imgLaplacian);
        imshow("Deteccion de Bordes Canny con Filtro", imgCannySF);
        imshow("Deteccion de Bordes Sobel con Filtro", imgLaplacianSF);

        if (waitKey(30) == 's') {
            destroyAllWindows();
            break;
        }
    }
    captura.release();
    return 0;  
}
