//
//  main.cpp
//  OpenCV
//
//  Created by Christian Roman Mendoza on 21/12/11.
//  Copyright (c) 2011 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>

CvHaarClassifierCascade *cascade;
CvMemStorage *storage;
CvCapture* capture;

int main (int argc, const char * argv[])
{
    float scale = 1.15;
    int elements = 7;
    
    // Array
    IplImage* images[10];
    
    images[0] = cvLoadImage("megusta.png");
    images[1] = cvLoadImage("foreveralone.png");
    images[2] = cvLoadImage("pokerface.png");
    images[3] = cvLoadImage("trollface.png");
    images[4] = cvLoadImage("fuckyeah.png");
    images[5] = cvLoadImage("gordo.png");
    images[6] = cvLoadImage("yuno.png");
    
    srand(time(NULL));
    int j = (int)rand() % elements;
    
    clock_t actual = clock();
    clock_t anterior = clock();
    
    capture = cvCreateCameraCapture(0);
    IplImage* originalImg;
    
    char *filename = "haarcascade_frontalface_alt.xml";
    
    cascade = ( CvHaarClassifierCascade* )cvLoad( filename, 0, 0, 0 );
    storage = cvCreateMemStorage( 0 );
    
    cvNamedWindow( "image", CV_WINDOW_AUTOSIZE );
    
    if( capture )
    {
        for(;;)
        {
            actual = clock();
            if(actual - anterior > (5*CLOCKS_PER_SEC)){
                j = (int)rand() % elements;
                anterior = actual;
            }
            
            cvGrabFrame(capture);
            originalImg = cvRetrieveFrame(capture);
            
            if(!originalImg) break;
            
            CvSeq *faces = cvHaarDetectObjects(originalImg,cascade,storage,1.1,3,0,cvSize( 40, 40 ) );
            
            for( int i = 0 ; i < ( faces ? faces->total : 0 ) ; i++ ) {
                CvRect *r = ( CvRect* )cvGetSeqElem( faces, i );
                
                IplImage *tmp = cvCreateImage(cvSize(r->width*scale, r->height*scale), images[j]->depth,images[j]->nChannels);
                cvResize(images[j], tmp, CV_INTER_CUBIC);
                cvSetImageROI(originalImg,cvRect(r->x-r->width/7,r->y-r->height/15,tmp->width,tmp->height));
                cvAdd(originalImg, tmp, originalImg);
                cvReleaseImage(&tmp);
                cvResetImageROI(originalImg);
                
            }

            cvShowImage("image", originalImg);
            
            char c = cvWaitKey(10);
            if( c == 27 ) break;
            
        }
        cvReleaseCapture(&capture);
    }
    cvDestroyWindow("image");
    
    return 0;
}