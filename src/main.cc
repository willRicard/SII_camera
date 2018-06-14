/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2018 Guillaume Ricard <guillaume.ricard@mail.com>, Brieuc
 * Julien
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE
 *
 */

#define _USE_MATH_DEFINES
#include <unistd.h>

#include <cmath>
#include <iostream>
#include <string>

#include <opencv/cv.hpp>
#include <opencv/highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

// Retourne la position du premier visage reconnu dans l'image
static cv::Point FaceDetect(cv::Mat &image, cv::CascadeClassifier &cascade) {
  std::vector<cv::Rect> faces;
  cv::Mat gray;
  cv::cvtColor(image, gray, CV_BGR2GRAY);
  cv::equalizeHist(gray, gray);

  // Détection
  cascade.detectMultiScale(gray, faces, 1.3, 2, CV_HAAR_SCALE_IMAGE,
                           cv::Size(40, 48), cv::Size(image.cols, image.rows));

  // Ajout de rectangles
  if (faces.size() > 0) {
    cv::Rect face = faces[0];
    cv::rectangle(image, cv::Point(face.x, face.y),
                  cv::Point(face.x + face.width, face.y + face.height),
                  CV_RGB(255, 0, 0));
    return cv::Point(face.x, face.y);
  } else {
    return cv::Point(0, 0);
  }
}

static int ErrorCallback(int status, const char *func, const char *err,
                         const char *file, int line, void *) {
  std::cerr << "Erreur OpenCV(" << status << ") " << file << ":" << line << ": "
            << func << err << file;
  return 0;
}

int main() {
  cv::redirectError(ErrorCallback);

  // Charge la description d'un visage
  cv::CascadeClassifier cascade;
  if (!cascade.load("cascade.xml")) {
    std::cerr << "Impossible de charger la description d'un visage depuis le "
                 "fichier `cascade.xml`"
              << std::endl;
    return 1;
  }

  cv::namedWindow("SII - Camera", cv::WINDOW_AUTOSIZE);
  cv::moveWindow("SII - Camera", 400, 0);

  cv::VideoCapture capture(1);
  cv::Mat frame;

  if (!capture.isOpened()) {
    std::cerr << "Impossible d'ouvrir la caméra !" << std::endl;
    return 1;
  }

  // On initialise la rotation de la caméra.
  cv::Point2f rotation;
  rotation.x = 70.0;
  rotation.y = 45.0;

  printf("%c%c\n", (char)rotation.x, (char)rotation.y);

  sleep(2);

  for (;;) {
    capture >> frame;

    if (frame.empty()) {
      return 1;
    }

    cv::flip(frame, frame, 1);

    // Coin supérieur gauche du rectangle en dehors duquel on tourne pour
    // s'aligner avec le visage.
    cv::Point center;
    center.x = frame.cols / 2 - 128;
    center.y = frame.rows / 2 - 128;

    cv::rectangle(frame, center, center + cv::Point(256, 256),
                  CV_RGB(0, 0, 255));

    cv::Point face = FaceDetect(frame, cascade);

    cv::Point distance = center - face;

    // Si le visage est trop loin du centre, on tourne la caméra et on attend
    // un peu
    if (face.x != 0 && face.y != 0 &&
        (abs(distance.x) > 64 || abs(distance.y) > 64)) {

      // Avec la caméra utilisée, la largeur de l'image correspond à une
      // rotation de 40°. La vitesse de rotation du servomoteur SG90 est estimée
      // à 90°/s.
      rotation.x -= 0.5 * (float)distance.x / (float)frame.cols * 40.0;
      rotation.y -= 0.5 * (float)distance.y / (float)frame.rows * 40.0;

      std::cerr << rotation.x << " " << rotation.y << std::endl;

      if (rotation.x < 0) {
        rotation.x = 0;
      } else if (rotation.x > 90) {
        rotation.x = 90;
      }

      if (rotation.y < 0) {
        rotation.y = 0;
      } else if (rotation.y > 90) {
        rotation.y = 90;
      }

      printf("%c%c\n", (char)rotation.x, (char)rotation.y);

      useconds_t time = 2 * fmax(rotation.x, rotation.y) / 90.0 * 100000;
      usleep(time);
    }

    cv::imshow("SII - Camera", frame);

    // Gestion des évènements
    int key = cv::waitKey(1);
    switch (key) {
    case 0x1b: // Échap
      cv::destroyAllWindows();
      return 0;
    }
  }

  return 0;
}
