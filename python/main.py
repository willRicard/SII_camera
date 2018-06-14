#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Reconnaissance faciale - Projet SII 1 """

from time import sleep

import numpy as np
import cv2 as cv

WIN_MAIN = "Face Recognition!"


def face_detect(image, classifier):
    """ Retourne le premier visage trouve dans l'image """
    # La détection a lieu dans une image en noir et blanc
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.equalizeHist(gray, gray)
    faces = classifier.detectMultiScale(gray, 1.3, 2, cv.CASCADE_SCALE_IMAGE,
                                        (40, 48), gray.shape)
    if not faces.size:
        return None
    face = faces[0]

    # On trace un rectangle autour de chaque visage
    cv.rectangle(image, (face[0], face[1]),
                 (face[0] + face[2], face[1] + face[3]), (255, 0, 0))
    return face


def commande_rotation(rotation):
    """ Envoi de la commande de rotation à l'Arduino """
    print(chr(rotation['x']) + chr(rotation['y']))


def main():
    """ Fonction principale """
    # On charge la description d'un visage
    cascade = cv.CascadeClassifier()
    cascade.load("../data/cascade.xml")

    cv.namedWindow(WIN_MAIN, cv.WINDOW_AUTOSIZE)
    cv.moveWindow(WIN_MAIN, 400, 0)

    capture = cv.VideoCapture(1)
    if not capture.isOpened():
        raise Exception("Impossible d'ouvrir la caméra !")

    frame = np.zeros((100, 100))

    rotation = {'x': 70.0, 'y': 45.0}
    commande_rotation(rotation)

    sleep(2)

    while True:
        read_ok, frame = capture.read()

        if not read_ok:
            raise Exception("Image vide.")

        # On retourne l'image
        cv.flip(frame, 1, frame)

        face = face_detect(frame, cascade)

        # Coin supérieur gauche du rectangle en dehors duquel on tourne pour
        # s'aligner avec le visage.
        center = np.array([frame.cols / 2 - 128, frame.rows / 2 - 128])

        cv.rectangle(frame, center, center + np.array([256, 256]), (0, 0, 255))

        distance = center - face

        # Si le visage est trop loin du centre, on tourne la caméra et on attent
        # un peu
        if face[0] != 0 and face[1] != 0 and (np.abs(distance) > 64).any():
            # Avec la caméra utilisée, la largeur de l'image correspond à une
            # rotation de 40°. La vitesse de rotation du servomoteur SG90 est estimée
            # à 90°/s.
            rotation['x'] -= 0.5 * distance[0] / float(frame.cols) * 40.0
            rotation['y'] -= 0.5 * distance[1] / float(frame.rows) * 40.0

            if rotation['x'] < 0:
                rotation.x = 0
            elif rotation['x'] > 90:
                rotation.y = 90

            if rotation['y'] < 0:
                rotation['y'] = 0
            elif rotation['y'] > 90:
                rotation['y'] = 90

            commande_rotation(rotation)

            sleep(0.2 * max(rotation['x'], rotation['y']) / 90.0)

        cv.imshow(WIN_MAIN, frame)

        # Gestion des évènements
        key = cv.waitKey(100)
        if key == 0x1b:  # Échap
            cv.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
