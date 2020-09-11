# -*- coding: utf-8 -*-
# @Author: AnnaZhang
# @Date:   2020-08-28 10:29:50
# @Last Modified by:   AnnaZhang
# @Last Modified time: 2020-09-11 10:39:03
import numpy as np
import pandas as pd
import math
from math import tan, radians, pi
import random
from scipy.interpolate import interp1d
import matplotlib as plt
import matplotlib.pyplot as pyplot
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter
import cv2
import imutils
from multiprocessing import Pool
import os
import h5py

SCALE = 100
WIDTH = 2 * SCALE + 1
HEIGHT = 2 * SCALE + 1
SQRT2 = math.sqrt(2)
LOWEST_LIMIT = SCALE * 0.3
EPS = 0.05
PI = np.pi


def judge_line(points_df, n_pieces=8):

    if_valid = 1
    # unvalid if the line is too bumping
    for i in range(3):

        x1 = np.array(points_df["x"])[i + 0]
        y1 = np.array(points_df["y"])[i + 0]
        x2 = np.array(points_df["x"])[i + 1]
        y2 = np.array(points_df["y"])[i + 1]
        theta1 = np.array(points_df["theta"])[i + 0]
        theta2 = np.array(points_df["theta"])[i + 1]
        if theta1 >= theta2:
            if_valid = 0
        if ((x1 - x2) ** 2 + (y1 - y2) ** 2) < LOWEST_LIMIT:
            if_valid = 0
        if i == 0:
            if np.arctan((y2 - y1) / (x2 - x1)) < PI / 2 - EPS or np.arctan((y2 - y1) / (x2 - x1)) > PI / 2 + EPS:
                if_valid = 0
                # print(f"GG: {np.arctan((y2 - y1) / (x2 - x1))}")
            # else:
            # print("---------------------------------------------------")
        if i == 2:
            if (np.arctan((y2 - y1) / (x2 - x1)) < (-2 * PI / n_pieces - EPS)) or (
                np.arctan((y2 - y1) / (x2 - x1)) > (-2 * PI / n_pieces + EPS)
            ):

                if_valid = 0
            # elif if_valid:
            # print("---------------------------------------------------")
    for i in range(2):
        x1 = np.array(points_df["x"])[i + 0]
        y1 = np.array(points_df["y"])[i + 0]
        x2 = np.array(points_df["x"])[i + 1]
        y2 = np.array(points_df["y"])[i + 1]
        x3 = np.array(points_df["x"])[i + 2]
        y3 = np.array(points_df["y"])[i + 2]
        if (x2 - x1) * (x3 - x2) + (y2 - y1) * (y3 - y2) < 0 - EPS:
            if_valid = 0
        # if if_valid == 1:
        # print(points_df)
    return if_valid


def judge_region(rho, theta, n_pieces=8):
    if_valid = 1
    # unvalid if arctan not in 2*pi/n

    """
    if (y / x) > tan(2 * pi / n_pieces):
        if_valid = 0
    # unvalid if too center or too far
    if (x ** 2 + y ** 2) > (SCALE ** 2):
        if_valid = 0
    if (x ** 2 + y ** 2) < ((SCALE / 2) ** 2):
        if_valid = 0
    """

    return if_valid


def gen_random_points(n_pieces=8):
    n_points = 0
    org_points_df = pd.DataFrame(columns=["rho", "theta", "x", "y"])

    # first point
    rho = random.uniform(SCALE * 0.25, SCALE * 0.65)
    theta = 0
    x = rho * math.cos(radians(theta))
    y = rho * math.sin(radians(theta))
    org_points_df = org_points_df.append(pd.DataFrame({"rho": [rho], "theta": [theta], "x": [x - 0.1], "y": [y]}))
    # org_points_df = org_points_df.append(pd.DataFrame({"rho": [rho], "theta": [theta], "x": [x], "y": [y]}))

    ## next two points
    for _ in range(2):
        rho = random.uniform(SCALE * 0.3, SCALE * 0.7)
        theta = random.uniform(0, 360 / n_pieces)
        x = rho * math.cos(radians(theta))
        y = rho * math.sin(radians(theta))
        temp_df = pd.DataFrame({"rho": [rho], "theta": [theta], "x": [x + 1], "y": [y]})
        org_points_df = org_points_df.append(temp_df)

    # last point
    rho = random.uniform(SCALE * 0.25, SCALE * 0.65)
    theta = 360 / n_pieces
    x = rho * math.cos(radians(theta))
    y = rho * math.sin(radians(theta))
    org_points_df = org_points_df.append(pd.DataFrame({"rho": [rho], "theta": [theta], "x": [x], "y": [y]}))
    # org_points_df = org_points_df.append(
    # pd.DataFrame({"rho": [rho], "theta": [theta], "x": [x + 0.01], "y": [y - 0.1]})
    # )
    return org_points_df


def get_length(points_df, n=4):

    length = list()
    length_total = 0
    for i in range(n - 1):
        x1 = np.array(points_df["x"])[i + 0]
        y1 = np.array(points_df["y"])[i + 0]
        x2 = np.array(points_df["x"])[i + 1]
        y2 = np.array(points_df["y"])[i + 1]
        current_len = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        length.append(current_len)
        length_total += current_len
    return length, length_total


def interp(points_df):
    length, length_total = get_length(points_df)
    t1 = length[0] / length_total * WIDTH
    t2 = (length[0] + length[1]) / length_total * WIDTH
    t = np.array([0, int(t1), int(t2), WIDTH])
    x = np.array(points_df["x"]).astype(float)
    y = np.array(points_df["y"]).astype(float)
    fx = interp1d(t, x, kind="cubic")
    fy = interp1d(t, y, kind="cubic")

    tt = np.linspace(0, WIDTH, WIDTH + 1)
    xx = fx(tt).astype(int)
    # f = interp1d(x, y, kind="cubic", fill_value="extrapolate")

    yy = fy(tt).astype(int)
    line_df = pd.DataFrame({"x": xx, "y": yy})
    return line_df


def expand(image):

    rotated_1 = imutils.rotate(image, 90)
    rotated_2 = imutils.rotate(image, 180)
    rotated_3 = imutils.rotate(image, 270)
    flip_1 = cv2.flip(image, 0, dst=None)
    flip_2 = imutils.rotate(flip_1, 90)
    flip_3 = imutils.rotate(flip_1, 180)
    flip_4 = imutils.rotate(flip_1, 270)

    image = cv2.add(image, rotated_1)
    image = cv2.add(image, rotated_2)
    image = cv2.add(image, rotated_3)
    image = cv2.add(image, flip_1)
    image = cv2.add(image, flip_2)
    image = cv2.add(image, flip_3)
    image = cv2.add(image, flip_4)
    return image


def filling(line_df):
    im = np.zeros([WIDTH, WIDTH], dtype=np.int)
    y_prev = np.array(line_df["y"])[0] + SCALE
    for row in line_df.itertuples():
        x = int(getattr(row, "x") + SCALE)
        y = getattr(row, "y") + SCALE
        if y >= 0 and y < WIDTH:
            ymin = int(min(y_prev, y))
            ymax = int(max(y_prev, y)) + 1
            if y_prev >= 0 and y_prev < WIDTH and x != SCALE and ymax - ymin < 20:

                im[x, ymin:ymax] = 255
                y_prev = y
            else:

                im[x, y] = 255
                y_prev = y

    return im


def generate_image(number):
    img_path = f"dos-momentum-4fold/png/{number:04}.png"
    h5_path = f"dos-momentum-4fold/h5/{number:04}.h5"
    # begin generation
    while True:
        fixed_points = gen_random_points(n_pieces=8)
        if judge_line(fixed_points, n_pieces=8):

            line_df = interp(fixed_points)

            im = filling(line_df).astype(float)
            im = expand(im)

            kernel = np.ones((10, 10), np.float32) / 25
            im_blur = cv2.GaussianBlur(im, (9, 9), 2)
            im_blur = im_blur * 3
            max_im = np.ones([WIDTH, WIDTH], dtype=np.int) * 255
            im_blur = np.where(im_blur > max_im, max_im, im_blur)

            # im_blur = im
            # im_blur = cv2.filter2D(im, -1, kernel)
            # im_blur = gaussian_filter(im, sigma=1)
            im_blur = Image.fromarray(im_blur)

            with h5py.File(h5_path, "w") as h:
                h.create_dataset('/isoE', data=im_blur, dtype='float32', compression='gzip')

            im_blur = im_blur.convert("L")
            im_blur.save(img_path, "png")
            return img_path


if __name__ == "__main__":
    with Pool(250) as pool:
        for f in pool.imap_unordered(generate_image, list(range(10000))):
            print(f'{f} generated')

