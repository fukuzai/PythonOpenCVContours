# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:27:44 2019

@author: fukuzai
"""
#参考
#https://postd.cc/image-processing-101/

#環境
#openCV : 4.0.0
#python : 3.6.5

import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

#画像読み込み
coins = cv2.imread('coins-orig.jpg')

#入力画像を描画
plt.imshow(coins)

#グレー画像に変換
coins_gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)

#ガウシアンフィルタ
coins_preprocessed = cv2.GaussianBlur(coins_gray, (5, 5), 0)
 
#二値化処理
_, coins_binary = cv2.threshold(coins_preprocessed, 130, 255, cv2.THRESH_BINARY)
 
#白黒反転処理
#openCVでは白側が処理対象となるため
coins_binary = cv2.bitwise_not(coins_binary)

#二値画像を描画
plt.imshow(coins_binary,plt.cm.gray)

# 輪郭を見つける
coins_contours, _ = cv2.findContours(coins_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
# matをコピー
coins_and_contours = np.copy(coins)
 
# 輪郭の選択
min_coin_area = 60
large_contours = [cnt for cnt in coins_contours if cv2.contourArea(cnt) > min_coin_area]
 
# 輪郭を描画した配列を作成
cv2.drawContours(coins_and_contours, large_contours, -1, (255,0,0))
 
# 輪郭の個数を出力
print('コインの数: %d' % len(large_contours))

#画像と輪郭を描画
plt.imshow(coins_and_contours)

#外接矩形
#matをコピー
bounding_img = np.copy(coins)
 
# for each contour find bounding box and draw rectangle
for contour in large_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(bounding_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

#画像と外接矩形を描画   
plt.imshow(bounding_img)