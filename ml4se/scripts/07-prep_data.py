# -*- coding: utf-8 -*-
#
# 手書き文字サンプルの抽出
#
# 2015/06/08 ver1.0
#

import re
from subprocess import Popen, PIPE

#------------#
# Parameters #
#------------#
Num = 600           # 抽出する文字数
Chars = '[036]'     # 抽出する数字（任意の個数の数字を指定可能）


labels = open('train-labels.txt', 'r')
images = open('train-images.txt', 'r')
labels_out = open('sample-labels.txt', 'w')
images_out = open('sample-images.txt', 'w')
chars = re.compile(Chars)

while True:
    label = labels.readline()
    image = images.readline()
    if (not image) or (not label):
        break
    if not chars.search(label):
        continue

    line = ''
    for c in image.split(" "):
        if int(c) > 127:
            line += '1,'
        else:
            line += '0,'
    line = line[:-1]
    labels_out.write(label)
    images_out.write(line + '\n')
    Num -= 1
    if Num == 0:
        break

labels.close()
images.close()
labels_out.close()
images_out.close()

images = open('sample-images.txt', 'r')
samples = open('samples.txt', 'w')
c = 0

while True:
    line = images.readline()
    if not line:
        break
    x = 0
    for s in line.split(','):
        if int(s) == 1:
            samples.write('#')
        else:
            samples.write(' ')
        x += 1
        if x % 28 == 0:
            samples.write('\n')
    c += 1
    if c == 10:
        break

images.close()
samples.close()

