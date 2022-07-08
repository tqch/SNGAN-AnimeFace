################################################################################
# example code of converting a sequence of static JPEG images to a gif animation
################################################################################

import os
import sys
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re

img_dir = sys.argv[1]
collection = os.path.basename(img_dir)

font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"

ext = re.compile(r"\.(jpg|jpeg|png|bmp|svg)$")
rank = re.compile(r"\d+")
img_list = sorted([
    f for f in os.listdir(img_dir)
    if ext.search(f)
], key=lambda x: int(rank.search(x).group(0)))

outs = []
res = 530
box = 66 * 2
txtlen = 8
fontsize = math.floor(box * 1.578 / txtlen)
for img in img_list:
    with Image.open(os.path.join(img_dir, img)) as im:
        im.putalpha(255)
        txt = Image.new("RGBA", size=im.size, color=(255, 255, 255, 0))
        d = ImageDraw.Draw(txt)
        fnt = ImageFont.truetype(font_path, size=fontsize)
        d.text((res-box, 0), f"epoch:{img.split('-')[-1][:-4]}", font=fnt, fill=(0, 255, 0, math.floor(255*0.7)))
        outs.append(Image.alpha_composite(im, txt))
nimgs = len(img_list)
duration = list((1000/24 * np.linspace(3, 1, nimgs)).round())
duration[-1] += 1000
outs[0].save(
    os.path.join(img_dir, collection+".gif"),
    save_all=True, append_images=outs[1:], optimize=True, duration=duration, loop=0)
