import random
import shutil
import os
import six
import create_img.captcha_create_settings as settings
import time

try:
    from cStringIO import StringIO
except ImportError:
    from io import BytesIO as StringIO
from PIL import Image, ImageDraw, ImageFont

try:
    import json
except ImportError:
    from django.utils import simplejson as json

# Distance of the drawn text from the top of the captcha image
DISTANCE_FROM_TOP = 4


def getsize(font, text):
    if hasattr(font, 'getoffset'):
        return tuple([x + y for x, y in zip(font.getsize(text), font.getoffset(text))])
    else:
        return font.getsize(text)


def makeimg(size):
    if settings.CAPTCHA_BACKGROUND_COLOR == "transparent":
        image = Image.new('RGBA', size)
    else:
        image = Image.new('RGB', size, settings.CAPTCHA_BACKGROUND_COLOR)
    return image


def captcha_image(text, scale=1):
    if isinstance(settings.CAPTCHA_FONT_PATH, six.string_types):
        fontpath = settings.CAPTCHA_FONT_PATH
    elif isinstance(settings.CAPTCHA_FONT_PATH, (list, tuple)):
        fontpath = random.choice(settings.CAPTCHA_FONT_PATH)

    if fontpath.lower().strip().endswith('ttf'):
        font = ImageFont.truetype(fontpath, settings.CAPTCHA_FONT_SIZE * scale)
    else:
        font = ImageFont.load(fontpath)

    if settings.CAPTCHA_IMAGE_SIZE:
        size = settings.CAPTCHA_IMAGE_SIZE
    else:
        size = getsize(font, text)
        size = (size[0] * 2, int(size[1] * 1.4))

    image = makeimg(size)
    xpos = 2

    charlist = []
    for char in text:
        if char in settings.CAPTCHA_PUNCTUATION and len(charlist) >= 1:
            charlist[-1] += char
        else:
            charlist.append(char)
    for char in charlist:
        fgimage = Image.new('RGB', size, settings.CAPTCHA_FOREGROUND_COLOR)
        charimage = Image.new('L', getsize(font, ' %s ' % char), '#000000')
        chardraw = ImageDraw.Draw(charimage)
        chardraw.text((0, 0), ' %s ' % char, font=font, fill='#ffffff')
        if settings.CAPTCHA_LETTER_ROTATION:
            charimage = charimage.rotate(random.randrange(*settings.CAPTCHA_LETTER_ROTATION), expand=0,
                                         resample=Image.BICUBIC)
        charimage = charimage.crop(charimage.getbbox())
        maskimage = Image.new('L', size)

        maskimage.paste(charimage,
                        (xpos, DISTANCE_FROM_TOP, xpos + charimage.size[0], DISTANCE_FROM_TOP + charimage.size[1]))
        size = maskimage.size
        image = Image.composite(fgimage, image, maskimage)
        xpos = xpos + 2 + charimage.size[0]

    if settings.CAPTCHA_IMAGE_SIZE:
        # centering captcha on the image
        tmpimg = makeimg(size)
        tmpimg.paste(image, (int((size[0] - xpos) / 2), int((size[1] - charimage.size[1]) / 2 - DISTANCE_FROM_TOP)))
        image = tmpimg.crop((0, 0, size[0], size[1]))
    else:
        image = image.crop((0, 0, xpos + 1, size[1]))
    draw = ImageDraw.Draw(image)

    for f in settings.noise_functions():
        draw = f(draw, image)
    for f in settings.filter_functions():
        image = f(image)
    return image


# save_path = "D:\\machine_learn\\train1\\"
save_path = "D:\\machine_learn\\test111\\"
num = 2000
captcha_len = 4
chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

print("创建开始 路径%s 数量%d 长度%d 字符集%s" % (save_path, num, captcha_len, chars))
start = time.time()
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)
for xx in range(num):
    text = ""
    for x in range(captcha_len):
        text += random.choice(chars)
    image = captcha_image(text)
    image.save(save_path + text + ".png")
print("创建完毕 用时%.2fs" % (time.time() - start))
