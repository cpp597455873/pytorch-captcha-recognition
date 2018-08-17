# -*- coding: UTF-8 -*-
import numpy as np
import captcha_setting


def encode(text):
    vector = np.zeros(captcha_setting.ALL_CHAR_SET_LEN * captcha_setting.MAX_CAPTCHA, dtype=float)
    for i, c in enumerate(text):
        for index, value in enumerate(captcha_setting.ALL_CHAR_SET):
            if (value == c):
                idx = i * captcha_setting.ALL_CHAR_SET_LEN + index
                vector[idx] = 1.0
    return vector


def decode(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_idx = c % captcha_setting.ALL_CHAR_SET_LEN
        text.append(captcha_setting.ALL_CHAR_SET[char_idx])
    return "".join(text)


if __name__ == '__main__':
    e = encode("XKNK")
    print(e)
    print(decode(e))
