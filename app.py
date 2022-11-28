# from tkinter import image_names
import streamlit as st
from PIL import Image
from io import BytesIO, BufferedReader
import cv2
import numpy as np


st.title("CARTOON MAKER")
st.header("Upload your fav juicy pics cvt into amazing cartoon char")
st.info("File type should be : png jpg jpeg")


def take_file():
    return st.file_uploader("", type=["png", "jpg", "jpeg"])


def car(img):
    IMG = img
    Image_data = Image.open(IMG)
    st.info("Uploaded Image")
    st.image(Image_data)
    image = Image.open(img)
    # st.image(image, caption='Input', use_column_width=True)
    img_array = np.array(image)
    # cv2.imwrite('out.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    img = img_array
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_grey, 3)
    # edge masking
    edges = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 3)
    img_bb = cv2.bilateralFilter(img_blur, 15, 75, 75)
    kernel = np.ones((1, 1), np.uint8)
    img_erode = cv2.erode(img_bb, kernel, iterations=3)
    img_dilate = cv2.dilate(img_erode, kernel, iterations=3)
    img_style = cv2.stylization(img, sigma_s=150, sigma_r=0.25)
    st.info("Cartoon Preview")
    # st.image(img_style)
    return img_style


image_file = take_file()
# print(image_file)
if image_file is not None:
    st.success("Uploaded!!!")
    cartoon = car(image_file)
    cartoon = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
    st.image(cartoon)
    # result = Image.fromarray(cartoon.astype('uint8'), 'RGB')
    # img = Image.open(result)
    # st.download_button("Save", cartoon)
    im_rgb = cartoon[:, :, [2, 1, 0]]  # numpy.ndarray
    ret, img_enco = cv2.imencode(".png", im_rgb)  # numpy.ndarray
    srt_enco = img_enco.tostring()  # bytes
    img_BytesIO = BytesIO(srt_enco)  # _io.BytesIO
    img_BufferedReader = BufferedReader(img_BytesIO)  # _io.BufferedReader
    st.download_button(
        label="Downlaod",
        data=img_BufferedReader,
        file_name="Cartoon.png",
        mime="image/png"
    )
else:
    st.error("Try Here!!!")
