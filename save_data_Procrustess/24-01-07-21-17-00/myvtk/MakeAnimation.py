import glob
from PIL import Image, ImageDraw, ImageFont
import cv2

# files = glob.glob("C:\\Users\\Chen\\Desktop\\New Folder\\*.jpg")
# font = ImageFont.truetype("Gidole-Regular.ttf", size=25)

imgs = []
j = 0
# txt = case_name[j]
for i in range(len(files)):
    im = Image.open(files[i])
    draw = ImageDraw.Draw(im)
    # draw.text((25, 300), case_name[j], 'gray', font = font)
    #print (j)
    if i>0 and (i % 18 == 0):
        txt = case_name[j]
        j = j+1
    imgs.append(im)

imgs[0].save('C:\\Users\\Chen\\Desktop\\New Folder\\pillow_imagedraw_txt.gif',
               save_all=True, append_images=imgs[1:], optimize=False, duration=200, loop=0)


def makeAnime(files=[], savePath=''):
    imgs = []
    j = 0
    # txt = case_name[j]
    for i in range(len(files)):
        im = Image.open(files[i])
        draw = ImageDraw.Draw(im)
        # draw.text((25, 300), case_name[j], 'gray', font = font)
        # print (j)
        # if i>0 and (i % 18 == 0):
            # txt = case_name[j]
            # j = j+1
        imgs.append(im)

    imgs[0].save(savePath,
                save_all=True, append_images=imgs[1:], optimize=False, duration=200, loop=0)