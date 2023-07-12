from PIL import Image

im = Image.open('man_cafe.jpg')

im_res1 = Image.open('./runs/detect/exp/man_cafe.jpg')
im_res1.show()