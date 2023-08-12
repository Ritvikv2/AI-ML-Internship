from PIL import Image

# Open the original image
im = Image.open('man_cafe.jpg')

# Open the resulting image from the detection process
im_res1 = Image.open('./runs/detect/exp/man_cafe.jpg')

# Display the resulting image
im_res1.show()