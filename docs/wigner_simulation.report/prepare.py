import Image
import os, sys

files = os.listdir("./")
images = []
for file in files:
    f, e = os.path.splitext(file)
    if not e.endswith("bmp"):
        continue
    images.append(Image.open(file))

resized = []
for i, image in enumerate(images):
    resized.append(image.resize((32, 32)))

res = Image.new("RGBA", (32 * 6, 32 * 4), (0, 0, 0, 0))
for i, image in enumerate(resized):
    res.paste(image, (32 * (i % 6), 32 * (i / 6)))
res.save("res.png")
