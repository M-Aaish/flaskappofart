
import numpy as np
import cv2
import random
import time
import rotate_brush as rb
import gradient
from thready import amap
import threading
import os
import json

canvaslock = threading.Lock()

# Lock for the canvas for thread safety
canvaslock.acquire()
canvaslock.release()

def load(im):
    # print('loading', filename, '...')
    global imname, flower, canvas, hist
    global rescale, xs_small, ys_small, smallerflower

    # imname = filename.split('.')[0]

    # original image
    flower = im

    xshape = flower.shape[1]
    yshape = flower.shape[0]

    rescale = xshape / 640
    if rescale < 1:
        rescale = 1

    xs_small = int(xshape / rescale)
    ys_small = int(yshape / rescale)

    smallerflower = cv2.resize(flower, dsize=(xs_small, ys_small)).astype('float32') / 255

    flower = flower.astype('float32') / 255

    canvas = flower.copy()
    canvas[:, :] = 0.8  # Initialize canvas with some color

    hist = []
    # print(filename, 'loaded.')

# load()

def rn():
    return random.random()

def savehist(filename='hist.json'):
    with open(filename, 'w') as f:
        json.dump(hist, f)

def record(sth):
    hist.append(sth)

def positive_sharpen(i, overblur=False, coeff=8.):
    blurred = cv2.blur(i, (5, 5))
    sharpened = i + (i - blurred) * coeff
    if overblur:
        return cv2.blur(np.maximum(sharpened, i), (11, 11))
    return cv2.blur(np.maximum(sharpened, i), (3, 3))

def diff(i1, i2, overblur=False):
    d = (i1 - i2)
    d = d * d

    d = positive_sharpen(np.sum(d, -1), overblur=overblur)
    return d

def get_random_color():
    return np.array([rn(), rn(), rn()]).astype('float32')

def limit(x, minimum, maximum):
    return min(max(x, minimum), maximum)

# History and replay section
hist = []

def repaint(upscale=1., batchsize=16):
    starttime = time.time()

    newcanvas = np.array(canvas).astype('uint8')

    if upscale != 1.:
        newcanvas = cv2.resize(newcanvas, dsize=(int(newcanvas.shape[1] * upscale), int(newcanvas.shape[0] * upscale)))

    newcanvas[:, :, :] = int(0.8 * 255)

    def paintone(histitem):
        x, y, radius, srad, angle, cb, cg, cr, brushname = histitem

        cb, cg, cr = int(cb * 255), int(cg * 255), int(cr * 255)

        b, key = rb.get_brush(brushname)

        radius, srad = int(radius), int(srad)

        if angle == -1.:
            angle = rn() * 360

        rb.compose(newcanvas, b, x=x, y=y, rad=radius, srad=srad, angle=angle, color=[cb, cg, cr], useoil=True, lock=canvaslock)

    batch = []
    k = 0
    while k < len(hist):
        while len(batch) < batchsize and k < len(hist):
            batch.append(hist[k])
            k += 1
        amap(paintone, batch)
        batch = []

    print(time.time() - starttime, 's elapsed')

def paint_one(x, y, brushname='random', angle=-1., minrad=10, maxrad=60):
    oradius = rn() * rn() * maxrad + minrad
    fatness = 1 / (1 + rn() * rn() * 6)

    brush, key = rb.get_brush(brushname)

    def intrad(orad):
        radius = int(orad)
        srad = int(orad * fatness + 1)
        return radius, srad

    radius, srad = intrad(oradius)

    if angle == -1.:
        angle = rn() * 360

    c = flower[int(y), int(x), :]

    delta = 1e-4

    def get_roi(newx, newy, newrad):
        radius, srad = intrad(newrad)

        xshape = flower.shape[1]
        yshape = flower.shape[0]

        yp = int(min(newy + radius, yshape - 1))
        ym = int(max(0, newy - radius))
        xp = int(min(newx + radius, xshape - 1))
        xm = int(max(0, newx - radius))

        if yp <= ym or xp <= xm:
            raise NameError('zero roi')

        ref = flower[ym:yp, xm:xp]
        bef = canvas[ym:yp, xm:xp]
        aftr = np.array(bef)

        return ref, bef, aftr

    def paint_aftr_w(color, angle, nx, ny, nr):
        ref, bef, aftr = get_roi(nx, ny, nr)
        radius, srad = intrad(nr)

        rb.compose(aftr, brush, x=radius, y=radius, rad=radius, srad=srad, angle=angle, color=color, usefloat=True, useoil=False)
        err_aftr = np.mean(diff(aftr, ref))
        return err_aftr

    def paint_final_w(color, angle, nr):
        radius, srad = intrad(nr)

        rb.compose(canvas, brush, x=x, y=y, rad=radius, srad=srad, angle=angle, color=color, usefloat=True, useoil=True, lock=canvaslock)

        rec = [x, y, radius, srad, angle, color[0], color[1], color[2], brushname]
        rec = [float(r) if type(r) == np.float64 or type(r) == np.float32 else r for r in rec]
        record(rec)

    def calc_gradient(err):
        b, g, r = c[0], c[1], c[2]
        cc = b, g, r

        err_aftr = paint_aftr_w((b + delta, g, r), angle, x, y, oradius)
        gb = err_aftr - err

        err_aftr = paint_aftr_w((b, g + delta, r), angle, x, y, oradius)
        gg = err_aftr - err

        err_aftr = paint_aftr_w((b, g, r + delta), angle, x, y, oradius)
        gr = err_aftr - err

        err_aftr = paint_aftr_w(cc, (angle + 5.) % 360, x, y, oradius)
        ga = err_aftr - err

        err_aftr = paint_aftr_w(cc, angle, x + 2, y, oradius)
        gx = err_aftr - err

        err_aftr = paint_aftr_w(cc, angle, x, y + 2, oradius)
        gy = err_aftr - err

        err_aftr = paint_aftr_w(cc, angle, x, y, oradius + 3)
        gradius = err_aftr - err

        return np.array([gb, gg, gr]) / delta, ga / 5, gx / 2, gy / 2, gradius / 3, err

    return paint_final_w(c, angle, oradius)

# Automatically run for 10 epochs
def run_for_epochs(epochs=10):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} starting...")
        for _ in range(100):  # Paint 100 random strokes per epoch
            x = random.randint(0, flower.shape[1] - 1)
            y = random.randint(0, flower.shape[0] - 1)
            brushname = 'random'
            angle = random.uniform(0, 360)
            paint_one(x, y, brushname=brushname, angle=angle)
        repaint(batchsize=16)

        # Display the canvas image after each epoch


    savehist("hist_epoch10.json")  # Save the history after 10 epochs
    return canvas

#flower = cv2.imread("flower.jpg")
def oil_main(flower ,itr):
	load(flower)
# Start the automated painting for 10 epochs
	canvas=	run_for_epochs(epochs=itr)
	return canvas
#canvas= oil_main(flower)
#canvas= cv2.resize(canvas, (512, 512))
#cv2.imshow(f"Epoch", canvas)  # Show the current state of the canvas
#cv2.waitKey(0)  # Wait indefinitely until a key is pressed
#cv2.destroyAllWindows() 
