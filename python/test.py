import structure3223
import numpy
import cv2


def main():
    depth = numpy.empty(shape=(240, 320), dtype='uint16')
    ir = numpy.empty(shape=(240, 320), dtype='uint16')
    structure3223.read_frame(depth=depth, ir=ir)
    cv2.imwrite("depth.jpeg", depth)
    cv2.imwrite("ir.jpeg", ir)


if __name__ == '__main__':
    structure3223.init()
    main()
    structure3223.destroy()
