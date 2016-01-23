import structure3223
import cv2


def main():
    depth, ir = structure3223.read_frame()
    cv2.imwrite("depth.jpeg", depth)
    cv2.imwrite("ir.jpeg", ir)


if __name__ == '__main__':
    structure3223.init()
    main()
    structure3223.destroy()
