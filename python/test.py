import structure3223
import numpy
import cv2


def munge_floats_to_img(xyz):
    img = numpy.empty(shape=(240,320,3), dtype='uint8')
    for i in range(0, 3):
        img[:,:,i] = xyz[i,:,:] * 255. / xyz[i, :,:].max()
    print (img.shape)
    return img

def munge_thetas_to_img(thetas):
    img = numpy.empty(shape=(240,320), dtype='uint8')
    img[:,:] = thetas * 255. / thetas.max()
    print (img)
    return img


def main():
    depth = numpy.empty(shape=(240, 320), dtype='uint16')
    ir = numpy.empty(shape=(240, 320), dtype='uint16')
    xyz = numpy.empty(shape=(3, 240, 320), dtype='float32')
    thetas = numpy.zeros(shape=(240, 320), dtype='float32')
    structure3223.read_frame(depth=depth, ir=ir)
    structure3223.depth_to_xyz(depth=depth, xyz=xyz)
    print("x mid: \n", xyz[0, 120:122, 159:162])
    print("y mid: \n", xyz[1, 120:122, 159:162])
    print("z mid: \n", xyz[2, 120:122, 159:162])
    structure3223.xyz_to_theta(xyz=xyz, theta=thetas)
    print("t mid: \n", thetas[120:122, 160:162])
    print("theta[121, 161] = %f", thetas[121, 161])
    print(" depends on x[160,
    cv2.imwrite("depth.jpeg", depth)
    cv2.imwrite("ir.jpeg", ir)
    cv2.imwrite("xyz.jpeg", munge_floats_to_img(xyz))
    cv2.imwrite("theta.jpeg", munge_thetas_to_img(thetas))


if __name__ == '__main__':
    structure3223.init()
    main()
    structure3223.destroy()
