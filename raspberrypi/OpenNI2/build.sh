#! /bin/bash
SRC=OpenNI-Linux-Arm-2.2
PKG=openni_2.2-1
LIBDIR=$PKG/usr/lib
mkdir $PKG
mkdir -p $PKG/usr/include/ni2
mkdir -p $LIBDIR
cp -a $SRC/Include/* $PKG/usr/include/ni2
cp $SRC/Redist/libOpenNI2.jni.so $LIBDIR/libOpenNI2.jni.so
cp $SRC/Redist/libOpenNI2.so $LIBDIR/libOpenNI2.so

mkdir -p $LIBDIR/OpenNI2/Drivers
cp $SRC/Redist/OpenNI2/Drivers/libOniFile.so $LIBDIR/OpenNI2/Drivers/libOniFile.so
cp $SRC/Redist/OpenNI2/Drivers/libPS1080.so $LIBDIR/OpenNI2/Drivers/libPS1080.so
cp $SRC/Redist/OpenNI2/Drivers/libPSLink.so $LIBDIR/OpenNI2/Drivers/libPSLink.so
mkdir -p $PKG/etc/udev/rules.d
cp $SRC/primesense-usb.rules $PKG/etc/udev/rules.d/557-primesense-usb.rules

#mkdir -p $PKG/usr/share/OpenNI2-doc
#cp -a $SRC/Documentation $PKG/usr/share/OpenNI2-doc/

# todo: where should this really be put?
mkdir -p $PKG/usr/lib/java
cp $SRC/Redist/org.openni.jar $PKG/usr/lib/java/OpenNI2.jar

mkdir -p $PKG/DEBIAN/
cp control $PKG/DEBIAN/control
dpkg-deb --build $PKG
