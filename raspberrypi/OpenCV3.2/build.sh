PKG=opencv_3.2-1

mkdir -p $PKG/DEBIAN
cp control $PKG/DEBIAN/control
dpkg-deb --build $PKG
