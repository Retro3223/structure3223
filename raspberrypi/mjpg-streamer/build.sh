PKG=mjpg-streamer-1.0

mkdir -p $PKG/DEBIAN
cp control $PKG/DEBIAN/control
dpkg-deb --build $PKG
