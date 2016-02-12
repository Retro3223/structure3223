PKG=mjpg-streamer-1.0
# assumes youve pulled down the repo and run
# make
# make install DESTDIR=mjpg-streamer-1.0

mkdir -p $PKG/DEBIAN
cp control $PKG/DEBIAN/control
dpkg-deb --build $PKG
