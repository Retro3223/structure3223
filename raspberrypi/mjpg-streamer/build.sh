PKG=mjpg-streamer-1.0

mkdir -p $PKG/DEBIAN
mkdir -p $PKG/etc/default
mkdir -p $PKG/etc/init.d
cp etc/default/* $PKG/etc/default
cp etc/init.d/* $PKG/etc/init.d
cp control $PKG/DEBIAN/control
cp conffiles $PKG/DEBIAN/conffiles
dpkg-deb --build $PKG
