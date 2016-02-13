hmm. 

Roughly, I grabbed https://github.com/jacksonliam/mjpg-streamer and ran


make
make install DESTDIR=mjpg-streamer-1.0
bash build.sh

and then slapped the startup scripts (which I mostly copied from robotpy's mjpg-streamer opkg) in and ran

sudo update-rc.d mjpg-streamer defaults

If I were less lazy, I might incorporate the last parts into the packaging..
