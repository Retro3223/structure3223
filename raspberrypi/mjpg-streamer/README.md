hmm. 

Roughly, I grabbed https://github.com/jacksonliam/mjpg-streamer and ran


make

then created directory mjpg-streamer-1.0
AND created directories bin and lib inside it because the makefile is dumb and doesnt, then 

make install DESTDIR=mjpg-streamer-1.0

then went into the input_file plugin and ran make in it and copied the so over to the DESTDIR manually, then ran

bash build.sh

and then slapped the startup scripts (which I mostly copied from robotpy's mjpg-streamer opkg) in and ran

sudo update-rc.d mjpg-streamer defaults

If I were less lazy, I might incorporate the last parts into the packaging..
