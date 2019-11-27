sudo vim /usr/share/alsa/alsa.conf
	#defaults.ctl.card (usbdevicenum)
	#defaults.pcm.card (usbdevicenum)
sudo vim /etc/asound.conf

{ctl,pcm}.!default {
	type hw
	card (num)
}

aplay -l

lsusb
